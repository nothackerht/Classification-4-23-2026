from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import sys
import re
import json
import argparse
import traceback
import warnings
import gc
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
# --- parallel config (define before numpy/sklearn imports) ---
PHYS_CORES = int(os.environ.get("CV_PHYS_CORES", os.cpu_count() or 8))
THREADS_PER_WORKER = int(os.environ.get("CV_THREADS_PER_WORKER", "1"))
N_PROC = max(1, PHYS_CORES // max(1, THREADS_PER_WORKER))

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_WORKER)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_WORKER)

# Safety cap
MAX_PROC_ENV = os.environ.get("CV_MAX_PROC", "").strip()
if MAX_PROC_ENV:
    try:
        N_PROC = min(N_PROC, int(MAX_PROC_ENV))
    except ValueError:
        pass
else:
    N_PROC = min(N_PROC, 6)

# ---------- MEMORY OPTIMIZATION ----------
# Use float32 globally to halve memory footprint
GLOBAL_DTYPE = np.float32

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
_log = logging.getLogger(__name__)

def _flush():
    """Flush stdout/stderr so interleaved output doesn't get lost."""
    sys.stdout.flush()
    sys.stderr.flush()

print(f"[PARALLEL] PHYS_CORES={PHYS_CORES}, N_PROC={N_PROC}")
_flush()


import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error, explained_variance_score,
)
from sklearn.inspection import permutation_importance

# Local modules
from data_loader import load_data
from preprocessing import Preprocessing

# Try importing PreprocessingNew for CORAL scenarios
try:
    from preprocessing_new import PreprocessingNew, coral_align_target_to_source
    HAS_PREPROCESSING_NEW = True
except ImportError:
    HAS_PREPROCESSING_NEW = False
    warnings.warn("preprocessing_new not found; CORAL scenarios will be skipped.")

from plot_parity import parity_plot_sample_level, save_outer_predictions_excel


# ==================== CONSTANTS ====================

REPS = 9  # default spectra per sample (before QC)

TARGETS = [
    ("Splicing Index", "target_SI"),
    ("Hand Grip Strength (%)", "HGS_pp_avg"),
    ("Average Ankle Dorsiflexion (%)", "ADF_pp_avg"),
]

PREPROCESS_GRID = [
    [], ["EMSC"], ["Normalization"], ["SNV"], ["Second Derivative"],
    ["EMSC", "Normalization"], ["EMSC", "SNV"], ["EMSC", "Second Derivative"],
    ["Normalization", "EMSC"], ["Normalization", "SNV"], ["Normalization", "Second Derivative"],
    ["SNV", "EMSC"], ["SNV", "Normalization"], ["SNV", "Second Derivative"],
    ["Second Derivative", "EMSC"], ["Second Derivative", "Normalization"], ["Second Derivative", "SNV"],
]

PLS_COMPONENTS = list(range(2, 19))
IPLS_COMPONENTS = list(range(2, 11))
IPLS_INTERVALS = [5, 10, 15]

SCENARIOS = ["baseline", "qc_only", "coral_only", "qc_coral"]


# ==================== FILENAME / ID HELPERS ====================

def _sample_id_from_filename(fname: str) -> str:
    s = os.path.basename(str(fname))
    return s.split("_Map")[0]


def _normalize_qc_filename(fname: str) -> str:
    base = os.path.basename(str(fname)).strip()
    base = os.path.splitext(base)[0]
    if "_Map" in base:
        parts = base.split("_Map")
        return parts[0] + "_Map" + parts[1][0]
    raise ValueError(f"Cannot parse Map index from filename: {fname}")


# ==================== METRIC HELPERS ====================

def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":    float(r2_score(y_true, y_pred)),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "evs":   float(explained_variance_score(y_true, y_pred)),
    }


# ==================== SAMPLE COLLAPSING ====================

def collapse_mean_by_groups(y_spec: np.ndarray, groups: np.ndarray):
    """Collapse spectrum-level values to sample-level means using integer group ids."""
    y = np.asarray(y_spec).ravel()
    g = np.asarray(groups).astype(int).ravel()
    ug = np.unique(g)
    means = np.array([y[g == gi].mean() for gi in ug])
    return means, ug


def collapse_mean_std_n_by_sample_ids(y_spec, sample_ids):
    """Collapse spectrum-level predictions to per-sample mean/std/n using string SampleIDs."""
    y = np.asarray(y_spec).ravel()
    sids = np.asarray(sample_ids).astype(str).ravel()
    df = pd.DataFrame({"sid": sids, "y": y})
    g = df.groupby("sid", sort=False)["y"]
    sids_u = list(g.groups.keys())
    means = g.mean().loc[sids_u].values
    stds = g.std(ddof=0).fillna(0).loc[sids_u].values
    ns = g.count().loc[sids_u].values.astype(int)
    return means, stds, ns, np.array(sids_u, dtype=str)


def _sample_means_stds(y_like: np.ndarray, reps: int = REPS):
    """Collapse fixed-rep spectra to sample level."""
    y = np.asarray(y_like)
    assert y.shape[0] % reps == 0
    N = y.shape[0] // reps
    if y.ndim == 1:
        y3 = y.reshape(N, reps)
        return y3.mean(axis=1), y3.std(axis=1)
    y3 = y.reshape(N, reps, -1)
    return y3.mean(axis=1), y3.std(axis=1)


# ==================== TORCH REGRESSOR ====================

class TorchRegressor:
    def __init__(self, hidden_layer_sizes=(256, 128), activation="relu",
                 alpha=1e-4, learning_rate_init=1e-3, batch_size=64,
                 max_iter=200, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def _build(self, n_features):
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}.get(self.activation, nn.ReLU)
        layers = []
        last = n_features
        for h in self.hidden_layer_sizes:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.model_ = self._build(X.shape[1]).to(self.device)
        opt = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        loss_fn = nn.MSELoss()

        idx = np.arange(X.shape[0])
        for _ in range(self.max_iter):
            rng.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                bi = idx[s:s + self.batch_size]
                xb = torch.from_numpy(X[bi]).to(self.device)
                yb = torch.from_numpy(y[bi]).to(self.device)
                opt.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        xb = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            return self.model_(xb).cpu().numpy().ravel()


# ==================== PREPROCESSING ====================

def _fit_transform_pair_basic(Xtr, Xte, methods):
    """Train-only preprocessing using the basic Preprocessing class (no CORAL)."""
    if not methods:
        return Xtr.astype(np.float32), Xte.astype(np.float32)
    # Work in (features, spectra) space
    Xtr_fxN = Xtr.T.copy()
    Xte_fxN = Xte.T.copy()
    for m in methods:
        if m == "EMSC":
            ref = np.mean(Xtr_fxN, axis=1)
            Xtr_fxN = Preprocessing(Xtr_fxN).emsc(Xtr_fxN, reference=ref)
            Xte_fxN = Preprocessing(Xte_fxN).emsc(Xte_fxN, reference=ref)
        elif m == "Normalization":
            Xtr_fxN = Preprocessing(Xtr_fxN).normalize_spectrum(Xtr_fxN)
            Xte_fxN = Preprocessing(Xte_fxN).normalize_spectrum(Xte_fxN)
        elif m == "SNV":
            Xtr_fxN = Preprocessing(Xtr_fxN).snv(Xtr_fxN)
            Xte_fxN = Preprocessing(Xte_fxN).snv(Xte_fxN)
        elif m == "Second Derivative":
            Xtr_fxN = Preprocessing(Xtr_fxN).second_derivative(Xtr_fxN)
            Xte_fxN = Preprocessing(Xte_fxN).second_derivative(Xte_fxN)
    return Xtr_fxN.T.astype(np.float32), Xte_fxN.T.astype(np.float32)


def fit_apply_preprocessing_coral(Xtr_2D, Xte_2D, methods, tr_groups=None, te_sample_ids=None):
    """Preprocessing with CORAL alignment (requires preprocessing_new)."""
    if not HAS_PREPROCESSING_NEW:
        raise RuntimeError("CORAL requires preprocessing_new module.")

    pp = PreprocessingNew()
    Xtr_fxN = Xtr_2D.T
    Xte_fxN = Xte_2D.T
    pp.fit(Xtr_fxN, methods, emsc_ref_mode="train_mean")
    Xtr_p = pp.transform(Xtr_fxN, methods).T
    Xte_p = pp.transform(Xte_fxN, methods).T

    # Sample-weighted CORAL alignment
    if tr_groups is not None and te_sample_ids is not None:
        Xte_p = _coral_align_sample_weighted(Xtr_p, Xte_p, tr_groups, te_sample_ids)

    return Xtr_p.astype(np.float32), Xte_p.astype(np.float32)


def _coral_align_sample_weighted(X_source, X_target, source_groups, target_sample_ids, reg=1e-6):
    """CORAL alignment with sample-weighted covariance."""
    # Per-sample means for source
    g = np.asarray(source_groups).astype(int).ravel()
    ug = np.unique(g)
    Xs = np.array([X_source[g == gi].mean(axis=0) for gi in ug])

    # Per-sample means for target
    sids = np.asarray(target_sample_ids).astype(str)
    seen = {}
    for sid in sids:
        if sid not in seen:
            seen[sid] = True
    uniq = list(seen.keys())
    Xt = np.array([X_target[sids == sid].mean(axis=0) for sid in uniq])

    ms = Xs.mean(axis=0, keepdims=True)
    mt = Xt.mean(axis=0, keepdims=True)
    Xs0 = Xs - ms
    Xt0 = Xt - mt

    d = Xs0.shape[1]
    Cs = (Xs0.T @ Xs0) / max(1, Xs0.shape[0] - 1) + reg * np.eye(d)
    Ct = (Xt0.T @ Xt0) / max(1, Xt0.shape[0] - 1) + reg * np.eye(d)

    def _inv_sqrtm(M):
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-12, None)
        return (V * (1.0 / np.sqrt(w))) @ V.T

    def _sqrtm(M):
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-12, None)
        return (V * np.sqrt(w)) @ V.T

    A = _inv_sqrtm(Ct) @ _sqrtm(Cs)
    return (X_target - mt) @ A + ms


# ==================== iPLS HELPERS ====================

def _make_intervals(n_features: int, num_intervals: int) -> List[Tuple[int, int]]:
    if num_intervals < 2:
        return [(0, n_features)]
    edges = np.linspace(0, n_features, num_intervals + 1).round().astype(int)
    intervals = []
    for i in range(num_intervals):
        a, b = int(edges[i]), int(edges[i + 1])
        if b > a:
            intervals.append((a, b))
    return intervals or [(0, n_features)]


def _best_interval_grouped(X_tr, Y_tr, groups_tr, preprocess_methods, n_components,
                            num_intervals, reps=REPS, n_splits=5, use_coral=False):
    """Choose best interval by grouped CV RMSE on train. Returns ((a,b), best_mse)."""
    intervals = _make_intervals(X_tr.shape[1], num_intervals)
    inner = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_tr))))
    interval_mse = []

    for (a, b) in intervals:
        fold_mse = []
        for iti, ivo in inner.split(X_tr, groups=groups_tr):
            Xt_i_tr, Xt_i_vl = X_tr[iti][:, a:b], X_tr[ivo][:, a:b]
            Yt_i_tr, Yt_i_vl = Y_tr[iti], Y_tr[ivo]
            # Preprocessing always without CORAL during selection
            Xt_tr_p, Xt_vl_p = _fit_transform_pair_basic(Xt_i_tr, Xt_i_vl, preprocess_methods)
            pls = PLSRegression(n_components=int(n_components))
            pls.fit(Xt_tr_p, Yt_i_tr)
            Y_vl_hat = pls.predict(Xt_vl_p)

            g_vl = groups_tr[ivo]
            Y_vl_s, _ = collapse_mean_by_groups(Yt_i_vl.ravel(), g_vl)
            Y_hat_s, _ = collapse_mean_by_groups(Y_vl_hat.ravel(), g_vl)
            fold_mse.append(mean_squared_error(Y_vl_s, Y_hat_s))

        interval_mse.append(float(np.mean(fold_mse)))

    best_idx = int(np.argmin(interval_mse))
    return intervals[best_idx], float(interval_mse[best_idx])


# ==================== HP TAG HELPERS ====================

def _svr_tag(hp):
    return f"C={hp.get('C','?')}_eps={hp.get('epsilon','?')}_kern={hp.get('kernel','?')}_gam={hp.get('gamma','?')}"

def _ac_tag(hp):
    h = hp.get("hidden_layer_sizes", (128,))
    return f"h={'x'.join(map(str,h))}_lr={hp.get('learning_rate_init','?')}_a={hp.get('alpha','?')}"


# ==================== SIMPLICITY COMPARATOR ====================

def _is_simpler(model_a, hp_a, model_b, hp_b):
    order = {"pls": 0, "ipls": 1, "svr": 2, "acfnn": 3}
    ma, mb = model_a.lower(), model_b.lower()
    if ma != mb:
        return order.get(ma, 99) < order.get(mb, 99)
    if ma == "pls":
        return int(hp_a.get("n_components", 99)) < int(hp_b.get("n_components", 99))
    if ma == "ipls":
        return (int(hp_a.get("num_intervals", 99)), int(hp_a.get("n_components", 99))) < \
               (int(hp_b.get("num_intervals", 99)), int(hp_b.get("n_components", 99)))
    if ma == "svr":
        return float(hp_a.get("C", 999)) < float(hp_b.get("C", 999))
    if ma == "acfnn":
        ha = hp_a.get("hidden_layer_sizes", (128,))
        hb = hp_b.get("hidden_layer_sizes", (128,))
        return (sum(ha), len(ha)) < (sum(hb), len(hb))
    return False


# ==================== FEATURE IMPORTANCE ====================

def compute_vip(pls_model, X, Y):
    """Compute VIP scores for fitted PLS model."""
    T = pls_model.x_scores_       # (n, n_comp)
    W = pls_model.x_weights_      # (p, n_comp)
    Q = pls_model.y_loadings_     # (1, n_comp) or (k, n_comp)

    p, h = W.shape
    # Fraction of Y variance explained by each component
    SS = np.zeros(h)
    for a in range(h):
        ta = T[:, a:a + 1]
        qa = Q[:, a:a + 1] if Q.ndim == 2 else Q[a:a + 1]
        SS[a] = float(np.asarray(ta.T @ ta).item() * np.asarray(qa.T @ qa).item())

    SS_total = SS.sum()
    if SS_total < 1e-12:
        return np.ones(p)

    vip = np.zeros(p)
    for j in range(p):
        s = 0.0
        for a in range(h):
            s += (W[j, a] ** 2) * SS[a]
        vip[j] = np.sqrt(p * s / SS_total)
    return vip


def compute_permutation_importance_custom(model, X, y, n_repeats=5, seed=42):
    """Permutation importance — works on a private copy so it is thread-safe."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).ravel()
    # CRITICAL: work on a copy so we never mutate the caller's array.
    # Under the threading backend multiple threads could share underlying
    # memory; in-place mutation caused silent data corruption and crashes.
    X_work = X.copy()
    base_mse = mean_squared_error(y, model.predict(X_work).ravel())
    importances = np.zeros(X_work.shape[1])

    for j in range(X_work.shape[1]):
        col_orig = X_work[:, j].copy()
        scores = 0.0
        for _ in range(n_repeats):
            X_work[:, j] = rng.permutation(col_orig)
            perm_mse = mean_squared_error(y, model.predict(X_work).ravel())
            scores += (perm_mse - base_mse)
        X_work[:, j] = col_orig  # restore
        importances[j] = scores / n_repeats

    return importances


# ==================== QC FILTERING ====================

def apply_qc_filtering(all_spectra, fn_list, rep_list, qc_flags_xlsx, box_label):
    """
    Apply QC filtering to spectra array based on QC_flags.xlsx.
    Returns: filtered spectra, filtered filenames, filtered rep_indices, kept_column_indices
    """
    qc = pd.read_excel(qc_flags_xlsx, sheet_name="per_spectrum")
    if "box" not in qc.columns:
        raise ValueError("QC_flags.xlsx must contain 'box' column.")
    qc["box"] = qc["box"].astype(str).str.strip()
    qc_box = qc[qc["box"] == box_label].copy()
    if qc_box.empty:
        raise ValueError(f"No QC rows for box='{box_label}'.")

    keep_map = _build_keep_by_file_rep(qc_box)

    keys = _file_rep_keys(fn_list, rep_list)
    missing = [k for k in keys if k not in keep_map]
    if missing:
        raise KeyError(f"QC missing ({box_label}). Example: {missing[0]}")

    keep_cols = [i for i, k in enumerate(keys) if keep_map[k]]
    return (
        all_spectra[:, keep_cols],
        [fn_list[i] for i in keep_cols],
        [rep_list[i] for i in keep_cols],
        keep_cols,
    )


def _build_keep_by_file_rep(qc_box_df):
    if "filename" not in qc_box_df.columns or "keep" not in qc_box_df.columns:
        raise ValueError("QC needs 'filename' and 'keep' columns.")
    rep_col = None
    for c in ["rep", "Rep", "rep_idx", "rep_index", "spectrum_rep"]:
        if c in qc_box_df.columns:
            rep_col = c
            break
    if rep_col is None:
        raise ValueError("QC needs rep column.")

    q = qc_box_df.copy()
    q["filename"] = q["filename"].apply(_normalize_qc_filename)
    q["_rep_idx"] = pd.to_numeric(q[rep_col], errors="coerce").astype(int)
    vals = set(q["_rep_idx"].unique())
    if vals.issubset({1, 2, 3}):
        q["_rep_idx"] -= 1
    return {(row["filename"], int(row["_rep_idx"])): bool(row["keep"]) for _, row in q.iterrows()}


def _file_rep_keys(fn_list, rep_list):
    keys = []
    for f, r in zip(fn_list, rep_list):
        norm = _normalize_qc_filename(f)
        keys.append((norm, int(r)))
    return keys


# ==================== FOLD ITERATOR ====================

def _iter_sample_folds(n_samples, n_splits, seed, strat_labels=None):
    sample_ids = np.arange(n_samples, dtype=int)
    use_strat = False
    if strat_labels is not None:
        labs = np.asarray(strat_labels)
        if labs.shape[0] == n_samples and len(np.unique(labs)) >= 2:
            counts = {c: (labs == c).sum() for c in np.unique(labs)}
            if min(counts.values()) >= n_splits:
                use_strat = True
    if use_strat:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(sample_ids, strat_labels):
            yield sample_ids[tr], sample_ids[te]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(sample_ids):
            yield sample_ids[tr], sample_ids[te]


# ==================== GLOBAL GROUPED DATA CONTAINER ====================

@dataclass
class GlobalGrouped:
    """Holds all spectra + meta. QC-safe (variable spectra per sample)."""
    all_spectra_fxN: np.ndarray    # (features, n_spectra)
    meta: pd.DataFrame             # per-sample rows
    spec_sample_ids: np.ndarray    # (n_spectra,) SampleID per column
    reps: int = REPS               # default (may vary with QC)
    is_qc: bool = False            # True if QC was applied

    def build_xy(self, target_col: str, include_types=("DM1", "Control")):
        meta0 = self.meta.copy()
        # Normalize SampleID
        if "SampleID" not in meta0.columns:
            if meta0.index.name == "SampleID":
                meta0 = meta0.reset_index()
            elif "Sample_ID" in meta0.columns:
                meta0 = meta0.rename(columns={"Sample_ID": "SampleID"})
        meta0["SampleID"] = meta0["SampleID"].astype(str)

        meta0 = meta0[meta0["Type"].isin(include_types)].copy()

        # Set control defaults
        if "Control" in include_types:
            for col, val in [("target_SI", 0), ("HGS_pp_avg", 100), ("ADF_pp_avg", 100)]:
                if col in meta0.columns:
                    meta0.loc[meta0["Type"] == "Control", col] = val

        meta0 = meta0.dropna(subset=[target_col]).copy().reset_index(drop=True)
        meta_sids = meta0["SampleID"].to_numpy(dtype=str)

        # Filter spectra to only those belonging to kept samples
        spec_sids = np.asarray(self.spec_sample_ids).astype(str)
        mask_spec = np.isin(spec_sids, meta_sids)
        present = np.unique(spec_sids[mask_spec])
        meta0 = meta0[meta0["SampleID"].isin(present)].copy().reset_index(drop=True)
        meta_sids = meta0["SampleID"].to_numpy(dtype=str)

        mask_spec = np.isin(spec_sids, meta_sids)
        X_all = self.all_spectra_fxN[:, mask_spec].T  # (n_spec, features)

        sid_to_i = {sid: i for i, sid in enumerate(meta_sids)}
        groups = np.array([sid_to_i[sid] for sid in spec_sids[mask_spec]], dtype=int)

        y_s = meta0[target_col].to_numpy(dtype=float)
        Y = y_s[groups]  # spectrum-level y

        return X_all, Y, y_s, groups, meta0


# ==================== NESTED CV SELECTOR ====================

def nested_select_one(model_name, methods, target_col, n_outer, n_inner, seed,
                      rm_tr, include_types, pls_components, ipls_components,
                      ipls_intervals, svr_grid, acfnn_grid, use_coral=False, wavenumbers=None,
                      skip_importance=False, perm_repeats=3):
    """
    Nested CV on TRAIN. Returns selection + CV metrics + per-fold importance vectors.
    """
    X, y_spec, y_s, groups, meta_kept = rm_tr.build_xy(target_col, include_types=include_types)
    N = len(meta_kept)
    if N < 2:
        raise ValueError(f"Too few training samples (N={N}) for target '{target_col}'.")

    # Strat labels for controls
    strat_labels = None
    if "Type" in meta_kept.columns and meta_kept["Type"].isin(["DM1", "Control"]).any():
        strat_labels = (meta_kept["Type"].to_numpy() == "DM1").astype(int)

    # HP candidates
    if model_name == "pls":
        hp_candidates = [{"n_components": int(c)} for c in pls_components]
    elif model_name == "ipls":
        hp_candidates = [{"n_components": int(c), "num_intervals": int(I)}
                         for c in ipls_components for I in ipls_intervals]
    elif model_name == "svr":
        hp_candidates = svr_grid
    elif model_name == "acfnn":
        hp_candidates = acfnn_grid
    else:
        raise ValueError(model_name)

    # Resolve SampleIDs for each group index so we can save them with predictions
    meta_sids = meta_kept["SampleID"].to_numpy(dtype=str)
    group_to_sid = {i: meta_sids[i] for i in range(N)}

    n_outer = min(n_outer, N)
    fold_idx = 0
    rmse_folds = []
    r2_folds = []
    inner_trace = []
    y_true_all, y_pred_all = [], []
    y_pred_std_all = []                  # per-sample prediction std (across 9 spectra)
    sid_all = []                        # SampleIDs aligned to y_true_all / y_pred_all
    fold_importances = []

    for tr_s, te_s in _iter_sample_folds(N, n_outer, seed, strat_labels=strat_labels):
        fold_idx += 1
        mask_tr = np.isin(groups, tr_s)
        mask_te = np.isin(groups, te_s)

        Xtr, Xte = X[mask_tr], X[mask_te]
        ytr, yte = y_spec[mask_tr], y_spec[mask_te]
        gtr, gte = groups[mask_tr], groups[mask_te]

        # Inner CV for HP selection
        n_inner_eff = min(n_inner, max(2, len(np.unique(tr_s))))
        strat_labels_tr = strat_labels[tr_s] if strat_labels is not None else None

        # --- Precompute preprocessing for each inner fold (cache once, reuse across HPs) ---
        inner_folds_data = []
        for itr_s, iva_s in _iter_sample_folds(len(tr_s), n_inner_eff, seed + 17, strat_labels=strat_labels_tr):
            inner_tr_samples = tr_s[itr_s]
            inner_va_samples = tr_s[iva_s]
            m_tr_i = np.isin(groups, inner_tr_samples)
            m_va_i = np.isin(groups, inner_va_samples)

            Xitr, Xiva = X[m_tr_i], X[m_va_i]
            yitr, yiva = y_spec[m_tr_i], y_spec[m_va_i]
            giva = groups[m_va_i]

            # Preprocess ONCE for full-spectrum models (PLS, SVR, ACFNN)
            if model_name in ("pls", "svr", "acfnn"):
                Xt_tr, Xt_va = _fit_transform_pair_basic(Xitr, Xiva, methods)
            else:
                Xt_tr, Xt_va = None, None  # iPLS does its own per-interval

            inner_folds_data.append({
                "Xitr": Xitr, "Xiva": Xiva,
                "Xt_tr": Xt_tr, "Xt_va": Xt_va,
                "yitr": yitr, "yiva": yiva, "giva": giva,
                "gitr": groups[m_tr_i],
            })

        # --- Also pre-cache iPLS interval preprocessing across all (num_intervals, fold, interval) ---
        _ipls_preproc_cache = {}
        if model_name == "ipls":
            seen_intervals = set(int(hp["num_intervals"]) for hp in hp_candidates)
            for num_iv in seen_intervals:
                intervals = _make_intervals(X.shape[1], num_iv)
                for fi, fd in enumerate(inner_folds_data):
                    for a, b in intervals:
                        key = (num_iv, fi, a, b)
                        _ipls_preproc_cache[key] = _fit_transform_pair_basic(
                            fd["Xitr"][:, a:b], fd["Xiva"][:, a:b], methods)

        def score_hp(hp, inner_folds_data, _ipls_preproc_cache):
            mse_folds_inner = []
            for fi, fd in enumerate(inner_folds_data):
                if model_name == "pls":
                    mdl = PLSRegression(n_components=int(hp["n_components"]))
                    mdl.fit(fd["Xt_tr"], fd["yitr"].reshape(-1, 1))
                    yhat = mdl.predict(fd["Xt_va"]).ravel()
        
                elif model_name == "svr":
                    scaler = StandardScaler()
                    Xt_tr_sc = scaler.fit_transform(fd["Xt_tr"])
                    Xt_va_sc = scaler.transform(fd["Xt_va"])
                    hp_clean = {k: v for k, v in hp.items() if k != "_target_col"}
                    mdl = SVR(**hp_clean)
                    mdl.fit(Xt_tr_sc, fd["yitr"].ravel())
                    yhat = mdl.predict(Xt_va_sc)
        
                elif model_name == "acfnn":
                    mdl = TorchRegressor(random_state=seed, **hp)
                    mdl.fit(fd["Xt_tr"], fd["yitr"].ravel())
                    yhat = mdl.predict(fd["Xt_va"])
        
                elif model_name == "ipls":
                    intervals = _make_intervals(fd["Xitr"].shape[1], int(hp["num_intervals"]))
                    best_rmse_iv = np.inf
                    best_yhat = None
                    for a, b in intervals:
                        Xt_tr, Xt_va = _ipls_preproc_cache[(int(hp["num_intervals"]), fi, a, b)]
                        mdl = PLSRegression(n_components=int(hp["n_components"]))
                        mdl.fit(Xt_tr, fd["yitr"].reshape(-1, 1))
                        yh = mdl.predict(Xt_va).ravel()
                        yh_s, _ = collapse_mean_by_groups(yh, fd["giva"])
                        yiva_s, _ = collapse_mean_by_groups(fd["yiva"], fd["giva"])
                        r = float(np.sqrt(mean_squared_error(yiva_s, yh_s)))
                        if r < best_rmse_iv:
                            best_rmse_iv = r
                            best_yhat = yh
                    yhat = best_yhat
        
                else:
                    raise ValueError(model_name)
        
                yhat_s, _ = collapse_mean_by_groups(yhat, fd["giva"])
                yiva_s, _ = collapse_mean_by_groups(fd["yiva"], fd["giva"])
                mse_folds_inner.append(mean_squared_error(yiva_s, yhat_s))
        
            return float(np.mean(mse_folds_inner)), float(np.std(mse_folds_inner))
        
        hp_stats = [score_hp(hp, inner_folds_data, _ipls_preproc_cache) for hp in hp_candidates]
        mus = [m for (m, s) in hp_stats]
        best_ix = int(np.argmin(mus))
        pick_hp = hp_candidates[best_ix]

        # Free inner fold cached data to reclaim memory before outer fold eval
        del inner_folds_data
        _ipls_preproc_cache.clear()
        gc.collect()

        # ---- Outer fold evaluation + importance extraction ----
        picked_interval = None
        if model_name == "ipls":
            (a, b), _ = _best_interval_grouped(
                Xtr, ytr.reshape(-1, 1), gtr, methods,
                int(pick_hp["n_components"]), int(pick_hp["num_intervals"]),
                n_splits=max(3, min(5, len(np.unique(tr_s))))
            )
            picked_interval = (a, b)
            Xt_tr, Xt_te = _fit_transform_pair_basic(Xtr[:, a:b], Xte[:, a:b], methods)
            mdl = PLSRegression(n_components=int(pick_hp["n_components"]))
            mdl.fit(Xt_tr, ytr.reshape(-1, 1))
            yhat_te = mdl.predict(Xt_te).ravel()

            n_feat = X.shape[1]
            imp = np.zeros(n_feat)
            imp[a:b] = compute_vip(mdl, Xt_tr, ytr.reshape(-1, 1))
            fold_importances.append(imp)

        elif model_name == "pls":
            Xt_tr, Xt_te = _fit_transform_pair_basic(Xtr, Xte, methods)
            mdl = PLSRegression(n_components=int(pick_hp["n_components"]))
            mdl.fit(Xt_tr, ytr.reshape(-1, 1))
            yhat_te = mdl.predict(Xt_te).ravel()
            fold_importances.append(compute_vip(mdl, Xt_tr, ytr.reshape(-1, 1)))

        elif model_name == "svr":
            Xt_tr, Xt_te = _fit_transform_pair_basic(Xtr, Xte, methods)
            scaler = StandardScaler()
            Xt_tr_sc = scaler.fit_transform(Xt_tr)
            Xt_te_sc = scaler.transform(Xt_te)
            hp_clean = {k: v for k, v in pick_hp.items() if k != "_target_col"}
            mdl = SVR(**hp_clean)
            mdl.fit(Xt_tr_sc, ytr.ravel())
            yhat_te = mdl.predict(Xt_te_sc)
            if not skip_importance:
                class _ScaledSVR:
                    """Wraps SVR + scaler so permutation importance can call .predict(X_unscaled)."""
                    def __init__(self, _mdl, _scaler):
                        self._mdl = _mdl
                        self._scaler = _scaler
                    def predict(self, X):
                        return self._mdl.predict(self._scaler.transform(X))
                imp = compute_permutation_importance_custom(
                    _ScaledSVR(mdl, scaler),
                    Xt_tr, ytr, n_repeats=perm_repeats, seed=seed)
                fold_importances.append(imp)

        elif model_name == "acfnn":
            Xt_tr, Xt_te = _fit_transform_pair_basic(Xtr, Xte, methods)
            mdl = TorchRegressor(random_state=seed, **pick_hp)
            mdl.fit(Xt_tr, ytr.ravel())
            yhat_te = mdl.predict(Xt_te)
            if not skip_importance:
                imp = compute_permutation_importance_custom(mdl, Xt_te, yte, n_repeats=perm_repeats, seed=seed)
                fold_importances.append(imp)
        else:
            raise ValueError(model_name)

        # Collapse to sample level for this fold
        yhat_s, _ = collapse_mean_by_groups(yhat_te, gte)
        yte_s, _ = collapse_mean_by_groups(yte, gte)
        fold_sids = np.array([group_to_sid[g] for g in np.unique(gte)])

        # Also compute per-sample std from spectra within each sample
        yhat_std_s = np.array([yhat_te[gte == g].std(ddof=0) for g in np.unique(gte)])

        fold_rmse = float(np.sqrt(mean_squared_error(yte_s, yhat_s)))
        fold_r2 = float(r2_score(yte_s, yhat_s)) if len(yte_s) > 1 else np.nan
        fold_mae = float(mean_absolute_error(yte_s, yhat_s))

        rmse_folds.append(fold_rmse)
        r2_folds.append(fold_r2)
        y_true_all.append(yte_s)
        y_pred_all.append(yhat_s)
        y_pred_std_all.append(yhat_std_s)
        sid_all.append(fold_sids)

        # Store rich inner trace for this fold
        inner_trace.append({
            "outer_fold": fold_idx,
            "selected_hp": pick_hp,
            "selected_hp_tag": str(pick_hp),
            "inner_best_mse_mean": mus[best_ix],
            "inner_best_mse_std": hp_stats[best_ix][1],
            "fold_rmse": fold_rmse,
            "fold_r2": fold_r2,
            "fold_mae": fold_mae,
            "n_train_samples": len(tr_s),
            "n_test_samples": len(te_s),
            "n_train_spectra": int(mask_tr.sum()),
            "n_test_spectra": int(mask_te.sum()),
            "ipls_interval": picked_interval,
            # Per-sample predictions for this fold (for reconstruction)
            "_fold_sids": fold_sids,
            "_fold_y_true": yte_s,
            "_fold_y_pred": yhat_s,
            "_fold_y_pred_std": yhat_std_s,
        })

    # Aggregate
    YT = np.concatenate(y_true_all).ravel()
    YP = np.concatenate(y_pred_all).ravel()
    YP_STD = np.concatenate(y_pred_std_all).ravel()
    SID_ALL = np.concatenate(sid_all).ravel()
    cv_mets_pooled = _metrics(YT, YP)
    cv_mu_rmse = float(np.mean(rmse_folds))
    cv_sd_rmse = float(np.std(rmse_folds))
    cv_mu_r2 = float(np.mean(r2_folds))
    cv_sd_r2 = float(np.std(r2_folds))

    # Modal HP selection across folds
    def _hp_key(hp):
        if model_name == "pls": return f"pls|C={hp['n_components']}"
        if model_name == "svr": return f"svr|{_svr_tag(hp)}"
        if model_name == "acfnn": return f"ac|{_ac_tag(hp)}"
        if model_name == "ipls": return f"ipls|C={hp['n_components']}__I={hp['num_intervals']}"
        return str(hp)

    counts = Counter([_hp_key(r["selected_hp"]) for r in inner_trace])
    modal_key, _ = counts.most_common(1)[0]
    sel_hp = next(r["selected_hp"] for r in inner_trace if _hp_key(r["selected_hp"]) == modal_key)

    # Average importance across folds
    imp_mean = None
    if fold_importances:
        max_len = max(imp.shape[0] for imp in fold_importances)
        padded = [np.pad(imp, (0, max_len - imp.shape[0])) if imp.shape[0] < max_len else imp
                  for imp in fold_importances]
        imp_mean = np.mean(np.vstack(padded), axis=0)

    return {
        "model": model_name,
        "methods": methods,
        "selected_hp": sel_hp,
        "cv_metrics_pooled": cv_mets_pooled,
        "cv_mu_rmse": cv_mu_rmse,
        "cv_sd_rmse": cv_sd_rmse,
        "cv_mu_r2": cv_mu_r2,
        "cv_sd_r2": cv_sd_r2,
        "rmse_folds": rmse_folds,
        "r2_folds": r2_folds,
        "inner_trace": inner_trace,
        "y_true_cv": YT,
        "y_pred_cv": YP,
        "y_pred_cv_std": YP_STD,
        "cv_sample_ids": SID_ALL,
        "importance_mean": imp_mean,
        "fold_importances": fold_importances,
        "n_train_samples": N,
    }


# ==================== EXTERNAL EVALUATION ====================

def run_external_eval(model_name, methods, sel_hp, rm_tr, include_types,
                      Xte_target, spec_sample_ids_te_target, y_true_by_sampleid,
                      use_coral, wavenumbers, random_state):
    """Fit on full train, predict on external test. Returns metrics + predictions."""
    Xtr_all, ytr_all_spec, ytr_all_s, groups_all, meta_kept = rm_tr.build_xy(
        sel_hp["_target_col"], include_types=include_types)

    preprocess_fn = fit_apply_preprocessing_coral if use_coral else _fit_transform_pair_basic

    # iPLS interval
    ipls_interval = None
    if model_name == "ipls":
        (a, b), _ = _best_interval_grouped(
            Xtr_all, ytr_all_spec.reshape(-1, 1), groups_all, methods,
            int(sel_hp["n_components"]), int(sel_hp["num_intervals"]),
            n_splits=max(3, min(5, len(np.unique(groups_all))))
        )
        ipls_interval = (a, b)
        Xtr_fit = Xtr_all[:, a:b]
        Xte_fit = Xte_target[:, a:b]
    else:
        Xtr_fit = Xtr_all
        Xte_fit = Xte_target

    if use_coral:
        Xt_tr, Xt_te = preprocess_fn(Xtr_fit, Xte_fit, methods,
                                      tr_groups=groups_all,
                                      te_sample_ids=spec_sample_ids_te_target)
    else:
        Xt_tr, Xt_te = preprocess_fn(Xtr_fit, Xte_fit, methods)

    # Train + predict
    if model_name == "pls":
        mdl = PLSRegression(n_components=int(sel_hp["n_components"]))
        mdl.fit(Xt_tr, ytr_all_spec.reshape(-1, 1))
        yhat_spec = mdl.predict(Xt_te).ravel()
        hp_tag = f"C={sel_hp['n_components']}"
    elif model_name == "svr":
        hp_clean = {k: v for k, v in sel_hp.items() if k != "_target_col"}
        scaler = StandardScaler()
        Xt_tr_sc = scaler.fit_transform(Xt_tr)
        Xt_te_sc = scaler.transform(Xt_te)
        mdl = SVR(**hp_clean)
        mdl.fit(Xt_tr_sc, ytr_all_spec.ravel())
        yhat_spec = mdl.predict(Xt_te_sc)
        hp_tag = _svr_tag(sel_hp)
    elif model_name == "acfnn":
        hp_clean = {k: v for k, v in sel_hp.items() if k != "_target_col"}
        mdl = TorchRegressor(random_state=random_state, **hp_clean)
        mdl.fit(Xt_tr, ytr_all_spec.ravel())
        yhat_spec = mdl.predict(Xt_te)
        hp_tag = _ac_tag(sel_hp)
    elif model_name == "ipls":
        mdl = PLSRegression(n_components=int(sel_hp["n_components"]))
        mdl.fit(Xt_tr, ytr_all_spec.reshape(-1, 1))
        yhat_spec = mdl.predict(Xt_te).ravel()
        hp_tag = f"C={sel_hp['n_components']}__I={sel_hp['num_intervals']}"
    else:
        raise ValueError(model_name)

    # Collapse to sample level
    y_pred_s, y_pred_s_std, y_pred_s_n, sids_unique = collapse_mean_std_n_by_sample_ids(
        yhat_spec, spec_sample_ids_te_target)
    y_true_s = np.array([y_true_by_sampleid[sid] for sid in sids_unique], dtype=float)

    ext_mets = _metrics(y_true_s, y_pred_s)

    return {
        "hp_tag": hp_tag,
        "ext_metrics": ext_mets,
        "y_true_s": y_true_s,
        "y_pred_s": y_pred_s,
        "y_pred_s_std": y_pred_s_std,
        "sids": sids_unique,
        "ipls_interval": ipls_interval,
    }


# ==================== COMBO RUNNER ====================

def run_combo(model, methods, target_col, args, include_types, rm_tr,
              Xte_target, spec_sample_ids_te_target, y_true_by_sampleid,
              svr_grid, acfnn_grid, pls_components, ipls_components, ipls_intervals,
              use_coral, wavenumbers, skip_importance=False, perm_repeats=3):
    """Run nested CV + external eval for one (model, preprocessing) combo."""
    try:
        sel = nested_select_one(
            model_name=model, methods=methods, target_col=target_col,
            n_outer=args.outer_folds, n_inner=args.inner_folds, seed=args.random_state,
            rm_tr=rm_tr, include_types=include_types,
            pls_components=pls_components, ipls_components=ipls_components,
            ipls_intervals=ipls_intervals, svr_grid=svr_grid, acfnn_grid=acfnn_grid,
            use_coral=use_coral, wavenumbers=wavenumbers,
            skip_importance=skip_importance, perm_repeats=perm_repeats,
        )

        hp = sel["selected_hp"].copy()
        hp["_target_col"] = target_col

        ext = run_external_eval(
            model, methods, hp, rm_tr, include_types,
            Xte_target, spec_sample_ids_te_target, y_true_by_sampleid,
            use_coral, wavenumbers, args.random_state
        )

        mlabel = " + ".join(methods) if methods else "No Preprocessing"
        row = {
            "target": target_col, "model": model, "preprocessing": mlabel,
            "hp": ext["hp_tag"],
            # Mean-of-fold metrics (primary selection criterion)
            "cv_rmse": sel["cv_mu_rmse"], "cv_rmse_sd": sel["cv_sd_rmse"],
            "cv_r2_mean": sel["cv_mu_r2"], "cv_r2_sd": sel["cv_sd_r2"],
            # Pooled metrics (computed on concatenated held-out predictions)
            "cv_rmse_pooled": sel["cv_metrics_pooled"]["rmse"],
            "cv_r2_pooled": sel["cv_metrics_pooled"]["r2"],
            "cv_mae_pooled": sel["cv_metrics_pooled"]["mae"],
            "cv_medae_pooled": sel["cv_metrics_pooled"]["medae"],
            "cv_evs_pooled": sel["cv_metrics_pooled"]["evs"],
            # External metrics
            "ext_rmse": ext["ext_metrics"]["rmse"], "ext_r2": ext["ext_metrics"]["r2"],
            "ext_mae": ext["ext_metrics"]["mae"], "ext_medae": ext["ext_metrics"]["medae"],
            "ext_evs": ext["ext_metrics"]["evs"],
            # Counts
            "n_train_samples": sel["n_train_samples"],
            "n_test_samples": len(ext["sids"]),
            "n_test_spectra": len(spec_sample_ids_te_target),
            # Per-sample external prediction spread
            "ext_pred_std_mean": float(np.mean(ext["y_pred_s_std"])),
            "ext_pred_std_max": float(np.max(ext["y_pred_s_std"])),
        }
        if ext["ipls_interval"]:
            a, b = ext["ipls_interval"]
            row["ipls_interval_a"] = a
            row["ipls_interval_b"] = b
            if wavenumbers is not None and len(wavenumbers) >= b:
                row["ipls_wn_lo"] = float(wavenumbers[a])
                row["ipls_wn_hi"] = float(wavenumbers[b - 1])

        return {
            "row": row,
            "sel": sel,
            "ext": ext,
            "model": model,
            "methods": methods,
            "hp": hp,
            "cv_mu_rmse": sel["cv_mu_rmse"],
        }
    except Exception as e:
        print(f"[ERROR] {model} | {methods} | {target_col}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return None


# ==================== PLOTTING ====================

def save_importance_plot(wavenumbers, importance, out_path, title, top_k=15):
    """Save importance spectrogram with top-k wavenumbers highlighted."""
    wavenumbers = np.asarray(wavenumbers).ravel()
    importance = np.asarray(importance).ravel()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 1]})

    ax1.plot(wavenumbers[:len(importance)], np.abs(importance), linewidth=1.2)
    ax1.set_xlabel("Wavenumber (cm⁻¹)")
    ax1.set_ylabel("|Importance|")
    ax1.set_title(f"{title} — |Importance|")

    # Highlight top-k
    order = np.argsort(np.abs(importance))[::-1]
    top_idx = np.sort(order[:min(top_k, len(order))])
    wn_plot = wavenumbers[:len(importance)]
    if len(wn_plot) > 1:
        dx = np.median(np.abs(np.diff(wn_plot)))
    else:
        dx = 1.0

    for idx in top_idx:
        wn = wn_plot[idx]
        ax1.axvspan(wn - 0.75 * dx, wn + 0.75 * dx, alpha=0.25, color="red")

    # Bottom: bar plot of top wavenumbers
    ax2.barh(range(min(top_k, len(order))),
             np.abs(importance[order[:min(top_k, len(order))]]),
             tick_label=[f"{wn_plot[i]:.0f}" for i in order[:min(top_k, len(order))]])
    ax2.set_xlabel("|Importance|")
    ax2.set_ylabel("Wavenumber")
    ax2.set_title(f"Top-{top_k} Wavenumbers")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==================== SVR GRID GENERATOR ====================

def generate_random_svr_params(num_iterations=20, seed=42):
    """
    Generate randomised SVR hyperparameter configurations.

    Design rationale for SERS spectral regression on small datasets:
      * RBF kernel is the primary kernel — smooth, infinite-dimensional, ideal
        for the subtle continuous relationships in spectral data.
      * Linear kernel is included as a simpler baseline (similar capacity to
        ridge regression but in the SVR loss framework).
      * C (regularisation inverse): log-uniform over [0.1, 1000].
        Low C → more regularised (good for small N / high p); high C → tighter fit.
      * epsilon (ε-insensitive tube): [0.01, 0.5].
        Smaller ε → tighter regression tube, larger → more tolerance.
      * gamma (RBF bandwidth): 'scale' (default = 1/(n_features*Var(X))) plus
        explicit values [1e-4, 1e-3, 1e-2, 0.1] spanning under- to over-fitting.

    These ranges are comparable in search breadth to the RF and ACFNN grids:
      * RF has ~20 random combos over 5 HPs
      * ACFNN has 3 architecture configs
      * SVR gets ~20 random combos over 4 HPs (C, ε, kernel, γ)
    """
    import random
    rng = random.Random(seed)
    grid = []
    for _ in range(num_iterations):
        kernel = rng.choice(["rbf", "rbf", "rbf", "linear"])  # 75% RBF, 25% linear
        C = rng.choice([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0])
        epsilon = rng.choice([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
        if kernel == "rbf":
            gamma = rng.choice(["scale", 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1])
        else:
            gamma = "scale"  # not used by linear kernel but sklearn accepts it
        grid.append({
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma,
        })
    return grid


# ==================== MAIN ====================

def main():
    ap = argparse.ArgumentParser(description="Unified SERS External Evaluation Pipeline")
    ap.add_argument("--train_data_dir", required=True)
    ap.add_argument("--train_meta_path", required=True)
    ap.add_argument("--test_data_dir", required=True)
    ap.add_argument("--test_meta_path", required=True)
    ap.add_argument("--qc_flags_xlsx", default=None,
                    help="Path to QC_flags.xlsx. If not provided, QC scenarios are skipped.")
    ap.add_argument("--include_types_list", default="DM1,Control;DM1",
                    help="Semicolon-separated include_types combos (default: 'DM1,Control;DM1')")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--outer_folds", type=int, default=10)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--scenarios", default="all",
                    help="Comma-separated: baseline,qc_only,coral_only,qc_coral or 'all'")
    ap.add_argument("--top_k_cv", type=int, default=10,
                    help="Number of top CV models to evaluate on external test")
    ap.add_argument("--models", default="pls,svr,ipls,acfnn",
                    help="Comma-separated model types to run (default: pls,svr,ipls,acfnn)")
    ap.add_argument("--targets", default="all",
                    help="Comma-separated target columns or 'all' (default: all = target_SI,HGS_pp_avg,ADF_pp_avg)")
    ap.add_argument("--preprocess_mode", default="full",
                    choices=["full", "minimal", "none"],
                    help="Preprocessing grid: full=17 combos (default), minimal=5 singles+none, none=no-preprocessing only")
    ap.add_argument("--n_svr_configs", type=int, default=20,
                    help="Number of random SVR HP configs (default: 20, use 3-5 for testing)")
    ap.add_argument("--acfnn_max_iter", type=int, default=0,
                    help="Override ACFNN max_iter for all configs (0 = use defaults: 200/300)")
    ap.add_argument("--pls_max_components", type=int, default=18,
                    help="Max PLS components to search (default: 18, use 5 for testing)")
    ap.add_argument("--ipls_max_components", type=int, default=10,
                    help="Max iPLS components to search (default: 10, use 4 for testing)")
    ap.add_argument("--ipls_intervals_list", default="5,10,15",
                    help="Comma-separated iPLS interval counts (default: 5,10,15)")
    ap.add_argument("--skip_importance", action="store_true",
                    help="Skip permutation importance for SVR/ACFNN (big speedup, ~40-60%% faster)")
    ap.add_argument("--perm_repeats", type=int, default=3,
                    help="Number of permutation importance repeats (default: 3, use 1 for speed)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Parse scenarios
    if args.scenarios == "all":
        scenarios = SCENARIOS[:]
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    # Check CORAL availability
    if not HAS_PREPROCESSING_NEW:
        scenarios = [s for s in scenarios if "coral" not in s]
        print("[WARN] preprocessing_new not found; CORAL scenarios disabled.")

    # Check QC availability
    if not args.qc_flags_xlsx:
        scenarios = [s for s in scenarios if "qc" not in s]
        print("[INFO] No QC_flags_xlsx provided; QC scenarios disabled.")

    # Parse include_types combos
    include_types_list = []
    for combo in args.include_types_list.split(";"):
        types = tuple([t.strip() for t in combo.split(",") if t.strip()])
        if types:
            include_types_list.append(types)

    # ---- Parse model list ----
    MODELS_TO_RUN = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    print(f"[CONFIG] Models: {MODELS_TO_RUN}")

    # ---- Parse target list ----
    if args.targets == "all":
        TARGETS_TO_RUN = TARGETS[:]
    else:
        target_cols_requested = [t.strip() for t in args.targets.split(",") if t.strip()]
        TARGETS_TO_RUN = [(nice, tcol) for nice, tcol in TARGETS if tcol in target_cols_requested]
        if not TARGETS_TO_RUN:
            raise ValueError(f"No valid targets found in: {args.targets}. Use target_SI, HGS_pp_avg, ADF_pp_avg")
    print(f"[CONFIG] Targets: {[t[1] for t in TARGETS_TO_RUN]}")

    # ---- Build preprocessing grid based on mode ----
    if args.preprocess_mode == "full":
        PREPROCESS_GRID_RUN = PREPROCESS_GRID[:]  # all 17
    elif args.preprocess_mode == "minimal":
        PREPROCESS_GRID_RUN = [[], ["EMSC"], ["Normalization"], ["SNV"], ["Second Derivative"]]
    elif args.preprocess_mode == "none":
        PREPROCESS_GRID_RUN = [[]]
    else:
        PREPROCESS_GRID_RUN = PREPROCESS_GRID[:]
    print(f"[CONFIG] Preprocessing combos: {len(PREPROCESS_GRID_RUN)}")

    # ---- Build HP grids ----
    PLS_COMPONENTS_RUN = list(range(2, args.pls_max_components + 1))
    IPLS_COMPONENTS_RUN = list(range(2, args.ipls_max_components + 1))
    IPLS_INTERVALS_RUN = [int(x) for x in args.ipls_intervals_list.split(",") if x.strip()]

    SVR_GRID = generate_random_svr_params(args.n_svr_configs, seed=args.random_state)

    ACFNN_GRID = [
        {"hidden_layer_sizes": (256, 128), "activation": "relu", "alpha": 1e-4,
         "learning_rate_init": 1e-3, "batch_size": 64, "max_iter": 200},
        {"hidden_layer_sizes": (512, 256), "activation": "relu", "alpha": 1e-4,
         "learning_rate_init": 5e-4, "batch_size": 64, "max_iter": 300},
        {"hidden_layer_sizes": (256, 256, 128), "activation": "relu", "alpha": 1e-5,
         "learning_rate_init": 1e-3, "batch_size": 128, "max_iter": 300},
    ]
    if args.acfnn_max_iter > 0:
        for cfg in ACFNN_GRID:
            cfg["max_iter"] = args.acfnn_max_iter

    n_combos = len(PREPROCESS_GRID_RUN) * len(MODELS_TO_RUN)
    n_total = n_combos * len(TARGETS_TO_RUN) * len(include_types_list) * len(scenarios)
    print(f"[CONFIG] {len(PREPROCESS_GRID_RUN)} preproc × {len(MODELS_TO_RUN)} models = {n_combos} combos/target")
    print(f"[CONFIG] × {len(TARGETS_TO_RUN)} targets × {len(include_types_list)} include_types × {len(scenarios)} scenarios = {n_total} total combo evaluations")
    print(f"[CONFIG] PLS components: 2-{args.pls_max_components} | iPLS components: 2-{args.ipls_max_components} | iPLS intervals: {IPLS_INTERVALS_RUN}")
    print(f"[CONFIG] SVR configs: {args.n_svr_configs} | ACFNN max_iter: {args.acfnn_max_iter or 'default'}")
    print(f"[CONFIG] skip_importance={args.skip_importance} | perm_repeats={args.perm_repeats}")
    print(f"[CONFIG] Backend: loky (process isolation) | dtype: float32")
    print()

    # ==================== MASTER SUMMARY ====================
    master_rows = []
    all_combos_mega = []  # accumulate all-combos CSVs across all scenarios for mega-file

    for scenario in scenarios:
        use_qc = "qc" in scenario
        use_coral = "coral" in scenario
        scenario_label = scenario.upper().replace("_", "+")

        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_label} (QC={use_qc}, CORAL={use_coral})")
        print(f"{'='*60}")

        for include_types in include_types_list:
            controls_label = "with_controls" if "Control" in include_types else "without_controls"
            print(f"\n--- Include types: {include_types} ({controls_label}) ---")

            # ---- Load data ----
            if use_qc:
                # QC needs filenames + rep_idx
                wn_tr, avg_tr, all_tr, fn_tr, rep_tr, meta_tr = load_data(
                    data_dir=args.train_data_dir, metadata_path=args.train_meta_path,
                    include_types=include_types, return_filenames=True,
                    return_rep_idx=True, strict=False, report_samples=2)
                wn_te, avg_te, all_te, fn_te, rep_te, meta_te = load_data(
                    data_dir=args.test_data_dir, metadata_path=args.test_meta_path,
                    include_types=include_types, return_filenames=True,
                    return_rep_idx=True, strict=False, report_samples=2)

                # Apply QC filtering
                all_tr, fn_tr, rep_tr, _ = apply_qc_filtering(
                    all_tr, fn_tr, rep_tr, args.qc_flags_xlsx, "Box1-2")
                all_te, fn_te, rep_te, _ = apply_qc_filtering(
                    all_te, fn_te, rep_te, args.qc_flags_xlsx, "Box3")

                spec_sample_ids_tr = np.array([_sample_id_from_filename(f) for f in fn_tr], dtype=str)
                spec_sample_ids_te = np.array([_sample_id_from_filename(f) for f in fn_te], dtype=str)

                print(f"[QC] Kept {all_tr.shape[1]} train spectra, {all_te.shape[1]} test spectra")

            elif use_coral:
                # CORAL needs filenames for SampleID mapping
                wn_tr, avg_tr, all_tr, fn_tr, rep_tr, meta_tr = load_data(
                    data_dir=args.train_data_dir, metadata_path=args.train_meta_path,
                    include_types=include_types, return_filenames=True,
                    return_rep_idx=True, strict=False, report_samples=2)
                wn_te, avg_te, all_te, fn_te, rep_te, meta_te = load_data(
                    data_dir=args.test_data_dir, metadata_path=args.test_meta_path,
                    include_types=include_types, return_filenames=True,
                    return_rep_idx=True, strict=False, report_samples=2)

                spec_sample_ids_tr = np.array([_sample_id_from_filename(f) for f in fn_tr], dtype=str)
                spec_sample_ids_te = np.array([_sample_id_from_filename(f) for f in fn_te], dtype=str)

            else:
                # Baseline: no filenames needed for grouping, but we load them for SampleID
                wn_tr, avg_tr, all_tr, fn_tr, meta_tr = load_data(
                    data_dir=args.train_data_dir, metadata_path=args.train_meta_path,
                    include_types=include_types, return_filenames=True,
                    strict=False, report_samples=2)
                wn_te, avg_te, all_te, fn_te, meta_te = load_data(
                    data_dir=args.test_data_dir, metadata_path=args.test_meta_path,
                    include_types=include_types, return_filenames=True,
                    strict=False, report_samples=2)

                spec_sample_ids_tr = np.array([_sample_id_from_filename(f) for f in fn_tr], dtype=str)
                spec_sample_ids_te = np.array([_sample_id_from_filename(f) for f in fn_te], dtype=str)

            # Normalize SampleID in metadata
            for df_name, df in [("train", meta_tr), ("test", meta_te)]:
                if "Sample_ID" in df.columns and "SampleID" not in df.columns:
                    df.rename(columns={"Sample_ID": "SampleID"}, inplace=True)

            # Coerce test targets
            for col in ("target_SI", "HGS_pp_avg", "ADF_pp_avg"):
                if col in meta_te.columns:
                    meta_te[col] = meta_te[col].astype(str).str.replace("%", "", regex=False).str.strip()
                    meta_te[col] = pd.to_numeric(meta_te[col], errors="coerce")

            # ---- Cast to float32 to halve memory (float64 is overkill for SERS) ----
            all_tr = all_tr.astype(np.float32)
            all_te = all_te.astype(np.float32)

            # Build GlobalGrouped
            rm_tr = GlobalGrouped(
                all_spectra_fxN=all_tr, meta=meta_tr,
                spec_sample_ids=spec_sample_ids_tr, is_qc=use_qc
            )

            # ---- Per-target evaluation ----
            for nice, tcol in TARGETS_TO_RUN:
                print(f"\n  === {nice} ({tcol}) ===")
                _flush()

                # Build test matrices for this target
                te_num = pd.to_numeric(meta_te[tcol], errors="coerce")
                te_keep = te_num.notna().to_numpy()
                if te_keep.sum() == 0:
                    print(f"  [SKIP] No valid test values for {tcol}")
                    continue

                meta_te_sids = meta_te.loc[te_keep, "SampleID"].astype(str) if "SampleID" in meta_te.columns \
                    else meta_te.loc[te_keep, "Sample_ID"].astype(str)
                keep_sids = set(meta_te_sids.tolist())
                y_true_by_sampleid = {
                    sid: float(y) for sid, y in zip(meta_te_sids, te_num[te_keep])
                }

                mask_te_spec = np.isin(spec_sample_ids_te, list(keep_sids))
                # all_te is ALWAYS (1731 features, n_spectra) from load_data
                # We need (n_spectra_kept, 1731 features) for modeling
                Xte_target = all_te[:, mask_te_spec].T
                spec_sample_ids_te_target = spec_sample_ids_te[mask_te_spec]

                if Xte_target.shape[0] < 3:
                    print(f"  [SKIP] Too few test spectra: {Xte_target.shape[0]}")
                    continue

                print(f"  [INFO] Test matrix: {Xte_target.shape[0]} spectra, "
                      f"{len(keep_sids)} samples, {Xte_target.shape[1]} features")
                _flush()

                tgt_out = os.path.join(args.out_dir, scenario, controls_label, tcol)
                os.makedirs(tgt_out, exist_ok=True)

                # ---- Run all combos ----
                combos_parallel = []
                models_parallel = [m for m in MODELS_TO_RUN if m != "acfnn"]
                for methods in PREPROCESS_GRID_RUN:
                    for model in models_parallel:
                        combos_parallel.append((model, methods))

                # LOKY backend: each worker is a separate process, so there
                # is no shared-memory corruption between threads.  This uses
                # more RAM than the old threading backend (~1-2 GB extra per
                # worker) but eliminates the class of silent data-corruption
                # crashes that killed the previous run.
                #
                # WHY THREADING FAILED:
                #   1) libsvm (SVR) and LAPACK (PLS) use internal C-level
                #      state that is NOT protected by the Python GIL.
                #      Multiple threads calling SVR.fit() or PLS.fit()
                #      simultaneously can corrupt each other's workspace
                #      buffers, causing segfaults or NaN results.
                #   2) The in-place permutation importance mutated X[:,j]
                #      while other threads were reading from the same
                #      underlying memory (numpy slices can share buffers
                #      even when they look like separate arrays).
                #   3) The Preprocessing class stores self.spectra on init;
                #      if two threads create Preprocessing(same_array) they
                #      share the reference, and any in-place op is a race.
                #
                # LOKY avoids all of this: each process has its own copy.
                n_jobs = min(N_PROC, len(combos_parallel))

                try:
                    print(f"  [PARALLEL] Running {len(combos_parallel)} combos with "
                          f"n_jobs={n_jobs} (loky backend)...")
                    _flush()

                    results = Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(run_combo)(
                            model, methods, tcol, args, include_types, rm_tr,
                            Xte_target, spec_sample_ids_te_target, y_true_by_sampleid,
                            SVR_GRID, ACFNN_GRID,
                            PLS_COMPONENTS_RUN, IPLS_COMPONENTS_RUN, IPLS_INTERVALS_RUN,
                            use_coral, wn_tr, args.skip_importance, args.perm_repeats,
                        )
                        for (model, methods) in combos_parallel
                    )
                    print(f"  [PARALLEL] Parallel block done. "
                          f"Successes: {sum(1 for r in results if r is not None)}/{len(results)}")
                    _flush()
                except Exception as e:
                    print(f"\n  [FATAL] Parallel execution crashed: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    _flush()
                    raise

                # AC-FNN sequential (Torch + multiprocessing is touchy)
                if "acfnn" in MODELS_TO_RUN:
                    print(f"  [SEQ] Running {len(PREPROCESS_GRID_RUN)} ACFNN combos sequentially...")
                    _flush()
                    for methods in PREPROCESS_GRID_RUN:
                        res = run_combo(
                            "acfnn", methods, tcol, args, include_types, rm_tr,
                            Xte_target, spec_sample_ids_te_target, y_true_by_sampleid,
                            SVR_GRID, ACFNN_GRID,
                            PLS_COMPONENTS_RUN, IPLS_COMPONENTS_RUN, IPLS_INTERVALS_RUN,
                            use_coral, wn_tr, args.skip_importance, args.perm_repeats,
                        )
                        results.append(res)

                # Filter None results
                results = [r for r in results if r is not None]
                if not results:
                    print(f"  [WARN] No successful results for {tcol}")
                    continue

                # ---- Save CSV of all combos ----
                rows = [r["row"] for r in results]
                df_all = pd.DataFrame(rows).sort_values(
                    by=["cv_rmse", "model", "preprocessing"]
                ).reset_index(drop=True)

                # Add scenario + controls columns so every CSV is self-describing
                df_all.insert(0, "scenario", scenario)
                df_all.insert(1, "controls", controls_label)

                csv_path = os.path.join(tgt_out, f"{tcol}__all_combos.csv")
                df_all.to_csv(csv_path, index=False)
                print(f"  [SAVED] {csv_path} ({len(df_all)} rows)")

                # Also accumulate for the mega-CSV
                all_combos_mega.append(df_all)

                # ---- Pick winner (lowest CV RMSE) ----
                best = None
                for r in results:
                    if best is None or r["cv_mu_rmse"] < best["cv_mu_rmse"] - 1e-12:
                        best = r
                    elif abs(r["cv_mu_rmse"] - best["cv_mu_rmse"]) <= 1e-12:
                        if _is_simpler(r["model"], r["hp"], best["model"], best["hp"]):
                            best = r

                # ---- Top-K CV models (sorted by cv_rmse) ----
                sorted_results = sorted(results, key=lambda r: r["cv_mu_rmse"])
                top_k = sorted_results[:args.top_k_cv]

                # ================================================
                # SAVE DETAILED OUTPUTS FOR EACH TOP-K COMBO
                # ================================================
                top_k_rows = []
                for rank, r in enumerate(top_k, 1):
                    is_winner = (r is best)
                    mlabel = " + ".join(r["methods"]) if r["methods"] else "NoPreproc"
                    safe_mlabel = mlabel.replace(" ", "").replace("+", "_")
                    combo_tag = f"rank{rank:02d}__{r['model']}__{safe_mlabel}"
                    combo_dir = os.path.join(tgt_out, combo_tag)
                    os.makedirs(combo_dir, exist_ok=True)

                    sel = r["sel"]
                    ext = r["ext"]

                    # ---------- 1. selection_trace.csv (per-fold HP + metrics) ----------
                    trace_rows = []
                    for t in sel["inner_trace"]:
                        trace_rows.append({
                            "outer_fold": t["outer_fold"],
                            "selected_hp": t["selected_hp_tag"],
                            "inner_best_mse_mean": t["inner_best_mse_mean"],
                            "inner_best_mse_std": t["inner_best_mse_std"],
                            "fold_rmse": t["fold_rmse"],
                            "fold_r2": t["fold_r2"],
                            "fold_mae": t["fold_mae"],
                            "n_train_samples": t["n_train_samples"],
                            "n_test_samples": t["n_test_samples"],
                            "n_train_spectra": t["n_train_spectra"],
                            "n_test_spectra": t["n_test_spectra"],
                            "ipls_interval": str(t["ipls_interval"]) if t["ipls_interval"] else "",
                        })
                    pd.DataFrame(trace_rows).to_csv(
                        os.path.join(combo_dir, "selection_trace.csv"), index=False)

                    # ---------- 2. cv_predictions.xlsx (per-sample, with SampleID + fold) ----------
                    cv_pred_rows = []
                    for t in sel["inner_trace"]:
                        fold = t["outer_fold"]
                        for sid, yt, yp, yp_std in zip(
                            t["_fold_sids"], t["_fold_y_true"],
                            t["_fold_y_pred"], t["_fold_y_pred_std"]
                        ):
                            cv_pred_rows.append({
                                "outer_fold": fold,
                                "SampleID": sid,
                                f"{tcol}__true": yt,
                                f"{tcol}__pred": yp,
                                f"{tcol}__pred_std": yp_std,
                            })
                    pd.DataFrame(cv_pred_rows).to_excel(
                        os.path.join(combo_dir, "cv_predictions.xlsx"), index=False)

                    # ---------- 3. ext_predictions.xlsx (per-sample external) ----------
                    ext_pred_df = pd.DataFrame({
                        "SampleID": ext["sids"],
                        f"{tcol}__true": ext["y_true_s"],
                        f"{tcol}__pred": ext["y_pred_s"],
                        f"{tcol}__pred_std": ext["y_pred_s_std"],
                    })
                    ext_pred_df.to_excel(
                        os.path.join(combo_dir, "ext_predictions.xlsx"), index=False)

                    # ---------- 4. CV parity plot ----------
                    parity_plot_sample_level(
                        y_true_sample=sel["y_true_cv"].reshape(-1, 1),
                        y_pred_sample=sel["y_pred_cv"].reshape(-1, 1),
                        y_pred_sample_std=sel["y_pred_cv_std"].reshape(-1, 1),
                        target_names=(nice,), out_dir=combo_dir,
                        fname_prefix="cv_parity",
                        title_suffix=f"CV | {r['model']} | {mlabel} | {scenario_label}"
                    )

                    # ---------- 5. External parity plot ----------
                    parity_plot_sample_level(
                        y_true_sample=ext["y_true_s"].reshape(-1, 1),
                        y_pred_sample=ext["y_pred_s"].reshape(-1, 1),
                        y_pred_sample_std=ext["y_pred_s_std"].reshape(-1, 1),
                        target_names=(nice,), out_dir=combo_dir,
                        fname_prefix="ext_parity",
                        title_suffix=f"EXT | {r['model']} | {mlabel} | {scenario_label}"
                    )

                    # ---------- 6. Feature importance (plot + CSV) ----------
                    if sel["importance_mean"] is not None:
                        save_importance_plot(
                            wn_tr, sel["importance_mean"],
                            os.path.join(combo_dir, "feature_importance.png"),
                            f"Rank {rank} | {nice} | {r['model']} | {mlabel}"
                        )
                        imp_df = pd.DataFrame({
                            "wavenumber": wn_tr[:len(sel["importance_mean"])],
                            "importance": sel["importance_mean"],
                            "abs_importance": np.abs(sel["importance_mean"]),
                        }).sort_values("abs_importance", ascending=False)
                        imp_df.to_csv(
                            os.path.join(combo_dir, "feature_importance.csv"), index=False)

                    # ---------- 7. Summary row for the top-K CSV ----------
                    ext_row = r["row"].copy()
                    ext_row["cv_rank"] = rank
                    ext_row["is_winner"] = is_winner
                    ext_row["scenario"] = scenario
                    ext_row["controls"] = controls_label

                    # --- Generalization gap metrics (CV vs External) ---
                    ext_row["gap_rmse"] = ext_row["ext_rmse"] - ext_row["cv_rmse"]
                    ext_row["gap_rmse_abs"] = abs(ext_row["gap_rmse"])
                    ext_row["gap_r2"] = ext_row["cv_r2_mean"] - ext_row["ext_r2"]  # positive = CV was better
                    ext_row["gap_r2_abs"] = abs(ext_row["gap_r2"])
                    # Ratio: ext_rmse / cv_rmse (1.0 = perfect transfer; >1 = degraded)
                    ext_row["transfer_ratio_rmse"] = (
                        ext_row["ext_rmse"] / ext_row["cv_rmse"]
                        if ext_row["cv_rmse"] > 1e-12 else np.nan
                    )

                    # --- CV fold stability (lower = more stable across folds) ---
                    ext_row["cv_rmse_cv_coeff"] = (
                        ext_row["cv_rmse_sd"] / ext_row["cv_rmse"]
                        if ext_row["cv_rmse"] > 1e-12 else np.nan
                    )  # coefficient of variation of fold RMSEs

                    # Add per-fold RMSEs as individual columns for backcalculation
                    for fi, fr in enumerate(sel["rmse_folds"]):
                        ext_row[f"cv_fold{fi+1}_rmse"] = fr
                    for fi, fr2 in enumerate(sel["r2_folds"]):
                        ext_row[f"cv_fold{fi+1}_r2"] = fr2
                    top_k_rows.append(ext_row)

                # --- Pareto front flags (non-dominated on cv_rmse × ext_rmse × gap_rmse_abs) ---
                # A combo is Pareto-optimal if no other combo is better on ALL three criteria
                for i, ri in enumerate(top_k_rows):
                    dominated = False
                    for j, rj in enumerate(top_k_rows):
                        if i == j:
                            continue
                        if (rj["cv_rmse"] <= ri["cv_rmse"] and
                            rj["ext_rmse"] <= ri["ext_rmse"] and
                            rj["gap_rmse_abs"] <= ri["gap_rmse_abs"] and
                            (rj["cv_rmse"] < ri["cv_rmse"] or
                             rj["ext_rmse"] < ri["ext_rmse"] or
                             rj["gap_rmse_abs"] < ri["gap_rmse_abs"])):
                            dominated = True
                            break
                    ri["pareto_front"] = not dominated

                # Add ALL top-K to master (not just winner) for cross-scenario comparison
                for ext_row in top_k_rows:
                    master_rows.append(ext_row.copy())

                # Save top-K summary CSV
                top_k_df = pd.DataFrame(top_k_rows)
                top_k_df.to_csv(
                    os.path.join(tgt_out, f"{tcol}__top{args.top_k_cv}_detailed.csv"), index=False)
                print(f"  [SAVED] top-{args.top_k_cv} detailed CSV with per-fold metrics + Pareto flags")

                # --- Free memory: drop bulky per-fold data from results ---
                del results, top_k, sorted_results, top_k_rows, top_k_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ==================== MEGA-CSV: all combos across all scenarios ====================
    if all_combos_mega:
        mega_df = pd.concat(all_combos_mega, ignore_index=True)
        mega_path = os.path.join(args.out_dir, "ALL_COMBOS_ALL_SCENARIOS.csv")
        mega_df.to_csv(mega_path, index=False)
        print(f"\n[MEGA] Saved all-combos across all scenarios: {mega_path} ({len(mega_df)} rows)")

    # ==================== MASTER SUMMARY EXCEL ====================
    if master_rows:
        master_df = pd.DataFrame(master_rows)
        master_path = os.path.join(args.out_dir, "MASTER_SUMMARY.xlsx")
        with pd.ExcelWriter(master_path, engine="openpyxl") as xw:
            # Sheet 1: ALL top-K across all scenarios/targets/controls
            master_df.to_excel(xw, sheet_name="all_top_k", index=False)

            # Sheet 2: Winners only (cv_rank == 1)
            winners = master_df[master_df["is_winner"] == True]
            if not winners.empty:
                winners.to_excel(xw, sheet_name="winners_only", index=False)

            # Sheet 3: Pareto-optimal combos only
            pareto = master_df[master_df["pareto_front"] == True]
            if not pareto.empty:
                pareto.to_excel(xw, sheet_name="pareto_front", index=False)

            # Per-scenario sheets
            for sc in scenarios:
                sub = master_df[master_df["scenario"] == sc]
                if not sub.empty:
                    sub.to_excel(xw, sheet_name=f"sc_{sc[:27]}", index=False)

            # Per-target sheets
            for _, tcol_s in TARGETS:
                sub = master_df[master_df["target"] == tcol_s]
                if not sub.empty:
                    sub.to_excel(xw, sheet_name=f"tgt_{tcol_s[:26]}", index=False)

            # Per-target × Pareto (the most useful sheet for deployment decisions)
            for _, tcol_s in TARGETS:
                sub = master_df[(master_df["target"] == tcol_s) & (master_df["pareto_front"] == True)]
                if not sub.empty:
                    sub.to_excel(xw, sheet_name=f"pareto_{tcol_s[:24]}", index=False)

        print(f"\n[MASTER] Saved: {master_path}")
        print(f"  Total rows: {len(master_df)} | Winners: {len(winners)} | Pareto-optimal: {len(pareto)}")

    print("\n" + "=" * 60)
    print("UNIFIED EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()