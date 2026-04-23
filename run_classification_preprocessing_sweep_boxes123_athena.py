# -*- coding: utf-8 -*-
"""
Athena-Ready Classification Preprocessing Sweep (Boxes 1-2-3)
with QC and CORAL scenario support.

Scenario variants per study design:
  Box12_LOSO       : baseline | qc_only
  Box123_LOSO      : baseline | qc_only
  Train12_Predict3 : baseline | qc_only | coral_only | qc_coral

QC: load only spectra flagged "keep" in QC_metrics_per_spectrum.csv.
    All patients remain equally weighted even when surviving spectrum
    counts differ (enforced via averaging in PATIENT_AVG mode and
    inverse-frequency sample weights in SPECTRUM_LEVEL mode).

CORAL: aligns Box3 test spectra to Box12 DM1 spectral space.
    Source domain = Box12 DM1-only spectra (post-preprocessing).
    Applied only in Train12_Predict3 coral_only / qc_coral variants.

Output path: <out_root>/<scenario>/<mode>/<variant>/<preproc>/<model>/
All outputs include a 'variant' column in metrics.csv.

Original evaluation logic UNCHANGED for baseline variant:
  - LOSO holds out all spectra per patient
  - EMSC reference computed from TRAIN only inside each fold
  - Baseline correction precomputed once (per-spectrum, fold-independent)
  - Spectrum-level mode trains on spectra, aggregates held-out predictions
"""

from __future__ import annotations

# ── Thread pinning BEFORE numpy/sklearn import ──────────────────────────────
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, log_loss
from sklearn.model_selection import LeaveOneGroupOut

# ── Project imports ──────────────────────────────────────────────────────────
from data_loader import load_data                          # baseline loader (original)
from preprocessing import Preprocessing                    # preprocessing module (original)

# QC-aware loader (new module in same directory)
try:
    from data_loader_qc import load_data as load_data_qc
    HAS_QC_LOADER = True
except ImportError:
    HAS_QC_LOADER = False

# CORAL alignment (new module in same directory)
try:
    from preprocessing_new import coral_align_target_to_source
    HAS_CORAL = True
except ImportError:
    HAS_CORAL = False

# True sample-weighted PLS (weighted_pls.py in same directory)
try:
    from weighted_pls import WeightedPLSRegression
    HAS_WEIGHTED_PLS = True
except ImportError:
    HAS_WEIGHTED_PLS = False


# =============================================================================
# CONSTANTS
# =============================================================================
N_REP = 9
PLS_COMPONENT_RANGE = [2, 3, 4, 5]
TOP_N_WN = 60
HIGHLIGHT_TOP_K_WN = 15
RANDOM_STATE = 0

# =============================================================================
# DEFAULT PATHS (Athena layout)
# =============================================================================
_HOME = Path(os.path.expanduser("~/sers_project"))
DEFAULT_TRAIN_DATA = _HOME / "data" / "Box12_spectra"
DEFAULT_TEST_DATA  = _HOME / "data" / "Box3_spectra"
DEFAULT_TRAIN_META = _HOME / "data" / "Box12_metadata.csv"
DEFAULT_TEST_META  = _HOME / "data" / "Box3_metadata.csv"
DEFAULT_OUT_ROOT   = _HOME / "results" / "classification"

# =============================================================================
# PREPROCESSING COMBOS
# =============================================================================
BASELINE_CFG = {"do_baseline": True, "lam": 1e6, "p": 0.01, "niter": 10}
NO_BASELINE  = {"do_baseline": False}

ALL_PREPROC = {
    "RAW":               {"methods": [],                           "baseline": NO_BASELINE},
    "SNV":               {"methods": ["SNV"],                      "baseline": NO_BASELINE},
    "NORM":              {"methods": ["Normalization"],            "baseline": NO_BASELINE},
    "EMSC":              {"methods": ["EMSC"],                     "baseline": NO_BASELINE},
    "SNV+2DER":          {"methods": ["SNV", "Second Derivative"], "baseline": NO_BASELINE},
    "BASELINE+SNV":      {"methods": ["SNV"],                      "baseline": BASELINE_CFG},
    "BASELINE+NORM":     {"methods": ["Normalization"],            "baseline": BASELINE_CFG},
    "BASELINE+EMSC":     {"methods": ["EMSC"],                     "baseline": BASELINE_CFG},
    "BASELINE+SNV+2DER": {"methods": ["SNV", "Second Derivative"], "baseline": BASELINE_CFG},
}

ALL_MODELS = ["svm", "plsda"]

# Base scenario names (study designs, without variant suffix)
ALL_SCENARIOS = [
    "Box12_LOSO_PATIENT_AVG",
    "Box12_LOSO_SPECTRUM_LEVEL",
    "Train12_Predict3_PATIENT_AVG",
    "Train12_Predict3_SPECTRUM_LEVEL",
    "Box123_LOSO_PATIENT_AVG",
    "Box123_LOSO_SPECTRUM_LEVEL",
]

# Which variants are allowed for each study design
ALL_SCENARIO_VARIANTS = {
    "Box12_LOSO_PATIENT_AVG":          ["baseline", "qc_only"],
    "Box12_LOSO_SPECTRUM_LEVEL":       ["baseline", "qc_only"],
    "Train12_Predict3_PATIENT_AVG":    ["baseline", "qc_only", "coral_only", "qc_coral"],
    "Train12_Predict3_SPECTRUM_LEVEL": ["baseline", "qc_only", "coral_only", "qc_coral"],
    "Box123_LOSO_PATIENT_AVG":         ["baseline", "qc_only"],
    "Box123_LOSO_SPECTRUM_LEVEL":      ["baseline", "qc_only"],
}

ALL_VARIANTS = ["baseline", "qc_only", "coral_only", "qc_coral"]


# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("classification_sweep")


def setup_logging(out_root: Path, level=logging.INFO):
    out_root.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(out_root / "sweep.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt); fh.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SERS Classification Preprocessing Sweep (Athena cluster)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--train-data", type=str, default=str(DEFAULT_TRAIN_DATA))
    p.add_argument("--test-data",  type=str, default=str(DEFAULT_TEST_DATA))
    p.add_argument("--train-meta", type=str, default=str(DEFAULT_TRAIN_META))
    p.add_argument("--test-meta",  type=str, default=str(DEFAULT_TEST_META))
    p.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUT_ROOT))

    p.add_argument("--preproc", nargs="*", default=None,
                   help=f"Preprocessing combos to run (default: all). Choices: {list(ALL_PREPROC.keys())}")
    p.add_argument("--models", nargs="*", default=None,
                   help=f"Models to run (default: all). Choices: {ALL_MODELS}")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help=f"Base scenarios to run (default: all). Choices: {ALL_SCENARIOS}")
    p.add_argument("--variants", nargs="*", default=None,
                   help=f"Variants to run (default: all applicable). Choices: {ALL_VARIANTS}")

    # QC data paths
    p.add_argument("--qc-metrics", type=str, default=None,
                   help="Path to QC_metrics_per_spectrum.csv (required for qc_only / qc_coral variants)")
    p.add_argument("--qc-box-train", type=str, default="Box1-2",
                   help="Box label for Box12 train data in QC file (default: 'Box1-2')")
    p.add_argument("--qc-box-test", type=str, default="Box3",
                   help="Box label for Box3 test data in QC file (default: 'Box3')")

    p.add_argument("--max-workers", "-j", type=int, default=1)
    p.add_argument("--force", action="store_true")

    # Fast-mode flags
    p.add_argument("--skip-plots",   action="store_true")
    p.add_argument("--skip-roc",     action="store_true")
    p.add_argument("--skip-overlay", action="store_true")
    p.add_argument("--skip-top-wn",  action="store_true")
    p.add_argument("--max-spectra-plot", type=int, default=250)

    return p.parse_args(argv)


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# LABEL / SI / ID HELPERS
# =============================================================================
def get_labels(meta: pd.DataFrame) -> np.ndarray:
    if "Type" not in meta.columns:
        raise KeyError("Metadata must contain a 'Type' column with values Control/DM1.")
    return meta["Type"].map({"Control": 0, "DM1": 1}).values.astype(int)

def get_si(meta: pd.DataFrame) -> np.ndarray:
    candidates = ["target_SI", "SI", "SplicingIndex", "Splicing_Index", "splicing_index"]
    for c in candidates:
        if c in meta.columns:
            return pd.to_numeric(meta[c], errors="coerce").values.astype(float)
    raise KeyError(f"Could not find SI column in metadata. Tried: {candidates}")

def try_get_si(meta: pd.DataFrame) -> np.ndarray | None:
    try:
        return get_si(meta)
    except Exception:
        return None

def get_patient_ids(meta: pd.DataFrame) -> np.ndarray:
    if "Sample_ID" in meta.columns:
        return meta["Sample_ID"].astype(str).values
    if "SampleID" in meta.columns:
        return meta["SampleID"].astype(str).values
    return meta.index.astype(str).values


# =============================================================================
# SHAPE HELPERS
# =============================================================================
def reshape_patient(X_spec: np.ndarray, n_rep: int) -> np.ndarray:
    """X_spec: (n_wavenumbers, n_spectra_total) -> (n_patients, n_rep, n_wavenumbers)"""
    n_w, n_total = X_spec.shape
    if n_total % n_rep != 0:
        raise ValueError(f"Total spectra ({n_total}) not divisible by N_REP ({n_rep}).")
    n_pat = n_total // n_rep
    return X_spec.T.reshape(n_pat, n_rep, n_w)


# =============================================================================
# ROBUST METRICS
# =============================================================================
def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))

def safe_confusion_matrix_2x2(y_true, y_pred):
    return confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[0, 1],
    )

def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true).astype(int) == np.asarray(y_pred).astype(int)))

def balanced_accuracy_present_classes(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = safe_confusion_matrix_2x2(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recalls = []
    if np.any(y_true == 0):
        denom0 = tn + fp
        recalls.append((tn / denom0) if denom0 > 0 else np.nan)
    if np.any(y_true == 1):
        denom1 = tp + fn
        recalls.append((tp / denom1) if denom1 > 0 else np.nan)
    recalls = [r for r in recalls if np.isfinite(r)]
    return float(np.mean(recalls)) if recalls else np.nan


# =============================================================================
# BASELINE CORRECTION (ALS)
# =============================================================================
def als_baseline(y, lam=1e6, p=0.01, niter=10):
    y = y.astype(float)
    L = y.size
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i] = 1; D[i, i + 1] = -2; D[i, i + 2] = 1
    w = np.ones(L)
    for _ in range(niter):
        W = np.diag(w)
        Z = W + lam * (D.T @ D)
        z = np.linalg.solve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_correct_matrix(X, lam=1e6, p=0.01, niter=10):
    """X: (n_features, n_spectra) column-wise spectra."""
    X = np.asarray(X)
    Xc = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        b = als_baseline(X[:, j], lam=lam, p=p, niter=niter)
        Xc[:, j] = X[:, j] - b
    return Xc

def baseline_correct_X3(X3: np.ndarray, baseline_cfg: dict) -> np.ndarray:
    """X3: (n_pat, n_rep, n_w) -> baseline-corrected same shape."""
    if not baseline_cfg or not baseline_cfg.get("do_baseline", False):
        return X3
    lam = float(baseline_cfg.get("lam", 1e6))
    p_val = float(baseline_cfg.get("p", 0.01))
    niter = int(baseline_cfg.get("niter", 10))
    n_pat, n_rep, n_w = X3.shape
    X_spec = X3.reshape(n_pat * n_rep, n_w).T
    X_spec_bc = baseline_correct_matrix(X_spec, lam=lam, p=p_val, niter=niter)
    return X_spec_bc.T.reshape(n_pat, n_rep, n_w)

def baseline_correct_spectra(X_spec_fxN: np.ndarray, baseline_cfg: dict) -> np.ndarray:
    """Apply ALS baseline correction to a (n_w, n_spectra) features-first matrix."""
    if not baseline_cfg or not baseline_cfg.get("do_baseline", False):
        return X_spec_fxN
    return baseline_correct_matrix(
        X_spec_fxN,
        lam=float(baseline_cfg.get("lam", 1e6)),
        p=float(baseline_cfg.get("p", 0.01)),
        niter=int(baseline_cfg.get("niter", 10)),
    )


# =============================================================================
# PREPROCESSING (leakage-safe)
# =============================================================================
def apply_preproc_leakage_safe(X_train, X_test, methods):
    """X_*: (n_w, n_spectra). EMSC reference from TRAIN only."""
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    emsc_ref = None
    if "EMSC" in methods:
        emsc_ref = np.mean(X_train, axis=1)

    def _apply(Z):
        out = np.asarray(Z)
        for m in methods:
            pre = Preprocessing(out)
            if m == "EMSC":
                out = pre.emsc(out, reference=emsc_ref)
            elif m == "Normalization":
                out = pre.normalize_spectrum(out)
            elif m == "SNV":
                out = pre.snv(out)
            elif m == "Second Derivative":
                out = pre.second_derivative(out)
            else:
                raise ValueError(f"Unknown preprocessing method: {m}")
        return out

    return _apply(X_train), _apply(X_test)


# =============================================================================
# QC / CORAL HELPER FUNCTIONS
# =============================================================================

def load_qc_cohort(data_dir, meta_path, qc_metrics_path, qc_box, min_kept=1):
    """
    Load QC-filtered cohort using data_loader_qc.

    Returns a dict with:
      X_spec     : (1731, k_total) features-first, k_total = total surviving spectra
      X_spec_bc  : None placeholder (filled in main() if baseline is needed)
      spec_sids  : (k_total,) str array — FilePrefix of each spectrum's patient
      y          : (n_pat,) int labels aligned to ids
      si         : (n_pat,) float or None
      ids        : (n_pat,) str FilePrefix IDs (ordered, same as meta["FilePrefix"])
      meta       : aligned_meta DataFrame
    """
    if not HAS_QC_LOADER:
        raise RuntimeError("data_loader_qc module not found. Cannot load QC data.")

    result = load_data_qc(
        data_dir=str(data_dir),
        metadata_path=str(meta_path),
        include_types=["DM1", "Control"],
        strict=False,
        qc_metrics_path=str(qc_metrics_path),
        qc_box=qc_box,
        min_kept_per_sample=min_kept,
    )
    # Unpack: (wavenumbers, averaged_spectra, all_spectra, aligned_meta,
    #          spectrum_sample_ids, kept_spectrum_keys)
    wn, avg, X_spec, meta, spec_sids_raw, _ = result

    ids = np.asarray(meta["FilePrefix"].values, dtype=str)  # consistent with spec_sids
    spec_sids = np.asarray(spec_sids_raw, dtype=str)
    y = get_labels(meta)
    si = try_get_si(meta)

    return {
        "X_spec":    X_spec,        # (1731, k_total)
        "X_spec_bc": None,          # filled later if needed
        "spec_sids": spec_sids,     # (k_total,)
        "y":         y,
        "si":        si,
        "ids":       ids,
        "meta":      meta,
    }


def build_patient_avg(X_spec_fxN: np.ndarray,
                      spec_sids: np.ndarray,
                      ids: np.ndarray) -> np.ndarray:
    """
    Average spectra within each patient to produce a patient-level feature matrix.

    X_spec_fxN : (n_w, k_total)  features-first
    spec_sids  : (k_total,) str  patient ID for each column
    ids        : (n_pat,) str    ordered patient IDs

    Returns (n_pat, n_w).
    """
    n_w = X_spec_fxN.shape[0]
    Xpat = np.zeros((len(ids), n_w), dtype=float)
    for i, sid in enumerate(ids):
        mask = (spec_sids == sid)
        Xpat[i] = X_spec_fxN[:, mask].mean(axis=1)
    return Xpat


def equal_patient_weights(spec_sids: np.ndarray,
                          ids: np.ndarray) -> np.ndarray:
    """
    Compute per-spectrum sample weights so every patient contributes equal
    total weight regardless of how many spectra survived QC.

    Weights are normalized so sum = len(ids), preserving effective n_samples
    for sklearn's internal loss scaling.

    spec_sids : (k_total,) str
    ids       : (n_pat,) str
    Returns   : (k_total,) float
    """
    n_spec_per_pat = {sid: int((spec_sids == sid).sum()) for sid in ids}
    w = np.array([1.0 / n_spec_per_pat[sid] for sid in spec_sids], dtype=float)
    w = w * len(ids) / w.sum()
    return w


def build_groups_from_sids(spec_sids: np.ndarray,
                            ids: np.ndarray) -> np.ndarray:
    """
    Map per-spectrum sample IDs to integer patient indices (0 .. n_pat-1).
    Used as the `groups` argument in LOSO inner CV for PLS-DA.
    """
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    return np.array([id_to_idx[sid] for sid in spec_sids], dtype=int)


def _apply_coral(Xtr_p_fxN: np.ndarray,
                 dm1_spec_mask: np.ndarray,
                 Xte_p_fxN: np.ndarray) -> np.ndarray:
    """
    Apply CORAL domain adaptation to test spectra.

    Source domain = training DM1 spectra only (not controls).
    Target domain = all test spectra.

    Xtr_p_fxN   : (n_w, n_tr_spec) preprocessed training spectra, features-first
    dm1_spec_mask: (n_tr_spec,) bool — True for DM1 training spectra
    Xte_p_fxN   : (n_w, n_te_spec) preprocessed test spectra, features-first

    Returns Xte_aligned : (n_w, n_te_spec) features-first, float64
    """
    if not HAS_CORAL:
        raise RuntimeError("preprocessing_new module not found. CORAL is unavailable.")

    X_source = Xtr_p_fxN[:, dm1_spec_mask].T     # (n_dm1, n_w)
    X_target = Xte_p_fxN.T                        # (n_te, n_w)

    if X_source.shape[0] < 2:
        raise ValueError(
            "CORAL requires at least 2 DM1 training spectra. "
            f"Found {X_source.shape[0]}."
        )

    Xte_aligned = coral_align_target_to_source(X_source, X_target)  # (n_te, n_w)
    return Xte_aligned.T.astype(np.float64)       # (n_w, n_te)


# =============================================================================
# PLOT + EXPORT HELPERS
# =============================================================================
def _legend_outside_right(ax, **kw):
    """Place legend to the right of axes; rely on bbox_inches='tight' when saving."""
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0, frameon=True, **kw)

def save_confusion_plot(cm, outpath: Path, title: str):
    fig = plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm)
    plt.title(title, fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks([0, 1], ["Control", "DM1"]); plt.yticks([0, 1], ["Control", "DM1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close(fig)

def save_score_hist(y_true, y_score, outpath: Path, title: str):
    fig = plt.figure(figsize=(6.6, 4.2))
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    plt.hist(y_score[y_true == 0], bins=20, alpha=0.7, label="Control")
    plt.hist(y_score[y_true == 1], bins=20, alpha=0.7, label="DM1")
    plt.title(title, fontsize=8); plt.xlabel("Score / Probability (DM1)"); plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close(fig)

def save_roc_or_note(y_true, y_score, outdir: Path, title: str):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        write_text(outdir / "roc_NA.txt", "ROC/AUC not computed: y_true contains only one class.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig = plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title}\nAUC={auc:.3f}", fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / "roc.png", dpi=220, bbox_inches="tight"); plt.close(fig)

def save_predprob_parity(y_true, y_score, outpath: Path, title: str, seed: int = 0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    rng = np.random.default_rng(seed)
    y_jit = y_true.astype(float) + rng.uniform(-0.06, 0.06, size=len(y_true))
    # Extra right margin so legend sits fully outside the axes without clipping.
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    m0 = (y_true == 0); m1 = (y_true == 1)
    ax.scatter(y_score[m0], y_jit[m0], s=24, alpha=0.9, label="Control")
    ax.scatter(y_score[m1], y_jit[m1], s=24, alpha=0.9, label="DM1")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "DM1"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Predicted Probability (DM1)")
    ax.set_title(title, fontsize=8, pad=6)
    _legend_outside_right(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight"); plt.close(fig)

def save_predprob_parity_si_gradient(y_true, y_score, si_values, outpath: Path, title: str, seed: int = 0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    si_values = np.asarray(si_values).astype(float)
    rng = np.random.default_rng(seed)
    y_jit = y_true.astype(float) + rng.uniform(-0.06, 0.06, size=len(y_true))
    # Extra width to accommodate colorbar on the right without crowding the axes.
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    m0 = (y_true == 0); m1 = (y_true == 1)
    ax.scatter(y_score[m0], y_jit[m0], s=24, alpha=0.9, label="Control")
    dm1_si = si_values[m1]; dm1_score = y_score[m1]; dm1_y = y_jit[m1]
    if np.all(~np.isfinite(dm1_si)):
        ax.scatter(dm1_score, dm1_y, s=24, alpha=0.9, label="DM1 (SI N/A)")
    else:
        vmin, vmax = np.nanmin(dm1_si), np.nanmax(dm1_si)
        sc = ax.scatter(dm1_score, dm1_y, s=24, alpha=0.9, c=dm1_si,
                        cmap="viridis", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("SI (DM1 only)", labelpad=6)
        ax.scatter([], [], s=24, alpha=0.9, color="C1", label="DM1 (colored by SI)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "DM1"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Predicted Probability (DM1)")
    ax.set_title(title, fontsize=8, pad=6)
    # Legend inside upper-left so it does not collide with the colorbar.
    ax.legend(loc="upper left", framealpha=0.85, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches="tight"); plt.close(fig)

def save_importance_spectrogram(wavenumbers, importance, outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    ax.plot(wavenumbers, np.abs(importance))
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("|Importance|")
    ax.set_title(title, fontsize=8, pad=6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close(fig)

def export_top_wavenumbers(wavenumbers: np.ndarray, importance: np.ndarray, out_csv: Path, top_n=60):
    order = np.argsort(np.abs(importance))[::-1]
    top = order[:top_n]
    pd.DataFrame({
        "rank": np.arange(1, len(top) + 1),
        "wavenumber": wavenumbers[top],
        "importance": importance[top],
        "abs_importance": np.abs(importance[top]),
    }).to_csv(out_csv, index=False)

def save_overlay_spectra_with_highlighted_wavenumbers(
    wavenumbers, X_spec, importance, outpath, title,
    top_k=15, max_spectra_to_plot=250, seed=0,
):
    wavenumbers = np.asarray(wavenumbers).ravel()
    X_spec = np.asarray(X_spec)
    importance = np.asarray(importance).ravel()
    if X_spec.ndim != 2 or X_spec.shape[0] != wavenumbers.size or importance.size != wavenumbers.size:
        raise ValueError("Overlay inputs shape mismatch.")
    order = np.argsort(np.abs(importance))[::-1]
    top_k = int(min(top_k, len(order)))
    top_idx = np.sort(order[:top_k])
    rng = np.random.default_rng(seed)
    n_spec = X_spec.shape[1]
    Xp = (X_spec[:, rng.choice(n_spec, size=min(max_spectra_to_plot, n_spec), replace=False)]
          if n_spec > max_spectra_to_plot else X_spec)
    fig, ax = plt.subplots(figsize=(16, 4.2))
    for j in range(Xp.shape[1]):
        ax.plot(wavenumbers, Xp[:, j], linewidth=1.2, alpha=0.08)
    if len(wavenumbers) > 1:
        dx = np.median(np.abs(np.diff(wavenumbers))); half_width = 0.75 * dx
    else:
        half_width = 1.0
    for idx in top_idx:
        wn = wavenumbers[idx]; ax.axvspan(wn - half_width, wn + half_width, alpha=0.25)
    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close(fig)


# =============================================================================
# MODELS + IMPORTANCE
# =============================================================================
def fit_predict_svm_linear(X_train, y_train, X_test, sample_weight=None):
    """
    Linear SVM classifier with optional per-sample weights.
    sample_weight : (n_train,) float or None — used when QC leaves variable
                    spectra counts and equal patient weighting is needed.
    """
    clf = SVC(kernel="linear", probability=True, class_weight="balanced",
              random_state=RANDOM_STATE, max_iter=50000, cache_size=500)
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test  = np.ascontiguousarray(X_test, dtype=np.float64)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    score = clf.predict_proba(X_test)[:, 1]
    importance = clf.coef_.ravel()
    return score, importance, clf

def _safe_max_components(n_samples: int, n_features: int) -> list[int]:
    max_k = min(n_samples - 1, n_features)
    return [k for k in PLS_COMPONENT_RANGE if k <= max_k]

def fit_predict_plsda(X_train, y_train, X_test, groups=None, sample_weight=None):
    """
    PLS-DA with balanced LogisticRegression calibrator, inner CV component selection.

    groups        : used for LOSO inner CV grouping (LeaveOneGroupOut).
    sample_weight : applied to the final LogisticRegression calibrator only.
                    PLSRegression does not support sample_weight; the PLS
                    projection is fit on all spectra equally. The calibrator
                    is weighted to enforce equal patient contribution.
    """
    feasible = _safe_max_components(X_train.shape[0], X_train.shape[1])
    if not feasible:
        raise ValueError(
            f"Cannot fit PLS-DA: n_train={X_train.shape[0]}, "
            f"n_features={X_train.shape[1]}, no feasible component count."
        )

    if len(feasible) == 1:
        best_k = feasible[0]
    else:
        if groups is not None:
            inner_cv = LeaveOneGroupOut()
            cv_splits = list(inner_cv.split(X_train, y_train, groups))
        else:
            inner_cv = LeaveOneOut()
            cv_splits = list(inner_cv.split(X_train))

        mean_losses = {}
        for k in feasible:
            fold_losses = []
            for tr_i, val_i in cv_splits:
                pls_k = PLSRegression(n_components=k)
                pls_k.fit(X_train[tr_i], y_train[tr_i])
                T_tr  = pls_k.transform(X_train[tr_i])
                T_val = pls_k.transform(X_train[val_i])
                cal_k = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
                cal_k.fit(T_tr, y_train[tr_i])
                prob_val = np.clip(cal_k.predict_proba(T_val), 1e-8, 1.0 - 1e-8)
                fold_losses.append(log_loss(y_train[val_i], prob_val, labels=[0, 1]))
            mean_losses[k] = np.mean(fold_losses)
        best_k = min(mean_losses, key=mean_losses.get)

    pls = PLSRegression(n_components=best_k)
    pls.fit(X_train, y_train)
    T_train = pls.transform(X_train)
    T_test  = pls.transform(X_test)
    cal = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    cal.fit(T_train, y_train, sample_weight=sample_weight)
    prob = cal.predict_proba(T_test)[:, 1]
    coef_lat = cal.coef_.ravel()
    importance = (pls.x_weights_ @ coef_lat).ravel()
    return prob, importance, (pls, cal, best_k)


def fit_predict_plsda_weighted(X_train, y_train, X_test, groups=None, sample_weight=None):
    """
    PLS-DA using true sample-weighted NIPALS PLS (WeightedPLSRegression).

    Unlike fit_predict_plsda, sample weights enter the latent-variable
    extraction itself (centering, cross-covariance direction, loadings,
    deflation) via WeightedPLSRegression — not just the logistic calibrator.

    Used exclusively for QC spectrum-level variants (qc_only, qc_coral)
    where surviving spectra counts differ per patient and equal patient
    contribution must be enforced throughout the PLS fit.

    groups        : LOSO inner CV grouping (LeaveOneGroupOut).
    sample_weight : (n_train,) inverse-frequency weights from
                    equal_patient_weights(). Re-normalised per inner fold.
    """
    if not HAS_WEIGHTED_PLS:
        raise RuntimeError(
            "weighted_pls module not found — WeightedPLSRegression unavailable. "
            "Ensure weighted_pls.py is on the Python path."
        )

    feasible = _safe_max_components(X_train.shape[0], X_train.shape[1])
    if not feasible:
        raise ValueError(
            f"Cannot fit PLS-DA: n_train={X_train.shape[0]}, "
            f"n_features={X_train.shape[1]}, no feasible component count."
        )

    if len(feasible) == 1:
        best_k = feasible[0]
    else:
        if groups is not None:
            inner_cv = LeaveOneGroupOut()
            cv_splits = list(inner_cv.split(X_train, y_train, groups))
        else:
            inner_cv = LeaveOneOut()
            cv_splits = list(inner_cv.split(X_train))

        mean_losses = {}
        for k in feasible:
            fold_losses = []
            for tr_i, val_i in cv_splits:
                # Re-normalise weights to the inner-fold training subset
                sw_fold = None
                if sample_weight is not None:
                    sw_fold = sample_weight[tr_i].copy()
                    sw_fold = sw_fold * len(tr_i) / sw_fold.sum()
                pls_k = WeightedPLSRegression(n_components=k)
                pls_k.fit(X_train[tr_i], y_train[tr_i], sample_weight=sw_fold)
                T_tr  = pls_k.transform(X_train[tr_i])
                T_val = pls_k.transform(X_train[val_i])
                cal_k = LogisticRegression(max_iter=5000, solver="lbfgs",
                                           class_weight="balanced")
                cal_k.fit(T_tr, y_train[tr_i])
                prob_val = np.clip(cal_k.predict_proba(T_val), 1e-8, 1.0 - 1e-8)
                fold_losses.append(log_loss(y_train[val_i], prob_val, labels=[0, 1]))
            mean_losses[k] = np.mean(fold_losses)
        best_k = min(mean_losses, key=mean_losses.get)

    pls = WeightedPLSRegression(n_components=best_k)
    pls.fit(X_train, y_train, sample_weight=sample_weight)
    T_train = pls.transform(X_train)
    T_test  = pls.transform(X_test)
    cal = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
    cal.fit(T_train, y_train, sample_weight=sample_weight)
    prob = cal.predict_proba(T_test)[:, 1]
    coef_lat = cal.coef_.ravel()
    importance = (pls.x_weights_ @ coef_lat).ravel()
    return prob, importance, (pls, cal, best_k)


# =============================================================================
# OUTPUT WRITER
# =============================================================================
def write_outputs(
    outdir: Path, scenario_name: str, mode_name: str, model_name: str,
    preproc_name: str, methods: list[str], baseline_cfg: dict,
    wavenumbers: np.ndarray, y_true_pat: np.ndarray, y_score_pat: np.ndarray,
    y_pred_pat: np.ndarray, si_pat, importance_vec: np.ndarray,
    fast_cfg: dict,
    *,
    variant: str = "baseline",
    patient_ids=None, spectrum_patient_ids=None, spectrum_rep_idx=None,
    X_overlay_spec=None, y_true_spec=None, y_score_spec=None,
    y_pred_spec=None, si_spec=None, importance_is_fold_aggregated=False,
):
    ensure_dir(outdir)
    skip_plots   = fast_cfg.get("skip_plots", False)
    skip_roc     = fast_cfg.get("skip_roc", False)
    skip_overlay = fast_cfg.get("skip_overlay", False)
    skip_top_wn  = fast_cfg.get("skip_top_wn", False)
    max_spec_plot = fast_cfg.get("max_spectra_plot", 250)

    # ── Metrics CSV ──
    cm   = safe_confusion_matrix_2x2(y_true_pat, y_pred_pat)
    auc  = safe_auc(y_true_pat, y_score_pat)
    acc  = accuracy(y_true_pat, y_pred_pat)
    bacc = balanced_accuracy_present_classes(y_true_pat, y_pred_pat)

    metrics = pd.DataFrame([{
        "scenario": scenario_name, "mode": mode_name, "variant": variant,
        "model": model_name, "preprocessing": preproc_name,
        "baseline_enabled": bool(baseline_cfg.get("do_baseline", False)),
        "baseline_lam":   baseline_cfg.get("lam")   if baseline_cfg.get("do_baseline") else None,
        "baseline_p":     baseline_cfg.get("p")     if baseline_cfg.get("do_baseline") else None,
        "baseline_niter": baseline_cfg.get("niter") if baseline_cfg.get("do_baseline") else None,
        "auc": auc, "acc": acc, "balanced_acc": bacc,
        "n_patients": int(len(y_true_pat)),
        "n_classes_y_true": int(len(np.unique(y_true_pat))),
        "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1]),
    }])
    metrics.to_csv(outdir / "metrics.csv", index=False)

    # ── Predictions CSV ──
    pid = patient_ids if patient_ids is not None else np.arange(len(y_true_pat)).astype(str)
    pd.DataFrame({
        "patient_id": np.asarray(pid).astype(str),
        "y_true":  np.asarray(y_true_pat).astype(int),
        "y_score": np.asarray(y_score_pat).astype(float),
        "y_pred":  np.asarray(y_pred_pat).astype(int),
    }).to_csv(outdir / "predictions.csv", index=False)

    # ── Spectrum-level predictions CSV ──
    if y_true_spec is not None and y_score_spec is not None and y_pred_spec is not None:
        spid = spectrum_patient_ids
        if spid is None:
            base = patient_ids if patient_ids is not None else np.arange(int(len(y_true_spec) / N_REP)).astype(str)
            spid = np.repeat(np.asarray(base).astype(str), N_REP)
        ridx = spectrum_rep_idx
        if ridx is None:
            ridx = np.tile(np.arange(N_REP), int(len(y_true_spec) / N_REP))
        spid = np.asarray(spid).astype(str)
        ridx = np.asarray(ridx).astype(int)
        spectrum_id = np.char.add(np.char.add(spid, "_rep"), ridx.astype(str))
        pd.DataFrame({
            "spectrum_row": np.arange(len(spectrum_id)).astype(int),
            "patient_id": spid, "rep_idx": ridx, "spectrum_id": spectrum_id,
            "y_true":  np.asarray(y_true_spec).astype(int),
            "y_score": np.asarray(y_score_spec).astype(float),
            "y_pred":  np.asarray(y_pred_spec).astype(int),
        }).to_csv(outdir / "predictions_spectrum.csv", index=False)

    # ── Plots ──
    if not skip_plots:
        tag = f"{scenario_name}|{mode_name}|{variant}|{model_name}|{preproc_name}"
        save_confusion_plot(cm, outdir / "confusion.png", f"Confusion|{tag}")
        save_score_hist(y_true_pat, y_score_pat, outdir / "score_hist.png", f"ScoreHist|{tag}")
        if not skip_roc:
            save_roc_or_note(y_true_pat, y_score_pat, outdir, f"ROC|{tag}")
        save_predprob_parity(y_true_pat, y_score_pat, outdir / "predprob_parity.png",
                             f"PredProb|{tag}", seed=RANDOM_STATE)
        if si_pat is not None:
            save_predprob_parity_si_gradient(
                y_true_pat, y_score_pat, si_pat,
                outdir / "predprob_parity_SIgradient.png",
                f"PredProb SI|{tag}", seed=RANDOM_STATE)
        else:
            write_text(outdir / "predprob_parity_SIgradient_NA.txt",
                       "SI not available; SI-gradient plot not produced.")
        if y_true_spec is not None and y_score_spec is not None:
            save_predprob_parity(
                y_true_spec, y_score_spec, outdir / "predprob_parity_spectrum.png",
                f"PredProb Spectrum|{tag}", seed=RANDOM_STATE)
            if si_spec is not None:
                save_predprob_parity_si_gradient(
                    y_true_spec, y_score_spec, si_spec,
                    outdir / "predprob_parity_spectrum_SIgradient.png",
                    f"PredProb Spectrum SI|{tag}", seed=RANDOM_STATE)
        save_importance_spectrogram(
            wavenumbers, importance_vec, outdir / "importance_spectrogram.png",
            f"|Imp| vs WN|{tag}" + (" FoldAgg" if importance_is_fold_aggregated else ""))

    # ── Top wavenumber export ──
    if not skip_top_wn:
        export_top_wavenumbers(wavenumbers, importance_vec, outdir / "top_wavenumbers.csv", top_n=TOP_N_WN)

    # ── Overlay plot ──
    if not skip_plots and not skip_overlay and X_overlay_spec is not None:
        try:
            save_overlay_spectra_with_highlighted_wavenumbers(
                wavenumbers=wavenumbers, X_spec=X_overlay_spec, importance=importance_vec,
                outpath=outdir / "spectra_overlay_highlight_topWN.png",
                title=f"Overlay+TopWN|{scenario_name}|{mode_name}|{variant}|{model_name}|{preproc_name}",
                top_k=HIGHLIGHT_TOP_K_WN, max_spectra_to_plot=max_spec_plot, seed=RANDOM_STATE)
        except Exception as e:
            logger.warning(f"overlay plot failed in {outdir}: {e}")

    # ── Run info JSON ──
    write_json(outdir / "run_info.json", {
        "scenario": scenario_name, "mode": mode_name, "variant": variant,
        "model": model_name, "preprocessing": preproc_name,
        "methods": methods, "baseline_cfg": baseline_cfg, "N_REP": N_REP,
        "PLS_COMPONENT_RANGE": PLS_COMPONENT_RANGE if model_name == "plsda" else None,
        "importance_fold_aggregated": bool(importance_is_fold_aggregated),
    })


# =============================================================================
# SCENARIO FUNCTIONS — BASELINE (original logic, N_REP fixed at 9)
# =============================================================================

def scenario_box12_loso_patient_avg(X3_tr, y_tr, si_tr, id_tr, wavenumbers,
                                    methods, baseline_cfg, outdir, model_name,
                                    fast_cfg, variant="baseline"):
    loo = LeaveOneOut(); n_pat = len(y_tr)
    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    fold_imps = []
    for tr_idx, te_idx in loo.split(np.arange(n_pat)):
        Xtr_spec = X3_tr[tr_idx].reshape(-1, X3_tr.shape[2]).T
        Xte_spec = X3_tr[te_idx].reshape(-1, X3_tr.shape[2]).T
        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)
        Xtr_pat = Xtr_p.T.reshape(len(tr_idx), N_REP, -1).mean(axis=1)
        Xte_pat = Xte_p.T.reshape(1, N_REP, -1).mean(axis=1)
        if model_name == "svm":
            score, imp, _ = fit_predict_svm_linear(Xtr_pat, y_tr[tr_idx], Xte_pat)
        elif model_name == "plsda":
            score, imp, _ = fit_predict_plsda(Xtr_pat, y_tr[tr_idx], Xte_pat)
        else:
            raise ValueError(model_name)
        s = float(np.ravel(score)[0])
        y_score_pat[te_idx] = s; y_pred_pat[te_idx] = int(s >= 0.5)
        fold_imps.append(np.abs(imp))
    imp_mean = np.mean(np.vstack(fold_imps), axis=0)
    Xfull_spec = X3_tr.reshape(-1, X3_tr.shape[2]).T
    Xfull_p, _ = apply_preproc_leakage_safe(Xfull_spec, Xfull_spec, methods)
    write_outputs(outdir=outdir, scenario_name="Scenario_Box12_LOSO", mode_name="PATIENT_AVG",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_tr,
                  y_score_pat=y_score_pat, y_pred_pat=y_pred_pat, si_pat=si_tr,
                  importance_vec=imp_mean, patient_ids=id_tr, X_overlay_spec=Xfull_p,
                  importance_is_fold_aggregated=True, fast_cfg=fast_cfg, variant=variant)


def scenario_box12_loso_spectrum_level(X3_tr, y_tr, si_tr, id_tr, wavenumbers,
                                       methods, baseline_cfg, outdir, model_name,
                                       fast_cfg, variant="baseline"):
    loo = LeaveOneOut(); n_pat = len(y_tr)
    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    y_true_spec_all, y_score_spec_all, y_pred_spec_all, si_spec_all = [], [], [], []
    spec_pid_all, spec_rep_all = [], []
    fold_imps = []
    for tr_idx, te_idx in loo.split(np.arange(n_pat)):
        Xtr_spec = X3_tr[tr_idx].reshape(-1, X3_tr.shape[2]).T
        Xte_spec = X3_tr[te_idx].reshape(-1, X3_tr.shape[2]).T
        ytr_spec = np.repeat(y_tr[tr_idx], N_REP)
        yte_spec = np.repeat(y_tr[te_idx], N_REP)
        si_te_spec = np.repeat(si_tr[te_idx], N_REP) if si_tr is not None else None
        spec_pid_all.append(np.repeat(id_tr[te_idx], N_REP))
        spec_rep_all.append(np.arange(N_REP))
        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)
        if model_name == "svm":
            score_spec, imp, _ = fit_predict_svm_linear(Xtr_p.T, ytr_spec, Xte_p.T)
        elif model_name == "plsda":
            groups_inner = np.repeat(np.arange(len(tr_idx)), N_REP)
            score_spec, imp, _ = fit_predict_plsda(Xtr_p.T, ytr_spec, Xte_p.T, groups=groups_inner)
        else:
            raise ValueError(model_name)
        score_spec = np.asarray(score_spec).ravel()
        s_pat = float(np.mean(score_spec))
        y_score_pat[te_idx] = s_pat; y_pred_pat[te_idx] = int(s_pat >= 0.5)
        y_true_spec_all.append(yte_spec); y_score_spec_all.append(score_spec)
        y_pred_spec_all.append((score_spec >= 0.5).astype(int))
        if si_te_spec is not None:
            si_spec_all.append(np.asarray(si_te_spec).ravel())
        fold_imps.append(np.abs(imp))
    imp_mean = np.mean(np.vstack(fold_imps), axis=0)
    y_true_spec  = np.concatenate(y_true_spec_all)
    y_score_spec = np.concatenate(y_score_spec_all)
    y_pred_spec  = np.concatenate(y_pred_spec_all)
    si_spec = np.concatenate(si_spec_all) if si_spec_all else None
    Xfull_spec = X3_tr.reshape(-1, X3_tr.shape[2]).T
    Xfull_p, _ = apply_preproc_leakage_safe(Xfull_spec, Xfull_spec, methods)
    write_outputs(outdir=outdir, scenario_name="Scenario_Box12_LOSO", mode_name="SPECTRUM_LEVEL",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_tr,
                  y_score_pat=y_score_pat, y_pred_pat=y_pred_pat, si_pat=si_tr,
                  importance_vec=imp_mean, patient_ids=id_tr,
                  spectrum_patient_ids=np.concatenate(spec_pid_all),
                  spectrum_rep_idx=np.concatenate(spec_rep_all),
                  X_overlay_spec=Xfull_p, y_true_spec=y_true_spec,
                  y_score_spec=y_score_spec, y_pred_spec=y_pred_spec, si_spec=si_spec,
                  importance_is_fold_aggregated=True, fast_cfg=fast_cfg, variant=variant)


def scenario_train12_predict3_patient_avg(
    X3_tr, y_tr, si_tr, X3_te, y_te, si_te, id_te,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    use_coral=False, variant="baseline",
):
    """
    Train on Box12, predict Box3 — patient-average mode.
    use_coral : if True, apply CORAL to Box3 test spectra using Box12 DM1 as source.
    """
    Xtr_spec = X3_tr.reshape(-1, X3_tr.shape[2]).T   # (n_w, 9*n_tr)
    Xte_spec = X3_te.reshape(-1, X3_te.shape[2]).T   # (n_w, 9*n_te)
    Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)

    if use_coral:
        # Source = Box12 DM1 spectra only (post-preprocessing)
        dm1_spec_mask = np.repeat(y_tr == 1, N_REP)  # (9*n_tr,)
        Xte_p = _apply_coral(Xtr_p, dm1_spec_mask, Xte_p)

    Xtr_pat = Xtr_p.T.reshape(len(y_tr), N_REP, -1).mean(axis=1)
    Xte_pat = Xte_p.T.reshape(len(y_te), N_REP, -1).mean(axis=1)

    if model_name == "svm":
        y_score, imp, _ = fit_predict_svm_linear(Xtr_pat, y_tr, Xte_pat)
    elif model_name == "plsda":
        y_score, imp, _ = fit_predict_plsda(Xtr_pat, y_tr, Xte_pat)
    else:
        raise ValueError(model_name)

    y_score = np.asarray(y_score).ravel()
    y_pred  = (y_score >= 0.5).astype(int)
    Xoverlay = np.concatenate([Xtr_p, Xte_p], axis=1)
    write_outputs(outdir=outdir, scenario_name="Scenario_Train12_Predict3", mode_name="PATIENT_AVG",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_te,
                  y_score_pat=y_score, y_pred_pat=y_pred, si_pat=si_te,
                  importance_vec=imp, patient_ids=id_te, X_overlay_spec=Xoverlay,
                  importance_is_fold_aggregated=False, fast_cfg=fast_cfg, variant=variant)


def scenario_train12_predict3_spectrum_level(
    X3_tr, y_tr, si_tr, X3_te, y_te, si_te, id_te,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    use_coral=False, variant="baseline",
):
    """
    Train on Box12, predict Box3 — spectrum-level mode.
    use_coral : if True, apply CORAL to Box3 test spectra using Box12 DM1 as source.
    """
    Xtr_spec = X3_tr.reshape(-1, X3_tr.shape[2]).T
    Xte_spec = X3_te.reshape(-1, X3_te.shape[2]).T
    ytr_spec = np.repeat(y_tr, N_REP)
    yte_spec = np.repeat(y_te, N_REP)
    si_te_spec = np.repeat(si_te, N_REP) if si_te is not None else None
    Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)

    if use_coral:
        dm1_spec_mask = np.repeat(y_tr == 1, N_REP)
        Xte_p = _apply_coral(Xtr_p, dm1_spec_mask, Xte_p)

    if model_name == "svm":
        score_spec, imp, _ = fit_predict_svm_linear(Xtr_p.T, ytr_spec, Xte_p.T)
    elif model_name == "plsda":
        groups_inner = np.repeat(np.arange(len(y_tr)), N_REP)
        score_spec, imp, _ = fit_predict_plsda(Xtr_p.T, ytr_spec, Xte_p.T, groups=groups_inner)
    else:
        raise ValueError(model_name)

    score_spec = np.asarray(score_spec).ravel()
    y_pred_spec = (score_spec >= 0.5).astype(int)
    score_pat = score_spec.reshape(len(y_te), N_REP).mean(axis=1)
    y_pred_pat = (score_pat >= 0.5).astype(int)
    write_outputs(outdir=outdir, scenario_name="Scenario_Train12_Predict3", mode_name="SPECTRUM_LEVEL",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_te,
                  y_score_pat=score_pat, y_pred_pat=y_pred_pat, si_pat=si_te,
                  importance_vec=imp, patient_ids=id_te,
                  spectrum_patient_ids=np.repeat(id_te, N_REP),
                  spectrum_rep_idx=np.tile(np.arange(N_REP), len(id_te)),
                  y_true_spec=yte_spec, y_score_spec=score_spec, y_pred_spec=y_pred_spec,
                  si_spec=si_te_spec, importance_is_fold_aggregated=False,
                  fast_cfg=fast_cfg, variant=variant)


def scenario_box123_loso_patient_avg(X3_all, y_all, si_all, id_all, wavenumbers,
                                     methods, baseline_cfg, outdir, model_name,
                                     fast_cfg, variant="baseline"):
    loo = LeaveOneOut(); n_pat = len(y_all)
    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    fold_imps = []
    for tr_idx, te_idx in loo.split(np.arange(n_pat)):
        Xtr_spec = X3_all[tr_idx].reshape(-1, X3_all.shape[2]).T
        Xte_spec = X3_all[te_idx].reshape(-1, X3_all.shape[2]).T
        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)
        Xtr_pat = Xtr_p.T.reshape(len(tr_idx), N_REP, -1).mean(axis=1)
        Xte_pat = Xte_p.T.reshape(1, N_REP, -1).mean(axis=1)
        if model_name == "svm":
            score, imp, _ = fit_predict_svm_linear(Xtr_pat, y_all[tr_idx], Xte_pat)
        elif model_name == "plsda":
            score, imp, _ = fit_predict_plsda(Xtr_pat, y_all[tr_idx], Xte_pat)
        else:
            raise ValueError(model_name)
        s = float(np.ravel(score)[0])
        y_score_pat[te_idx] = s; y_pred_pat[te_idx] = int(s >= 0.5)
        fold_imps.append(np.abs(imp))
    imp_mean = np.mean(np.vstack(fold_imps), axis=0)
    Xfull_spec = X3_all.reshape(-1, X3_all.shape[2]).T
    Xfull_p, _ = apply_preproc_leakage_safe(Xfull_spec, Xfull_spec, methods)
    write_outputs(outdir=outdir, scenario_name="Scenario_Box123_LOSO", mode_name="PATIENT_AVG",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_all,
                  y_score_pat=y_score_pat, y_pred_pat=y_pred_pat, si_pat=si_all,
                  importance_vec=imp_mean, patient_ids=id_all, X_overlay_spec=Xfull_p,
                  importance_is_fold_aggregated=True, fast_cfg=fast_cfg, variant=variant)


def scenario_box123_loso_spectrum_level(X3_all, y_all, si_all, id_all, wavenumbers,
                                        methods, baseline_cfg, outdir, model_name,
                                        fast_cfg, variant="baseline"):
    loo = LeaveOneOut(); n_pat = len(y_all)
    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    y_true_spec_all, y_score_spec_all, y_pred_spec_all, si_spec_all = [], [], [], []
    spec_pid_all, spec_rep_all = [], []
    fold_imps = []
    for tr_idx, te_idx in loo.split(np.arange(n_pat)):
        Xtr_spec = X3_all[tr_idx].reshape(-1, X3_all.shape[2]).T
        Xte_spec = X3_all[te_idx].reshape(-1, X3_all.shape[2]).T
        ytr_spec = np.repeat(y_all[tr_idx], N_REP)
        yte_spec = np.repeat(y_all[te_idx], N_REP)
        si_te_spec = np.repeat(si_all[te_idx], N_REP) if si_all is not None else None
        spec_pid_all.append(np.repeat(id_all[te_idx], N_REP))
        spec_rep_all.append(np.arange(N_REP))
        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)
        if model_name == "svm":
            score_spec, imp, _ = fit_predict_svm_linear(Xtr_p.T, ytr_spec, Xte_p.T)
        elif model_name == "plsda":
            groups_inner = np.repeat(np.arange(len(tr_idx)), N_REP)
            score_spec, imp, _ = fit_predict_plsda(Xtr_p.T, ytr_spec, Xte_p.T, groups=groups_inner)
        else:
            raise ValueError(model_name)
        score_spec = np.asarray(score_spec).ravel()
        s_pat = float(np.mean(score_spec))
        y_score_pat[te_idx] = s_pat; y_pred_pat[te_idx] = int(s_pat >= 0.5)
        y_true_spec_all.append(yte_spec); y_score_spec_all.append(score_spec)
        y_pred_spec_all.append((score_spec >= 0.5).astype(int))
        if si_te_spec is not None:
            si_spec_all.append(np.asarray(si_te_spec).ravel())
        fold_imps.append(np.abs(imp))
    imp_mean = np.mean(np.vstack(fold_imps), axis=0)
    y_true_spec  = np.concatenate(y_true_spec_all)
    y_score_spec = np.concatenate(y_score_spec_all)
    y_pred_spec  = np.concatenate(y_pred_spec_all)
    si_spec = np.concatenate(si_spec_all) if si_spec_all else None
    Xfull_spec = X3_all.reshape(-1, X3_all.shape[2]).T
    Xfull_p, _ = apply_preproc_leakage_safe(Xfull_spec, Xfull_spec, methods)
    write_outputs(outdir=outdir, scenario_name="Scenario_Box123_LOSO", mode_name="SPECTRUM_LEVEL",
                  model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
                  baseline_cfg=baseline_cfg, wavenumbers=wavenumbers, y_true_pat=y_all,
                  y_score_pat=y_score_pat, y_pred_pat=y_pred_pat, si_pat=si_all,
                  importance_vec=imp_mean, patient_ids=id_all,
                  spectrum_patient_ids=np.concatenate(spec_pid_all),
                  spectrum_rep_idx=np.concatenate(spec_rep_all),
                  X_overlay_spec=Xfull_p, y_true_spec=y_true_spec,
                  y_score_spec=y_score_spec, y_pred_spec=y_pred_spec, si_spec=si_spec,
                  importance_is_fold_aggregated=True, fast_cfg=fast_cfg, variant=variant)


# =============================================================================
# SCENARIO FUNCTIONS — QC (variable spectra per patient)
# =============================================================================

def _loso_patient_avg_qc_core(
    X_spec, spec_sids, y, si, ids,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    scenario_name: str, variant: str,
):
    """
    LOSO patient-average with QC-filtered data.

    Key properties:
    - Each held-out patient contributes exactly one averaged feature vector.
    - Averaging uses only QC-surviving spectra for that patient.
    - LOSO split is at the patient level (hold out all spectra of one patient).
    - EMSC reference is computed from training spectra only (leakage-free).

    X_spec    : (n_w, k_total) — features-first, only QC-kept spectra
    spec_sids : (k_total,) str — FilePrefix per spectrum column
    y         : (n_pat,) int — labels aligned to ids
    ids       : (n_pat,) str — ordered patient FilePrefix IDs
    """
    n_pat = len(ids)
    id_to_label = {ids[j]: int(y[j]) for j in range(n_pat)}

    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    fold_imps = []

    for i, te_id in enumerate(ids):
        te_mask = (spec_sids == te_id)
        tr_mask = ~te_mask
        tr_pat_indices = [j for j in range(n_pat) if j != i]
        tr_ids = ids[tr_pat_indices]

        Xtr_spec = X_spec[:, tr_mask]   # (n_w, k_tr)
        Xte_spec = X_spec[:, te_mask]   # (n_w, k_i) — k_i QC-surviving spectra for patient i
        tr_sids_fold = spec_sids[tr_mask]

        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)

        # Patient-average: each patient contributes one vector (equal weighting by design)
        Xtr_pat = build_patient_avg(Xtr_p, tr_sids_fold, tr_ids)   # (n_tr_pat, n_w)
        Xte_pat = Xte_p.T.mean(axis=0, keepdims=True)               # (1, n_w)

        y_tr_fold = y[tr_pat_indices]

        if model_name == "svm":
            score, imp, _ = fit_predict_svm_linear(Xtr_pat, y_tr_fold, Xte_pat)
        elif model_name == "plsda":
            score, imp, _ = fit_predict_plsda(Xtr_pat, y_tr_fold, Xte_pat)
        else:
            raise ValueError(model_name)

        s = float(np.ravel(score)[0])
        y_score_pat[i] = s
        y_pred_pat[i]  = int(s >= 0.5)
        fold_imps.append(np.abs(imp))

    imp_mean = np.mean(np.vstack(fold_imps), axis=0)

    # Overlay: reference EMSC from all spectra (visualisation only, not used in model)
    Xfull_p, _ = apply_preproc_leakage_safe(X_spec, X_spec, methods)

    write_outputs(
        outdir=outdir, scenario_name=scenario_name, mode_name="PATIENT_AVG",
        model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
        baseline_cfg=baseline_cfg, wavenumbers=wavenumbers,
        y_true_pat=y, y_score_pat=y_score_pat, y_pred_pat=y_pred_pat,
        si_pat=si, importance_vec=imp_mean, patient_ids=ids,
        X_overlay_spec=Xfull_p, importance_is_fold_aggregated=True,
        fast_cfg=fast_cfg, variant=variant,
    )


def _loso_spectrum_level_qc_core(
    X_spec, spec_sids, y, si, ids,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    scenario_name: str, variant: str,
):
    """
    LOSO spectrum-level with QC-filtered data.

    Equal patient weighting in training:
    - SVM  : sample_weight = 1/k_i per spectrum (normalized), passed to SVC.fit()
    - PLSDA: PLS projection fit equally on all spectra; LogisticRegression
             calibrator weighted by 1/k_i so the decision boundary does not
             favour patients with more surviving spectra.

    Test prediction:
    - All k_i QC-surviving spectra of the held-out patient are scored.
    - Their mean probability is the patient-level score (collapsed before metrics).

    X_spec    : (n_w, k_total) features-first
    spec_sids : (k_total,) str — patient FilePrefix per spectrum column
    y         : (n_pat,) int — labels
    ids       : (n_pat,) str — ordered patient IDs
    """
    n_pat = len(ids)

    y_score_pat = np.zeros(n_pat, dtype=float)
    y_pred_pat  = np.zeros(n_pat, dtype=int)
    y_true_spec_all, y_score_spec_all, y_pred_spec_all, si_spec_all = [], [], [], []
    spec_pid_all, spec_rep_all = [], []
    fold_imps = []

    id_to_label = {ids[j]: int(y[j]) for j in range(n_pat)}

    for i, te_id in enumerate(ids):
        te_mask = (spec_sids == te_id)
        tr_mask = ~te_mask
        tr_pat_indices = [j for j in range(n_pat) if j != i]
        tr_ids = ids[tr_pat_indices]

        Xtr_spec = X_spec[:, tr_mask]
        Xte_spec = X_spec[:, te_mask]
        tr_sids_fold = spec_sids[tr_mask]

        # Per-spectrum labels for training
        ytr_spec = np.array([id_to_label[sid] for sid in tr_sids_fold])
        yte_spec = np.repeat(int(y[i]), int(te_mask.sum()))

        Xtr_p, Xte_p = apply_preproc_leakage_safe(Xtr_spec, Xte_spec, methods)

        # Equal-patient sample weights: patient with k_i spectra gets weight 1/k_i
        # per spectrum, normalized so total weight == n_training_patients.
        sw_tr = equal_patient_weights(tr_sids_fold, tr_ids)

        # Integer patient group labels for PLS-DA inner CV
        groups_tr = build_groups_from_sids(tr_sids_fold, tr_ids)

        if model_name == "svm":
            score_spec, imp, _ = fit_predict_svm_linear(
                Xtr_p.T, ytr_spec, Xte_p.T, sample_weight=sw_tr)
        elif model_name == "plsda":
            # True weighted NIPALS PLS: weights enter latent-variable extraction,
            # not just the logistic calibrator.
            score_spec, imp, _ = fit_predict_plsda_weighted(
                Xtr_p.T, ytr_spec, Xte_p.T,
                groups=groups_tr, sample_weight=sw_tr)
        else:
            raise ValueError(model_name)

        score_spec = np.asarray(score_spec).ravel()

        # Collapse k_i test scores to a single patient probability
        s_pat = float(score_spec.mean())
        y_score_pat[i] = s_pat
        y_pred_pat[i]  = int(s_pat >= 0.5)

        n_te_spec = int(te_mask.sum())
        y_true_spec_all.append(yte_spec)
        y_score_spec_all.append(score_spec)
        y_pred_spec_all.append((score_spec >= 0.5).astype(int))
        spec_pid_all.append(np.full(n_te_spec, te_id, dtype=object))
        spec_rep_all.append(np.arange(n_te_spec))

        if si is not None:
            si_spec_all.append(np.repeat(float(si[i]), n_te_spec))

        fold_imps.append(np.abs(imp))

    imp_mean = np.mean(np.vstack(fold_imps), axis=0)
    y_true_spec  = np.concatenate(y_true_spec_all)
    y_score_spec = np.concatenate(y_score_spec_all)
    y_pred_spec  = np.concatenate(y_pred_spec_all)
    si_spec = np.concatenate(si_spec_all) if si_spec_all else None

    Xfull_p, _ = apply_preproc_leakage_safe(X_spec, X_spec, methods)

    write_outputs(
        outdir=outdir, scenario_name=scenario_name, mode_name="SPECTRUM_LEVEL",
        model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
        baseline_cfg=baseline_cfg, wavenumbers=wavenumbers,
        y_true_pat=y, y_score_pat=y_score_pat, y_pred_pat=y_pred_pat,
        si_pat=si, importance_vec=imp_mean, patient_ids=ids,
        spectrum_patient_ids=np.concatenate(spec_pid_all).astype(str),
        spectrum_rep_idx=np.concatenate(spec_rep_all).astype(int),
        X_overlay_spec=Xfull_p, y_true_spec=y_true_spec,
        y_score_spec=y_score_spec, y_pred_spec=y_pred_spec, si_spec=si_spec,
        importance_is_fold_aggregated=True, fast_cfg=fast_cfg, variant=variant,
    )


def scenario_train12_predict3_patient_avg_qc(
    qc_tr: dict, qc_te: dict,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    use_coral: bool = False, variant: str = "qc_only",
):
    """
    Train on Box12, predict Box3 — patient-average mode with QC-filtered data.

    use_coral : True for qc_coral — apply CORAL after preprocessing.
                CORAL source = Box12 DM1 QC-surviving spectra only.
                CORAL target = all Box3 QC-surviving spectra.

    Patient-average: each patient contributes exactly one averaged vector
    (from their QC-surviving spectra), so all patients are equally weighted.
    """
    X_spec_tr  = qc_tr["X_spec"]
    sids_tr    = qc_tr["spec_sids"]
    y_tr       = qc_tr["y"]
    si_tr      = qc_tr["si"]
    ids_tr     = qc_tr["ids"]

    X_spec_te  = qc_te["X_spec"]
    sids_te    = qc_te["spec_sids"]
    y_te       = qc_te["y"]
    si_te      = qc_te["si"]
    ids_te     = qc_te["ids"]

    Xtr_p, Xte_p = apply_preproc_leakage_safe(X_spec_tr, X_spec_te, methods)

    if use_coral:
        # CORAL source: Box12 DM1 spectra only (post-preprocessing)
        id_to_label_tr = {ids_tr[j]: int(y_tr[j]) for j in range(len(ids_tr))}
        dm1_mask = np.array([id_to_label_tr.get(sid, 0) == 1 for sid in sids_tr])
        Xte_p = _apply_coral(Xtr_p, dm1_mask, Xte_p)

    # Patient averages (equal weighting by averaging over QC-surviving spectra)
    Xtr_pat = build_patient_avg(Xtr_p, sids_tr, ids_tr)   # (n_tr_pat, n_w)
    Xte_pat = build_patient_avg(Xte_p, sids_te, ids_te)   # (n_te_pat, n_w)

    if model_name == "svm":
        y_score, imp, _ = fit_predict_svm_linear(Xtr_pat, y_tr, Xte_pat)
    elif model_name == "plsda":
        y_score, imp, _ = fit_predict_plsda(Xtr_pat, y_tr, Xte_pat)
    else:
        raise ValueError(model_name)

    y_score = np.asarray(y_score).ravel()
    y_pred  = (y_score >= 0.5).astype(int)
    Xoverlay = np.concatenate([Xtr_p, Xte_p], axis=1)

    write_outputs(
        outdir=outdir, scenario_name="Scenario_Train12_Predict3", mode_name="PATIENT_AVG",
        model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
        baseline_cfg=baseline_cfg, wavenumbers=wavenumbers,
        y_true_pat=y_te, y_score_pat=y_score, y_pred_pat=y_pred,
        si_pat=si_te, importance_vec=imp, patient_ids=ids_te,
        X_overlay_spec=Xoverlay, importance_is_fold_aggregated=False,
        fast_cfg=fast_cfg, variant=variant,
    )


def scenario_train12_predict3_spectrum_level_qc(
    qc_tr: dict, qc_te: dict,
    wavenumbers, methods, baseline_cfg, outdir, model_name, fast_cfg,
    use_coral: bool = False, variant: str = "qc_only",
):
    """
    Train on Box12, predict Box3 — spectrum-level mode with QC-filtered data.

    Equal patient weighting in training:
    - SVM: sample weights 1/k_i per spectrum normalized to n_training_patients
    - PLSDA: sample_weight applied to the LogisticRegression calibrator

    Test prediction collapse:
    - All QC-surviving spectra for each test patient are scored independently.
    - Patient-level probability = mean of per-spectrum probabilities.

    CORAL (if use_coral):
    - Source = Box12 DM1 QC-surviving spectra (post-preprocessing)
    - Target = all Box3 QC-surviving spectra (post-preprocessing)
    """
    X_spec_tr  = qc_tr["X_spec"]
    sids_tr    = qc_tr["spec_sids"]
    y_tr       = qc_tr["y"]
    si_tr      = qc_tr["si"]
    ids_tr     = qc_tr["ids"]

    X_spec_te  = qc_te["X_spec"]
    sids_te    = qc_te["spec_sids"]
    y_te       = qc_te["y"]
    si_te      = qc_te["si"]
    ids_te     = qc_te["ids"]

    Xtr_p, Xte_p = apply_preproc_leakage_safe(X_spec_tr, X_spec_te, methods)

    if use_coral:
        id_to_label_tr = {ids_tr[j]: int(y_tr[j]) for j in range(len(ids_tr))}
        dm1_mask = np.array([id_to_label_tr.get(sid, 0) == 1 for sid in sids_tr])
        Xte_p = _apply_coral(Xtr_p, dm1_mask, Xte_p)

    # Per-spectrum labels for training
    id_to_label_tr = {ids_tr[j]: int(y_tr[j]) for j in range(len(ids_tr))}
    ytr_spec = np.array([id_to_label_tr[sid] for sid in sids_tr])

    id_to_label_te = {ids_te[j]: int(y_te[j]) for j in range(len(ids_te))}
    yte_spec = np.array([id_to_label_te[sid] for sid in sids_te])

    # Equal patient weights for training (inverse-frequency, normalized)
    sw_tr = equal_patient_weights(sids_tr, ids_tr)
    groups_tr = build_groups_from_sids(sids_tr, ids_tr)

    if model_name == "svm":
        score_spec, imp, _ = fit_predict_svm_linear(
            Xtr_p.T, ytr_spec, Xte_p.T, sample_weight=sw_tr)
    elif model_name == "plsda":
        # True weighted NIPALS PLS: weights enter latent-variable extraction,
        # not just the logistic calibrator.
        score_spec, imp, _ = fit_predict_plsda_weighted(
            Xtr_p.T, ytr_spec, Xte_p.T,
            groups=groups_tr, sample_weight=sw_tr)
    else:
        raise ValueError(model_name)

    score_spec = np.asarray(score_spec).ravel()
    y_pred_spec = (score_spec >= 0.5).astype(int)

    # Collapse spectrum-level scores to patient-level
    # (variable spectra per patient — collapse by mean over each patient's spectra)
    score_pat  = np.zeros(len(ids_te), dtype=float)
    y_pred_pat = np.zeros(len(ids_te), dtype=int)
    for j, te_sid in enumerate(ids_te):
        mask = (sids_te == te_sid)
        score_pat[j] = float(score_spec[mask].mean())
        y_pred_pat[j] = int(score_pat[j] >= 0.5)

    # Build spectrum-level SI array (repeat each patient's SI for their spectra)
    si_spec = None
    if si_te is not None:
        id_to_si_te = {ids_te[j]: float(si_te[j]) for j in range(len(ids_te))}
        si_spec = np.array([id_to_si_te[sid] for sid in sids_te], dtype=float)

    Xoverlay = np.concatenate([Xtr_p, Xte_p], axis=1)

    write_outputs(
        outdir=outdir, scenario_name="Scenario_Train12_Predict3", mode_name="SPECTRUM_LEVEL",
        model_name=model_name, preproc_name=outdir.parts[-2], methods=methods,
        baseline_cfg=baseline_cfg, wavenumbers=wavenumbers,
        y_true_pat=y_te, y_score_pat=score_pat, y_pred_pat=y_pred_pat,
        si_pat=si_te, importance_vec=imp, patient_ids=ids_te,
        spectrum_patient_ids=sids_te.astype(str),
        spectrum_rep_idx=np.concatenate([
            np.arange(int((sids_te == sid).sum())) for sid in ids_te
        ]).astype(int),
        y_true_spec=yte_spec, y_score_spec=score_spec, y_pred_spec=y_pred_spec,
        si_spec=si_spec, X_overlay_spec=Xoverlay,
        importance_is_fold_aggregated=False, fast_cfg=fast_cfg, variant=variant,
    )


# =============================================================================
# COMBO GENERATION + WORKER
# =============================================================================
def generate_combos(preproc_names, model_names, scenario_names, variant_names):
    """Yield (preproc_name, model_name, scenario_key, variant) tuples."""
    for pre in preproc_names:
        for mdl in model_names:
            for sc in scenario_names:
                for var in variant_names:
                    if var in ALL_SCENARIO_VARIANTS.get(sc, []):
                        yield (pre, mdl, sc, var)


def _scenario_key_to_path_parts(key: str, variant: str):
    """Convert scenario key + variant -> (scenario_dir, mode_dir, variant_dir)."""
    mapping = {
        "Box12_LOSO_PATIENT_AVG":           ("Scenario_Box12_LOSO",       "PATIENT_AVG"),
        "Box12_LOSO_SPECTRUM_LEVEL":        ("Scenario_Box12_LOSO",       "SPECTRUM_LEVEL"),
        "Train12_Predict3_PATIENT_AVG":     ("Scenario_Train12_Predict3", "PATIENT_AVG"),
        "Train12_Predict3_SPECTRUM_LEVEL":  ("Scenario_Train12_Predict3", "SPECTRUM_LEVEL"),
        "Box123_LOSO_PATIENT_AVG":          ("Scenario_Box123_LOSO",      "PATIENT_AVG"),
        "Box123_LOSO_SPECTRUM_LEVEL":       ("Scenario_Box123_LOSO",      "SPECTRUM_LEVEL"),
    }
    scenario_dir, mode_dir = mapping[key]
    return scenario_dir, mode_dir, variant


def run_one_combo(combo, shared_data, out_root, fast_cfg, force):
    """
    Worker for one (preproc, model, scenario, variant) combination.
    Returns dict with combo tag + status + timing.
    """
    pre_name, model_name, scenario_key, variant = combo
    scenario_dir, mode_dir, variant_dir = _scenario_key_to_path_parts(scenario_key, variant)
    # Output path includes variant level: <scenario>/<mode>/<variant>/<preproc>/<model>
    outdir = Path(out_root) / scenario_dir / mode_dir / variant_dir / pre_name / model_name
    tag = f"{pre_name}|{model_name}|{scenario_key}|{variant}"

    if not force and (outdir / "metrics.csv").exists():
        return {"combo": tag, "status": "SKIPPED", "time_s": 0.0, "error": None}

    t0 = time.time()
    try:
        cfg = ALL_PREPROC[pre_name]
        methods = cfg["methods"]
        baseline_cfg = cfg["baseline"]
        use_baseline = baseline_cfg.get("do_baseline", False)

        ensure_dir(outdir)

        # ── Select baseline (non-QC) data arrays ──────────────────────────
        X3_tr_use  = shared_data["X3_tr_bc"]  if use_baseline else shared_data["X3_tr"]
        X3_te_use  = shared_data["X3_te_bc"]  if use_baseline else shared_data["X3_te"]
        X3_all_use = shared_data["X3_all_bc"] if use_baseline else shared_data["X3_all"]

        # ── Select QC data dict (if variant requires it) ───────────────────
        def _pick_qc(key):
            """Return the QC cohort dict with the right X_spec version."""
            d = shared_data.get(key)
            if d is None:
                raise RuntimeError(
                    f"QC data not loaded for '{key}'. "
                    "Pass --qc-metrics flag pointing to QC_metrics_per_spectrum.csv."
                )
            raw_key = "X_spec_bc" if use_baseline else "X_spec"
            x = d.get(raw_key)
            if x is None:
                x = d["X_spec"]   # fallback if bc not precomputed
            return {**d, "X_spec": x}

        # ── Dispatch ──────────────────────────────────────────────────────
        wn  = shared_data["wavenumbers"]

        if variant == "baseline":
            # ---- Original behavior ----
            if scenario_key == "Box12_LOSO_PATIENT_AVG":
                scenario_box12_loso_patient_avg(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"], shared_data["id_tr"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg, variant=variant)

            elif scenario_key == "Box12_LOSO_SPECTRUM_LEVEL":
                scenario_box12_loso_spectrum_level(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"], shared_data["id_tr"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg, variant=variant)

            elif scenario_key == "Train12_Predict3_PATIENT_AVG":
                scenario_train12_predict3_patient_avg(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"],
                    X3_te_use, shared_data["y_te"], shared_data["si_te"], shared_data["id_te"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=False, variant=variant)

            elif scenario_key == "Train12_Predict3_SPECTRUM_LEVEL":
                scenario_train12_predict3_spectrum_level(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"],
                    X3_te_use, shared_data["y_te"], shared_data["si_te"], shared_data["id_te"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=False, variant=variant)

            elif scenario_key == "Box123_LOSO_PATIENT_AVG":
                scenario_box123_loso_patient_avg(
                    X3_all_use, shared_data["y_all"], shared_data["si_all"], shared_data["id_all"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg, variant=variant)

            elif scenario_key == "Box123_LOSO_SPECTRUM_LEVEL":
                scenario_box123_loso_spectrum_level(
                    X3_all_use, shared_data["y_all"], shared_data["si_all"], shared_data["id_all"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg, variant=variant)

            else:
                raise ValueError(f"Unknown scenario key: {scenario_key}")

        elif variant == "qc_only":
            # ---- QC-filtered data, no CORAL ----
            if scenario_key in ("Box12_LOSO_PATIENT_AVG", "Box12_LOSO_SPECTRUM_LEVEL"):
                qc_tr = _pick_qc("qc_tr")
                fn = (_loso_patient_avg_qc_core if scenario_key.endswith("PATIENT_AVG")
                      else _loso_spectrum_level_qc_core)
                fn(qc_tr["X_spec"], qc_tr["spec_sids"], qc_tr["y"], qc_tr["si"], qc_tr["ids"],
                   wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                   scenario_name="Scenario_Box12_LOSO", variant=variant)

            elif scenario_key in ("Box123_LOSO_PATIENT_AVG", "Box123_LOSO_SPECTRUM_LEVEL"):
                qc_all = _pick_qc("qc_all")
                fn = (_loso_patient_avg_qc_core if scenario_key.endswith("PATIENT_AVG")
                      else _loso_spectrum_level_qc_core)
                fn(qc_all["X_spec"], qc_all["spec_sids"], qc_all["y"], qc_all["si"], qc_all["ids"],
                   wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                   scenario_name="Scenario_Box123_LOSO", variant=variant)

            elif scenario_key == "Train12_Predict3_PATIENT_AVG":
                qc_tr = _pick_qc("qc_tr"); qc_te = _pick_qc("qc_te")
                scenario_train12_predict3_patient_avg_qc(
                    qc_tr, qc_te, wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=False, variant=variant)

            elif scenario_key == "Train12_Predict3_SPECTRUM_LEVEL":
                qc_tr = _pick_qc("qc_tr"); qc_te = _pick_qc("qc_te")
                scenario_train12_predict3_spectrum_level_qc(
                    qc_tr, qc_te, wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=False, variant=variant)

            else:
                raise ValueError(f"Unknown scenario key for qc_only: {scenario_key}")

        elif variant == "coral_only":
            # ---- Baseline data + CORAL on test spectra ----
            # Only valid for Train12_Predict3 (CORAL has no meaning for LOSO)
            if not HAS_CORAL:
                raise RuntimeError(
                    "preprocessing_new module not found. "
                    "CORAL variants require coral_align_target_to_source."
                )
            if scenario_key == "Train12_Predict3_PATIENT_AVG":
                scenario_train12_predict3_patient_avg(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"],
                    X3_te_use, shared_data["y_te"], shared_data["si_te"], shared_data["id_te"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=True, variant=variant)

            elif scenario_key == "Train12_Predict3_SPECTRUM_LEVEL":
                scenario_train12_predict3_spectrum_level(
                    X3_tr_use, shared_data["y_tr"], shared_data["si_tr"],
                    X3_te_use, shared_data["y_te"], shared_data["si_te"], shared_data["id_te"],
                    wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=True, variant=variant)

            else:
                raise ValueError(f"coral_only is only valid for Train12_Predict3, not {scenario_key}")

        elif variant == "qc_coral":
            # ---- QC-filtered data + CORAL on test spectra ----
            if not HAS_CORAL:
                raise RuntimeError(
                    "preprocessing_new module not found. "
                    "CORAL variants require coral_align_target_to_source."
                )
            if scenario_key == "Train12_Predict3_PATIENT_AVG":
                qc_tr = _pick_qc("qc_tr"); qc_te = _pick_qc("qc_te")
                scenario_train12_predict3_patient_avg_qc(
                    qc_tr, qc_te, wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=True, variant=variant)

            elif scenario_key == "Train12_Predict3_SPECTRUM_LEVEL":
                qc_tr = _pick_qc("qc_tr"); qc_te = _pick_qc("qc_te")
                scenario_train12_predict3_spectrum_level_qc(
                    qc_tr, qc_te, wn, methods, baseline_cfg, outdir, model_name, fast_cfg,
                    use_coral=True, variant=variant)

            else:
                raise ValueError(f"qc_coral is only valid for Train12_Predict3, not {scenario_key}")

        else:
            raise ValueError(f"Unknown variant: {variant}")

        elapsed = time.time() - t0
        return {"combo": tag, "status": "OK", "time_s": round(elapsed, 2), "error": None}

    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        return {"combo": tag, "status": "FAILED", "time_s": round(elapsed, 2), "error": tb}


# =============================================================================
# SUMMARY WRITER
# =============================================================================
def write_master_summary(out_root: Path, results: list[dict]):
    rows = []
    for root_dir, dirs, files in os.walk(out_root):
        if "metrics.csv" in files:
            try:
                df = pd.read_csv(Path(root_dir) / "metrics.csv")
                rows.append(df)
            except Exception:
                pass
    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary.to_csv(out_root / "master_summary.csv", index=False)
        logger.info(f"Master summary: {len(summary)} combos -> {out_root / 'master_summary.csv'}")

    if results:
        pd.DataFrame(results).to_csv(out_root / "run_results.csv", index=False)


def write_error_log(out_root: Path, results: list[dict]):
    failed = [r for r in results if r["status"] == "FAILED"]
    if failed:
        err_path = out_root / "error_log.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for r in failed:
                f.write(f"\n{'='*60}\n{r['combo']}\n{r['error']}\n")
        logger.warning(f"{len(failed)} combos failed. See {err_path}")


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None):
    args = parse_args(argv)
    out_root = Path(args.output)
    setup_logging(out_root)

    logger.info("=" * 60)
    logger.info("SERS Classification Sweep — Athena (QC + CORAL enabled)")
    logger.info("=" * 60)

    # ── Resolve selections ────────────────────────────────────────────────────
    preproc_names  = args.preproc   if args.preproc   else list(ALL_PREPROC.keys())
    model_names    = args.models    if args.models    else ALL_MODELS
    scenario_names = args.scenarios if args.scenarios else ALL_SCENARIOS
    variant_names  = args.variants  if args.variants  else ALL_VARIANTS

    # Validate
    for p in preproc_names:
        assert p in ALL_PREPROC,    f"Unknown preproc: {p}"
    for m in model_names:
        assert m in ALL_MODELS,     f"Unknown model: {m}"
    for s in scenario_names:
        assert s in ALL_SCENARIOS,  f"Unknown scenario: {s}"
    for v in variant_names:
        assert v in ALL_VARIANTS,   f"Unknown variant: {v}"

    # Check module availability for CORAL
    need_coral = any(v in variant_names for v in ("coral_only", "qc_coral"))
    if need_coral and not HAS_CORAL:
        logger.warning(
            "CORAL variants requested but preprocessing_new.py not importable. "
            "coral_only and qc_coral combos will FAIL at runtime."
        )

    # Check QC loader availability
    need_qc = any(v in variant_names for v in ("qc_only", "qc_coral"))
    if need_qc and not HAS_QC_LOADER:
        logger.warning(
            "QC variants requested but data_loader_qc.py not importable. "
            "qc_only and qc_coral combos will FAIL at runtime."
        )

    fast_cfg = {
        "skip_plots":    args.skip_plots,
        "skip_roc":      args.skip_roc,
        "skip_overlay":  args.skip_overlay,
        "skip_top_wn":   args.skip_top_wn,
        "max_spectra_plot": args.max_spectra_plot,
    }

    # ── Load baseline (non-QC) data ───────────────────────────────────────────
    logger.info(f"Loading train data (baseline): {args.train_data}")
    wn_tr, _, X_tr, meta_tr, *_ = load_data(
        args.train_data, args.train_meta, ["DM1", "Control"], strict=False)
    logger.info(f"Loading test data  (baseline): {args.test_data}")
    wn_te, _, X_te, meta_te, *_ = load_data(
        args.test_data, args.test_meta, ["DM1", "Control"], strict=False)

    wavenumbers = np.asarray(wn_tr).astype(float)
    y_tr = get_labels(meta_tr); y_te = get_labels(meta_te)
    id_tr = get_patient_ids(meta_tr); id_te = get_patient_ids(meta_te)
    si_tr = try_get_si(meta_tr); si_te = try_get_si(meta_te)

    X3_tr  = reshape_patient(X_tr, N_REP)
    X3_te  = reshape_patient(X_te, N_REP)
    X3_all = np.concatenate([X3_tr, X3_te], axis=0)
    y_all  = np.concatenate([y_tr, y_te])
    si_all = np.concatenate([si_tr, si_te]) if (si_tr is not None and si_te is not None) else None
    id_all = np.concatenate([id_tr, id_te])

    # ── Precompute baseline corrections (per-spectrum, fold-independent) ──────
    need_baseline = any(ALL_PREPROC[p]["baseline"].get("do_baseline", False)
                        for p in preproc_names)
    if need_baseline:
        logger.info("Precomputing ALS baseline corrections (baseline data)...")
        X3_tr_bc  = baseline_correct_X3(X3_tr,  BASELINE_CFG)
        X3_te_bc  = baseline_correct_X3(X3_te,  BASELINE_CFG)
        X3_all_bc = baseline_correct_X3(X3_all, BASELINE_CFG)
        logger.info("Baseline correction done.")
    else:
        X3_tr_bc = X3_te_bc = X3_all_bc = None

    shared_data = {
        "wavenumbers": wavenumbers,
        "X3_tr":  X3_tr,  "X3_te":  X3_te,  "X3_all":  X3_all,
        "X3_tr_bc": X3_tr_bc, "X3_te_bc": X3_te_bc, "X3_all_bc": X3_all_bc,
        "y_tr": y_tr, "y_te": y_te, "y_all": y_all,
        "si_tr": si_tr, "si_te": si_te, "si_all": si_all,
        "id_tr": id_tr, "id_te": id_te, "id_all": id_all,
        # QC data (filled below if needed)
        "qc_tr":  None,
        "qc_te":  None,
        "qc_all": None,
    }

    # ── Load QC-filtered data if any QC variant is requested ─────────────────
    if need_qc and args.qc_metrics:
        qc_metrics_path = Path(args.qc_metrics)
        if not qc_metrics_path.exists():
            raise FileNotFoundError(f"QC metrics file not found: {qc_metrics_path}")

        logger.info(f"Loading QC-filtered data from: {qc_metrics_path}")
        logger.info(f"  Box train label: '{args.qc_box_train}' | Box test label: '{args.qc_box_test}'")

        qc_tr = load_qc_cohort(
            data_dir=args.train_data, meta_path=args.train_meta,
            qc_metrics_path=qc_metrics_path, qc_box=args.qc_box_train,
        )
        qc_te = load_qc_cohort(
            data_dir=args.test_data, meta_path=args.test_meta,
            qc_metrics_path=qc_metrics_path, qc_box=args.qc_box_test,
        )
        logger.info(
            f"  QC train: {len(qc_tr['ids'])} patients, "
            f"{qc_tr['X_spec'].shape[1]} surviving spectra"
        )
        logger.info(
            f"  QC test:  {len(qc_te['ids'])} patients, "
            f"{qc_te['X_spec'].shape[1]} surviving spectra"
        )

        # Combined (Box123 LOSO with QC)
        qc_all_X = np.hstack([qc_tr["X_spec"], qc_te["X_spec"]])
        qc_all_sids = np.concatenate([qc_tr["spec_sids"], qc_te["spec_sids"]])
        qc_all_y    = np.concatenate([qc_tr["y"], qc_te["y"]])
        qc_all_ids  = np.concatenate([qc_tr["ids"], qc_te["ids"]])
        qc_all_si   = (np.concatenate([qc_tr["si"], qc_te["si"]])
                       if (qc_tr["si"] is not None and qc_te["si"] is not None) else None)
        qc_all = {
            "X_spec":    qc_all_X,
            "X_spec_bc": None,
            "spec_sids": qc_all_sids,
            "y":         qc_all_y,
            "si":        qc_all_si,
            "ids":       qc_all_ids,
            "meta":      None,
        }

        # Precompute baseline corrections for QC data
        if need_baseline:
            logger.info("Precomputing ALS baseline corrections (QC data)...")
            qc_tr["X_spec_bc"]  = baseline_correct_spectra(qc_tr["X_spec"],  BASELINE_CFG)
            qc_te["X_spec_bc"]  = baseline_correct_spectra(qc_te["X_spec"],  BASELINE_CFG)
            qc_all["X_spec_bc"] = baseline_correct_spectra(qc_all["X_spec"], BASELINE_CFG)
            logger.info("QC baseline correction done.")

        shared_data["qc_tr"]  = qc_tr
        shared_data["qc_te"]  = qc_te
        shared_data["qc_all"] = qc_all

    elif need_qc and not args.qc_metrics:
        logger.warning(
            "QC variants (qc_only / qc_coral) requested but --qc-metrics not provided. "
            "These combos will FAIL at runtime. "
            "Run qc_sers_spectra.py first and pass --qc-metrics <path/to/QC_metrics_per_spectrum.csv>."
        )

    # ── Generate and run combos ───────────────────────────────────────────────
    combos = list(generate_combos(preproc_names, model_names, scenario_names, variant_names))
    logger.info(
        f"Total combos: {len(combos)} | Workers: {args.max_workers} | Force: {args.force}"
    )
    logger.info(f"Scenarios: {scenario_names}")
    logger.info(f"Variants:  {variant_names}")

    results = []
    if args.max_workers <= 1:
        for i, combo in enumerate(combos, 1):
            tag = f"{combo[0]}|{combo[1]}|{combo[2]}|{combo[3]}"
            logger.info(f"[{i}/{len(combos)}] START  {tag}")
            r = run_one_combo(combo, shared_data, out_root, fast_cfg, args.force)
            logger.info(f"[{i}/{len(combos)}] {r['status']:7s} {tag}  ({r['time_s']:.1f}s)")
            results.append(r)
    else:
        from joblib import Parallel, delayed
        logger.info(f"Launching joblib with {args.max_workers} workers...")

        def _worker(combo):
            return run_one_combo(combo, shared_data, out_root, fast_cfg, args.force)

        results = Parallel(n_jobs=args.max_workers, backend="loky", verbose=10)(
            delayed(_worker)(c) for c in combos
        )
        for r in results:
            logger.info(f"  {r['status']:7s} {r['combo']}  ({r['time_s']:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")
    n_fail = sum(1 for r in results if r["status"] == "FAILED")
    logger.info(f"Done: {n_ok} OK, {n_skip} skipped, {n_fail} failed")

    write_master_summary(out_root, results)
    write_error_log(out_root, results)

    logger.info(f"All outputs under: {out_root}")


if __name__ == "__main__":
    main()
