# audit_helpers.py
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KNOWN_PREPROC = {
    "RAW", "EMSC", "NORM", "SNV", "SNV+2DER",
    "BASELINE+EMSC", "BASELINE+NORM", "BASELINE+SNV", "BASELINE+SNV+2DER",
}

KNOWN_VARIANTS = {"baseline", "qc_only", "coral_only", "qc_coral"}

LOSO_GROUPS = {"Box12_LOSO", "Box123_LOSO"}
T12P3_GROUP = "Train12_Predict3"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Path discovery
# ══════════════════════════════════════════════════════════════════════════════

def _scenario_group(scenario: str) -> str:
    s = scenario.upper()
    if "BOX123" in s:
        return "Box123_LOSO"
    if "BOX12" in s and "LOSO" in s:
        return "Box12_LOSO"
    if "TRAIN12" in s or "PREDICT3" in s:
        return "Train12_Predict3"
    return scenario


def parse_prediction_path(pred_csv: Path, results_root: Path) -> Optional[dict]:
    """
    Parse metadata from a predictions.csv path relative to results_root.

    Supports:
      Old (4-level): Scenario/MODE/PREPROC/MODEL/predictions.csv  -> variant=baseline
      New (5-level): Scenario/MODE/VARIANT/PREPROC/MODEL/predictions.csv
    Returns None if path cannot be parsed.
    """
    try:
        parts = pred_csv.relative_to(results_root).parts[:-1]  # strip filename
    except ValueError:
        log.warning("Path %s is not under results_root %s", pred_csv, results_root)
        return None

    if len(parts) == 4:
        scenario, mode, preproc, model = parts
        if preproc not in KNOWN_PREPROC:
            log.debug("Skipping unrecognized preproc %s in %s", preproc, pred_csv)
            return None
        variant = "baseline"
    elif len(parts) == 5:
        scenario, mode, variant, preproc, model = parts
        if preproc not in KNOWN_PREPROC:
            log.debug("Skipping unrecognized preproc %s in %s", preproc, pred_csv)
            return None
    else:
        log.debug("Skipping unrecognized depth %d: %s", len(parts), pred_csv)
        return None

    return {
        "scenario": scenario,
        "mode": mode,
        "variant": variant,
        "preproc": preproc,
        "model": model,
        "scenario_group": _scenario_group(scenario),
        "leaf_dir": pred_csv.parent,
    }


def discover_runs(results_root: Path) -> list[dict]:
    """Recursively find all predictions.csv files and parse their metadata."""
    results_root = Path(results_root)
    runs = []
    skipped = 0
    for pred_csv in sorted(results_root.rglob("predictions.csv")):
        meta = parse_prediction_path(pred_csv, results_root)
        if meta is None:
            skipped += 1
            continue
        runs.append(meta)
    log.info("Discovered %d runs (%d skipped)", len(runs), skipped)
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Load master DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def _load_run_predictions(meta: dict) -> Optional[pd.DataFrame]:
    """Load predictions.csv for one run and prepend metadata columns."""
    pred_path = meta["leaf_dir"] / "predictions.csv"
    if not pred_path.exists():
        log.warning("Missing predictions.csv: %s", pred_path)
        return None
    try:
        df = pd.read_csv(pred_path)
    except Exception as exc:
        log.warning("Failed to read %s: %s", pred_path, exc)
        return None
    required = {"patient_id", "y_true", "y_score", "y_pred"}
    if not required.issubset(df.columns):
        log.warning("Missing columns in %s: %s", pred_path, required - set(df.columns))
        return None
    for col in ("scenario", "mode", "variant", "preproc", "model", "scenario_group"):
        df.insert(0, col, meta[col])
    return df


def load_all_predictions(results_root: Path) -> pd.DataFrame:
    """
    Discover all runs and return a master DataFrame with one row per patient × run.
    """
    runs = discover_runs(Path(results_root))
    frames = []
    for meta in runs:
        df = _load_run_predictions(meta)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No predictions loaded from {results_root}")
    master = pd.concat(frames, ignore_index=True)
    log.info(
        "Master DataFrame: %d rows, %d unique patients, %d scenario groups",
        len(master),
        master["patient_id"].nunique(),
        master["scenario_group"].nunique(),
    )
    return master


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Patient-level statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_loso_patient_stats(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(patient_id, scenario_group) statistics for LOSO scenarios only.
    Includes both Box12_LOSO and Box123_LOSO rows.
    """
    loso = master_df[master_df["scenario_group"].isin(LOSO_GROUPS)].copy()
    if loso.empty:
        log.warning("No LOSO rows found in master_df")
        return pd.DataFrame()

    loso["wrong"] = (loso["y_pred"] != loso["y_true"]).astype(int)
    loso["near_thresh"] = loso["y_score"].between(0.35, 0.65).astype(int)
    loso["conf_wrong"] = (
        (loso["y_pred"] != loso["y_true"]) &
        ((loso["y_score"] - 0.5).abs() > 0.3)
    ).astype(int)

    agg = loso.groupby(["patient_id", "scenario_group"]).agg(
        y_true=("y_true", "first"),
        n_runs=("y_score", "count"),
        n_misclassified=("wrong", "sum"),
        mean_score=("y_score", "mean"),
        std_score=("y_score", "std"),
        min_score=("y_score", "min"),
        max_score=("y_score", "max"),
        near_threshold_frac=("near_thresh", "mean"),
        confidently_wrong_frac=("conf_wrong", "mean"),
    ).reset_index()

    agg["misclassification_rate"] = agg["n_misclassified"] / agg["n_runs"]
    agg["std_score"] = agg["std_score"].fillna(0.0)
    return agg


def compute_t12p3_patient_stats(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-patient statistics for Train12_Predict3 (DM1-only external detection).
    Hard cases = DM1 samples with low predicted DM1 probability.
    """
    t12 = master_df[master_df["scenario_group"] == T12P3_GROUP].copy()
    if t12.empty:
        log.warning("No Train12_Predict3 rows found in master_df")
        return pd.DataFrame()

    t12["near_thresh"] = t12["y_score"].between(0.35, 0.65).astype(int)
    t12["low_score"] = (t12["y_score"] < 0.5).astype(int)
    t12["very_low_score"] = (t12["y_score"] < 0.3).astype(int)

    agg = t12.groupby("patient_id").agg(
        y_true=("y_true", "first"),
        n_runs=("y_score", "count"),
        mean_dm1_score=("y_score", "mean"),
        std_score=("y_score", "std"),
        min_score=("y_score", "min"),
        max_score=("y_score", "max"),
        near_threshold_frac=("near_thresh", "mean"),
        low_score_frac=("low_score", "mean"),
        very_low_score_frac=("very_low_score", "mean"),
    ).reset_index()

    agg["std_score"] = agg["std_score"].fillna(0.0)
    return agg


def compute_variant_breakdown(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(patient_id, scenario_group, variant) statistics.
    Used for VARIANT_SPECIFIC flag and QC/CORAL directionality.
    """
    rows = []
    for (pid, sg, var), grp in master_df.groupby(
        ["patient_id", "scenario_group", "variant"]
    ):
        is_t12 = sg == T12P3_GROUP
        row = {
            "patient_id": pid,
            "scenario_group": sg,
            "variant": var,
            "y_true": int(grp["y_true"].iloc[0]),
            "n_runs": len(grp),
            "mean_score": grp["y_score"].mean(),
        }
        if not is_t12:
            row["misclassification_rate"] = float(
                (grp["y_pred"] != grp["y_true"]).mean()
            )
        else:
            row["misclassification_rate"] = float("nan")
        row["low_score_frac"] = float((grp["y_score"] < 0.5).mean())
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Outlier flags
# ══════════════════════════════════════════════════════════════════════════════

def apply_loso_flags(
    loso_stats: pd.DataFrame,
    variant_breakdown: pd.DataFrame,
    misclass_threshold: float = 0.4,
) -> pd.DataFrame:
    """Add LOSO outlier flag columns to loso_stats."""
    df = loso_stats.copy()

    df["PERSISTENT_MISCLASS"] = df["misclassification_rate"] >= misclass_threshold

    df["BORDERLINE_UNSTABLE"] = (
        df["mean_score"].between(0.35, 0.65) & (df["std_score"] > 0.15)
    )

    df["CONFIDENTLY_WRONG"] = df["confidently_wrong_frac"] >= 0.25

    wrong_side = (
        ((df["y_true"] == 0) & (df["mean_score"] > 0.5)) |
        ((df["y_true"] == 1) & (df["mean_score"] < 0.5))
    )
    df["LABEL_SUSPICION"] = (df["misclassification_rate"] >= 0.7) & wrong_side

    # SCENARIO_SPECIFIC: compare Box12 vs Box123 rates for same patient
    rates_pivot = df.pivot_table(
        index="patient_id",
        columns="scenario_group",
        values="misclassification_rate",
        aggfunc="first",
    )
    if "Box12_LOSO" in rates_pivot.columns and "Box123_LOSO" in rates_pivot.columns:
        scenario_diff = (
            (rates_pivot["Box12_LOSO"] - rates_pivot["Box123_LOSO"])
            .abs()
            .rename("_scenario_diff")
        )
        df = df.join(scenario_diff, on="patient_id")
        df["SCENARIO_SPECIFIC"] = df["_scenario_diff"].fillna(0.0) > 0.3
        df = df.drop(columns=["_scenario_diff"])
    else:
        df["SCENARIO_SPECIFIC"] = False

    # VARIANT_SPECIFIC: max-min rate across variants per (patient, scenario_group)
    loso_vb = variant_breakdown[
        variant_breakdown["scenario_group"].isin(LOSO_GROUPS)
    ]
    if not loso_vb.empty and "misclassification_rate" in loso_vb.columns:
        var_range = (
            loso_vb.groupby(["patient_id", "scenario_group"])["misclassification_rate"]
            .agg(lambda x: x.dropna().max() - x.dropna().min() if len(x.dropna()) > 1 else 0.0)
            .rename("_var_range")
            .reset_index()
        )
        df = df.merge(var_range, on=["patient_id", "scenario_group"], how="left")
        df["VARIANT_SPECIFIC"] = df["_var_range"].fillna(0.0) > 0.4
        df = df.drop(columns=["_var_range"])
    else:
        df["VARIANT_SPECIFIC"] = False

    return df


def apply_t12p3_flags(
    t12p3_stats: pd.DataFrame,
    variant_breakdown: pd.DataFrame,
) -> pd.DataFrame:
    """Add Train12_Predict3-specific outlier flags (DM1 detectability framing)."""
    if t12p3_stats.empty:
        return t12p3_stats.copy()

    df = t12p3_stats.copy()

    df["PERSISTENT_LOW_SCORE_DM1"] = df["low_score_frac"] >= 0.4
    df["CONFIDENTLY_LOW_DM1_SCORE"] = df["very_low_score_frac"] >= 0.25
    df["BORDERLINE_UNSTABLE"] = (
        df["mean_dm1_score"].between(0.35, 0.65) & (df["std_score"] > 0.15)
    )

    t12_vb = variant_breakdown[variant_breakdown["scenario_group"] == T12P3_GROUP]
    if not t12_vb.empty:
        var_range = (
            t12_vb.groupby("patient_id")["low_score_frac"]
            .agg(lambda x: x.max() - x.min() if len(x) > 1 else 0.0)
            .rename("_var_range")
            .reset_index()
        )
        df = df.merge(var_range, on="patient_id", how="left")
        df["VARIANT_SPECIFIC"] = df["_var_range"].fillna(0.0) > 0.3
        df = df.drop(columns=["_var_range"])
    else:
        df["VARIANT_SPECIFIC"] = False

    return df


def apply_directionality_flags(
    loso_flagged: pd.DataFrame,
    t12p3_flagged: pd.DataFrame,
    variant_breakdown: pd.DataFrame,
    delta: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add IMPROVES/WORSENS_WITH_QC/CORAL/QC_CORAL columns.
    Flags are only True when both comparison variants are present for a patient.
    """
    comparisons = [
        ("QC", "baseline", "qc_only"),
        ("CORAL", "baseline", "coral_only"),
        ("QC_CORAL", "baseline", "qc_coral"),
    ]
    loso = loso_flagged.copy()
    t12p3 = t12p3_flagged.copy()

    for label, _, _ in comparisons:
        for df in (loso, t12p3):
            df[f"IMPROVES_WITH_{label}"] = False
            df[f"WORSENS_WITH_{label}"] = False

    for label, v_from, v_to in comparisons:
        # --- LOSO ---
        loso_vb = variant_breakdown[
            (variant_breakdown["scenario_group"].isin(LOSO_GROUPS)) &
            (variant_breakdown["variant"].isin([v_from, v_to]))
        ]
        for (pid, sg), grp in loso_vb.groupby(["patient_id", "scenario_group"]):
            r_from = grp.loc[grp["variant"] == v_from, "misclassification_rate"]
            r_to = grp.loc[grp["variant"] == v_to, "misclassification_rate"]
            if r_from.empty or r_to.empty:
                continue
            if pd.isna(r_from.iloc[0]) or pd.isna(r_to.iloc[0]):
                continue
            d = float(r_to.iloc[0]) - float(r_from.iloc[0])
            mask = (loso["patient_id"] == pid) & (loso["scenario_group"] == sg)
            if d <= -delta:
                loso.loc[mask, f"IMPROVES_WITH_{label}"] = True
            elif d >= delta:
                loso.loc[mask, f"WORSENS_WITH_{label}"] = True

        # --- T12P3 ---
        if not t12p3.empty:
            t12_vb = variant_breakdown[
                (variant_breakdown["scenario_group"] == T12P3_GROUP) &
                (variant_breakdown["variant"].isin([v_from, v_to]))
            ]
            for pid, grp in t12_vb.groupby("patient_id"):
                r_from = grp.loc[grp["variant"] == v_from, "low_score_frac"]
                r_to = grp.loc[grp["variant"] == v_to, "low_score_frac"]
                if r_from.empty or r_to.empty:
                    continue
                d = float(r_to.iloc[0]) - float(r_from.iloc[0])
                mask = t12p3["patient_id"] == pid
                if d <= -delta:
                    t12p3.loc[mask, f"IMPROVES_WITH_{label}"] = True
                elif d >= delta:
                    t12p3.loc[mask, f"WORSENS_WITH_{label}"] = True

    return loso, t12p3


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — Suspicion scores + ranked table
# ══════════════════════════════════════════════════════════════════════════════

def compute_loso_suspicion_score(loso_flagged: pd.DataFrame) -> pd.DataFrame:
    """Add suspicion_score column to LOSO flagged table."""
    df = loso_flagged.copy()
    df["suspicion_score"] = (
        df["misclassification_rate"] * 2.0
        + df["PERSISTENT_MISCLASS"].astype(float)
        + df["BORDERLINE_UNSTABLE"].astype(float)
        + df["CONFIDENTLY_WRONG"].astype(float)
        + df["LABEL_SUSPICION"].astype(float) * 2.0
        + df["SCENARIO_SPECIFIC"].astype(float) * 0.5
        + df["VARIANT_SPECIFIC"].astype(float) * 0.5
    )
    return df


def compute_t12p3_suspicion_score(t12p3_flagged: pd.DataFrame) -> pd.DataFrame:
    """Add suspicion_score column to Train12_Predict3 flagged table."""
    if t12p3_flagged.empty:
        return t12p3_flagged.copy()
    df = t12p3_flagged.copy()
    df["suspicion_score"] = (
        df["low_score_frac"] * 2.0
        + df["PERSISTENT_LOW_SCORE_DM1"].astype(float)
        + df["CONFIDENTLY_LOW_DM1_SCORE"].astype(float)
        + df["BORDERLINE_UNSTABLE"].astype(float)
    )
    return df


def build_ranked_table(
    loso_scored: pd.DataFrame,
    t12p3_scored: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine LOSO and T12P3 scored tables into one ranked suspicious-patient table.
    Takes the max suspicion_score per patient across scenario groups.
    """
    parts = []

    if not loso_scored.empty:
        loso_max = (
            loso_scored.groupby("patient_id")
            .agg(
                y_true=("y_true", "first"),
                suspicion_score=("suspicion_score", "max"),
                scenario_group=(
                    "scenario_group",
                    lambda x: "|".join(sorted(x.unique())),
                ),
            )
            .reset_index()
        )
        loso_max["source"] = "LOSO"
        parts.append(loso_max)

    if not t12p3_scored.empty:
        t12_part = t12p3_scored[["patient_id", "y_true", "suspicion_score"]].copy()
        t12_part["scenario_group"] = T12P3_GROUP
        t12_part["source"] = "Train12_Predict3"
        parts.append(t12_part)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    ranked = combined.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["class_label"] = ranked["y_true"].map({0: "Control", 1: "DM1"})
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
# PART 6 — Global visualizations
# ══════════════════════════════════════════════════════════════════════════════

def plot_global_heatmap(master_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Patients × run-combination heatmap colored by predicted DM1 score.
    Restricted to PATIENT_AVG mode to avoid double-counting.
    """
    df = master_df[master_df["mode"] == "PATIENT_AVG"].copy()
    if df.empty:
        log.warning("No PATIENT_AVG rows for heatmap; trying all modes")
        df = master_df.copy()
    if df.empty:
        log.warning("No data for heatmap; skipping")
        return

    df["run_key"] = (
        df["scenario_group"] + "|" + df["variant"] + "|"
        + df["preproc"] + "|" + df["model"]
    )
    pivot = df.pivot_table(
        index="patient_id", columns="run_key", values="y_score", aggfunc="mean"
    )

    ytrue_map = (
        master_df[["patient_id", "y_true"]]
        .drop_duplicates()
        .set_index("patient_id")["y_true"]
    )
    pivot = pivot.loc[pivot.index.intersection(ytrue_map.index)]
    pivot["_ytrue"] = pivot.index.map(ytrue_map)
    pivot["_mean"] = pivot.drop(columns=["_ytrue"]).mean(axis=1)
    pivot = pivot.sort_values(["_ytrue", "_mean"]).drop(columns=["_ytrue", "_mean"])

    n_patients, n_runs = pivot.shape
    fig_w = max(12, n_runs * 0.3)
    fig_h = max(8, n_patients * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(pivot.index, fontsize=6)
    ax.set_xticks(range(n_runs))
    ax.set_xticklabels(pivot.columns, fontsize=5, rotation=90)
    ax.set_xlabel("Run combination (scenario|variant|preproc|model)")
    ax.set_ylabel("Patient ID")
    ax.set_title(
        "Predicted DM1 score across all runs\n(green=high=DM1, red=low=control)"
    )
    plt.colorbar(im, ax=ax, label="Predicted DM1 probability")

    ytrue_sorted = pivot.index.map(ytrue_map)
    n_controls = int((ytrue_sorted == 0).sum())
    if 0 < n_controls < n_patients:
        ax.axhline(n_controls - 0.5, color="black", linewidth=1.5)

    plt.tight_layout()
    out = Path(output_dir) / "global_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap: %s", out)


def plot_misclassification_barplot(
    loso_scored: pd.DataFrame,
    t12p3_scored: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Sorted barplot of misclassification rate (LOSO) and low_score_frac (T12P3)."""
    has_loso = not loso_scored.empty
    has_t12 = not t12p3_scored.empty
    if not has_loso and not has_t12:
        log.warning("No data for barplot; skipping")
        return

    n_panels = sum([has_loso, has_t12])
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    if has_loso:
        ax = axes[panel]
        loso_avg = (
            loso_scored.groupby(["patient_id", "y_true"])["misclassification_rate"]
            .mean()
            .reset_index()
            .sort_values("misclassification_rate", ascending=False)
        )
        colors = loso_avg["y_true"].map({0: "#4472C4", 1: "#FF0000"})
        ax.bar(range(len(loso_avg)), loso_avg["misclassification_rate"], color=colors)
        ax.axhline(0.4, color="black", linestyle="--", linewidth=1, label="Threshold 0.4")
        ax.set_xticks(range(len(loso_avg)))
        ax.set_xticklabels(loso_avg["patient_id"], rotation=90, fontsize=7)
        ax.set_ylabel("Misclassification rate")
        ax.set_title("LOSO misclassification rate per patient\n(blue=Control, red=DM1)")
        ax.legend()
        panel += 1

    if has_t12:
        ax = axes[panel]
        t12_sorted = t12p3_scored.sort_values("low_score_frac", ascending=False)
        ax.bar(range(len(t12_sorted)), t12_sorted["low_score_frac"], color="#FF8C00")
        ax.axhline(0.4, color="black", linestyle="--", linewidth=1, label="Threshold 0.4")
        ax.set_xticks(range(len(t12_sorted)))
        ax.set_xticklabels(t12_sorted["patient_id"], rotation=90, fontsize=7)
        ax.set_ylabel("Low-score fraction (score < 0.5)")
        ax.set_title("Train12→Predict3: DM1 low-score fraction")
        ax.legend()

    plt.tight_layout()
    out = Path(output_dir) / "misclassification_barplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved barplot: %s", out)


def plot_scenario_comparison(
    master_df: pd.DataFrame,
    flagged_patients: list[str],
    output_dir: Path,
) -> None:
    """Bar chart per flagged patient: average score in each scenario group."""
    if not flagged_patients:
        log.warning("No flagged patients for scenario comparison; skipping")
        return

    df = master_df[
        master_df["patient_id"].isin(flagged_patients) &
        (master_df["mode"] == "PATIENT_AVG")
    ]
    if df.empty:
        # fall back to all modes
        df = master_df[master_df["patient_id"].isin(flagged_patients)]
    if df.empty:
        log.warning("No data for scenario comparison; skipping")
        return

    scenario_avg = (
        df.groupby(["patient_id", "scenario_group"])["y_score"].mean().reset_index()
    )
    n_patients = len(flagged_patients)
    fig_w = max(6, 4 * n_patients)
    fig, axes = plt.subplots(1, n_patients, figsize=(fig_w, 4), sharey=True)
    if n_patients == 1:
        axes = [axes]

    sg_order = ["Box12_LOSO", "Box123_LOSO", "Train12_Predict3"]
    colors = {
        "Box12_LOSO": "#4472C4",
        "Box123_LOSO": "#70AD47",
        "Train12_Predict3": "#FF8C00",
    }

    for ax, pid in zip(axes, flagged_patients):
        sub = scenario_avg[scenario_avg["patient_id"] == pid]
        scores = [
            float(sub.loc[sub["scenario_group"] == sg, "y_score"].values[0])
            if sg in sub["scenario_group"].values else float("nan")
            for sg in sg_order
        ]
        bar_colors = [colors[sg] for sg in sg_order]
        x = range(len(sg_order))
        bars = ax.bar(x, scores, color=bar_colors)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(pid, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [sg.replace("_", "\n") for sg in sg_order],
            rotation=0, ha="center", fontsize=7,
        )
        ax.set_ylabel("Mean predicted DM1 score")

    plt.suptitle(
        "Average predicted DM1 score per scenario — flagged patients", fontsize=10
    )
    plt.tight_layout()
    out = Path(output_dir) / "scenario_comparison_flagged.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved scenario comparison: %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# PART 7 — Spectral data loading
# ══════════════════════════════════════════════════════════════════════════════

def _get_data_loader():
    """Import private helpers from data_loader.py (same directory)."""
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from data_loader import _index_files, _load_txt_block, _extract_wavenumbers_txt
        return _index_files, _load_txt_block, _extract_wavenumbers_txt
    except ImportError as exc:
        log.warning("data_loader.py not importable (%s) — spectral plots unavailable", exc)
        return None, None, None


def find_patient_spectra_dir(patient_id: str, data_root: Path) -> Optional[Path]:
    """Return the spectra directory containing files for patient_id, or None."""
    _index_files, _, _ = _get_data_loader()
    if _index_files is None:
        return None
    for subdir in ["Box12_spectra", "Box3_spectra"]:
        d = Path(data_root) / subdir
        if not d.is_dir():
            continue
        try:
            groups = _index_files(d)
        except Exception:
            continue
        if patient_id in groups:
            return d
    return None


def load_patient_spectra(
    patient_id: str, data_root: Path
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load all raw spectral files for patient_id.
    Returns (wavenumbers, all_spectra, avg_spectrum) or None.
      wavenumbers: (1731,)
      all_spectra:  (1731, N_reps)
      avg_spectrum: (1731,)
    """
    _index_files, _load_txt_block, _extract_wavenumbers_txt = _get_data_loader()
    if _index_files is None:
        return None

    spectra_dir = find_patient_spectra_dir(patient_id, data_root)
    if spectra_dir is None:
        log.warning("No spectral files found for %s in %s", patient_id, data_root)
        return None

    try:
        groups = _index_files(spectra_dir)
        filenames = groups[patient_id]
        blocks = []
        wavenumbers = None
        for fname in filenames:
            fpath = spectra_dir / fname
            if wavenumbers is None:
                wavenumbers = _extract_wavenumbers_txt(fpath)
            block = _load_txt_block(fpath)  # (1731, 3)
            blocks.append(block)
        all_spectra = np.concatenate(blocks, axis=1)  # (1731, N_reps)
        avg_spectrum = all_spectra.mean(axis=1)
        return wavenumbers, all_spectra, avg_spectrum
    except Exception as exc:
        log.warning("Failed loading spectra for %s: %s", patient_id, exc)
        return None


def load_class_avg_spectra(
    y_true_label: int,
    master_df: pd.DataFrame,
    data_root: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load average spectrum for every patient of a given class.
    Returns {patient_id: (wavenumbers, avg_spectrum)}.
    """
    patients = master_df[master_df["y_true"] == y_true_label]["patient_id"].unique().tolist()
    result = {}
    for pid in patients:
        out = load_patient_spectra(pid, data_root)
        if out is None:
            continue
        wn, _, avg = out
        result[pid] = (wn, avg)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 8 — Per-patient spectral plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_raw_overlay(
    patient_id: str,
    y_true: int,
    all_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    same_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    opp_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plot A: all individual spectra of flagged patient over same-class background."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Same-class background (light grey)
    for pid, (wn, avg) in same_class_avgs.items():
        if pid == patient_id:
            continue
        ax.plot(wn, avg, color="lightgrey", linewidth=0.5, alpha=0.5)

    # Same-class mean
    same_others = [avg for pid, (_, avg) in same_class_avgs.items() if pid != patient_id]
    if same_others:
        sc_mean = np.array(same_others).mean(axis=0)
        sc_wn = next(iter(same_class_avgs.values()))[0]
        sc_label = "Control mean" if y_true == 0 else "DM1 mean"
        ax.plot(sc_wn, sc_mean, color="grey", linewidth=1.2, label=sc_label, alpha=0.8)

    # Opposite-class mean (dashed)
    if opp_class_avgs:
        opp_stack = np.array([avg for _, avg in opp_class_avgs.values()])
        opp_mean = opp_stack.mean(axis=0)
        opp_wn = next(iter(opp_class_avgs.values()))[0]
        opp_label = "Control mean" if y_true == 1 else "DM1 mean"
        ax.plot(opp_wn, opp_mean, color="navy", linewidth=1.2, linestyle="--",
                label=opp_label, alpha=0.8)

    # Flagged patient — ALL individual spectra
    n_reps = all_spectra.shape[1]
    pat_color = "#FF4500" if y_true == 0 else "#8B0000"
    for i in range(n_reps):
        label = f"{patient_id} (rep {i})" if i == 0 else "_nolegend_"
        ax.plot(wavenumbers, all_spectra[:, i], color=pat_color, linewidth=0.8,
                alpha=0.7, label=label)

    true_class = "Control" if y_true == 0 else "DM1"
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(
        f"{patient_id} (True class: {true_class}) — Raw spectra overlay\n"
        f"All {n_reps} individual spectra shown"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(output_dir) / "raw_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved raw_overlay: %s", out)


def plot_class_mean_comparison(
    patient_id: str,
    y_true: int,
    patient_avg: np.ndarray,
    wavenumbers: np.ndarray,
    same_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    opp_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plot B: patient mean vs true-class mean vs opposite-class mean."""
    fig, ax = plt.subplots(figsize=(12, 5))

    true_label = "Control" if y_true == 0 else "DM1"
    opp_label = "DM1" if y_true == 0 else "Control"
    pat_color = "#FF4500" if y_true == 0 else "#8B0000"

    if same_class_avgs:
        sc_others = [avg for pid, (_, avg) in same_class_avgs.items() if pid != patient_id]
        if sc_others:
            sc_mean = np.array(sc_others).mean(axis=0)
            ax.plot(wavenumbers, sc_mean, color="steelblue", linewidth=1.5,
                    label=f"{true_label} class mean", alpha=0.9)

    if opp_class_avgs:
        oc_mean = np.array([avg for _, avg in opp_class_avgs.values()]).mean(axis=0)
        ax.plot(wavenumbers, oc_mean, color="navy", linewidth=1.5, linestyle="--",
                label=f"{opp_label} class mean", alpha=0.9)

    ax.plot(wavenumbers, patient_avg, color=pat_color, linewidth=2,
            label=f"{patient_id} mean")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{patient_id} — Class mean comparison")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / "class_mean_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved class_mean_comparison: %s", out)


def plot_difference_from_class_mean(
    patient_id: str,
    y_true: int,
    patient_avg: np.ndarray,
    wavenumbers: np.ndarray,
    same_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    opp_class_avgs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    """Plot C: patient mean minus true-class mean and minus opposite-class mean."""
    fig, ax = plt.subplots(figsize=(12, 5))

    true_label = "Control" if y_true == 0 else "DM1"
    opp_label = "DM1" if y_true == 0 else "Control"

    if same_class_avgs:
        sc_others = [avg for pid, (_, avg) in same_class_avgs.items() if pid != patient_id]
        if sc_others:
            sc_mean = np.array(sc_others).mean(axis=0)
            diff_true = patient_avg - sc_mean
            ax.plot(wavenumbers, diff_true, color="steelblue", linewidth=1.5,
                    label=f"{patient_id} − {true_label} mean")

    if opp_class_avgs:
        oc_mean = np.array([avg for _, avg in opp_class_avgs.values()]).mean(axis=0)
        diff_opp = patient_avg - oc_mean
        ax.plot(wavenumbers, diff_opp, color="navy", linewidth=1.5, linestyle="--",
                label=f"{patient_id} − {opp_label} mean", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity difference")
    ax.set_title(f"{patient_id} — Difference from class mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / "difference_from_class_mean.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved difference_from_class_mean: %s", out)


def load_qc_data(qc_csv: Path) -> Optional[pd.DataFrame]:
    """Load QC_metrics_per_spectrum.csv. Returns None if missing or malformed."""
    qc_csv = Path(qc_csv)
    if not qc_csv.exists():
        log.warning("QC CSV not found: %s", qc_csv)
        return None
    try:
        df = pd.read_csv(qc_csv)
        required = {"sample", "filename", "keep"}
        if not required.issubset(df.columns):
            log.warning("QC CSV missing columns %s", required - set(df.columns))
            return None
        df["keep"] = df["keep"].astype(bool)
        return df
    except Exception as exc:
        log.warning("Failed to read QC CSV: %s", exc)
        return None


def plot_qc_overlay(
    patient_id: str,
    all_spectra: np.ndarray,
    wavenumbers: np.ndarray,
    qc_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot D: all raw spectra color-coded kept (green) vs QC-removed (red)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    n_reps = all_spectra.shape[1]

    if qc_df is not None:
        pat_qc = qc_df[qc_df["sample"] == patient_id].reset_index(drop=True)
        for i in range(n_reps):
            if i < len(pat_qc):
                kept = bool(pat_qc["keep"].iloc[i])
                color = "#2CA02C" if kept else "#D62728"
                label = f"rep {i} ({'kept' if kept else 'removed'})"
            else:
                color = "grey"
                label = f"rep {i} (no QC info)"
            ax.plot(wavenumbers, all_spectra[:, i], color=color, linewidth=0.8,
                    alpha=0.75, label=label if i < 5 else "_nolegend_")
    else:
        for i in range(n_reps):
            ax.plot(wavenumbers, all_spectra[:, i], color="grey",
                    linewidth=0.8, alpha=0.6)
        ax.text(0.02, 0.95, "QC data unavailable", transform=ax.transAxes,
                fontsize=10, color="red", va="top")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{patient_id} — All {n_reps} spectra (green=kept, red=QC-removed)")
    if n_reps <= 12:
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = Path(output_dir) / "qc_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved qc_overlay: %s", out)


def plot_score_distribution(
    patient_id: str,
    master_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot E: violin/strip of predicted scores per scenario group."""
    df = master_df[master_df["patient_id"] == patient_id]
    if df.empty:
        log.warning("No predictions found for %s", patient_id)
        return

    groups = sorted(df["scenario_group"].unique())
    n_g = len(groups)
    y_true = int(df["y_true"].iloc[0])
    true_label = "Control" if y_true == 0 else "DM1"

    fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 5), sharey=True)
    if n_g == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, sg in zip(axes, groups):
        scores = df[df["scenario_group"] == sg]["y_score"].dropna().values
        if len(scores) == 0:
            continue
        parts = ax.violinplot(scores, positions=[0], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#7FBA00")
            pc.set_alpha(0.6)
        jitter = rng.uniform(-0.05, 0.05, size=len(scores))
        ax.scatter(jitter, scores, color="black", s=15, alpha=0.7, zorder=3)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(-0.4, 0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_title(sg.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("Predicted DM1 score")
        ax.text(0.5, 0.02, f"n={len(scores)}", transform=ax.transAxes,
                ha="center", fontsize=8)

    plt.suptitle(
        f"{patient_id} (True: {true_label}) — Score distribution across runs",
        fontsize=10,
    )
    plt.tight_layout()
    out = Path(output_dir) / "score_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved score_distribution: %s", out)


# ══════════════════════════════════════════════════════════════════════════════
# PART 9 — Provenance report
# ══════════════════════════════════════════════════════════════════════════════

def build_provenance_report(
    flagged_patients_df: pd.DataFrame,
    data_root: Path,
    qc_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    For each flagged patient enumerate assigned spectral filenames and QC counts.
    Helps identify mapping errors and label issues.
    """
    _index_files, _, _ = _get_data_loader()
    have_loader = _index_files is not None

    rows = []
    for _, row in flagged_patients_df.iterrows():
        pid = str(row["patient_id"])
        y_true = int(row.get("y_true", -1))
        true_class = {0: "Control", 1: "DM1"}.get(y_true, "Unknown")

        filenames = []
        spectra_dir = None
        if have_loader:
            spectra_dir = find_patient_spectra_dir(pid, data_root)
            if spectra_dir is not None:
                try:
                    groups = _index_files(spectra_dir)
                    filenames = groups.get(pid, [])
                except Exception as exc:
                    log.warning("Could not index files for %s: %s", pid, exc)

        n_files = len(filenames)
        all_3_present = (n_files == 3)  # expect 3 Map files

        qc_keep = qc_remove = float("nan")
        if qc_df is not None:
            pat_qc = qc_df[qc_df["sample"] == pid]
            if not pat_qc.empty:
                qc_keep = int(pat_qc["keep"].sum())
                qc_remove = int((~pat_qc["keep"]).sum())

        rows.append({
            "patient_id": pid,
            "y_true": y_true,
            "true_class": true_class,
            "spectra_dir": str(spectra_dir) if spectra_dir else "",
            "n_files_found": n_files,
            "all_3_files_present": all_3_present,
            "filenames": "|".join(filenames),
            "qc_keep_count": qc_keep,
            "qc_remove_count": qc_remove,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PART 10 — Output writers
# ══════════════════════════════════════════════════════════════════════════════

def write_tables(
    loso_scored: pd.DataFrame,
    t12p3_scored: pd.DataFrame,
    ranked_table: pd.DataFrame,
    provenance_df: pd.DataFrame,
    output_root: Path,
) -> None:
    tables_dir = Path(output_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    def _save(df, name):
        p = tables_dir / name
        df.to_csv(p, index=False)
        log.info("Wrote %s (%d rows)", p, len(df))

    if not loso_scored.empty:
        _save(loso_scored, "all_loso_patient_summary.csv")
        box12 = loso_scored[loso_scored["scenario_group"] == "Box12_LOSO"]
        if not box12.empty:
            _save(box12, "box12_loso_patient_summary.csv")
        box123 = loso_scored[loso_scored["scenario_group"] == "Box123_LOSO"]
        if not box123.empty:
            _save(box123, "box123_loso_patient_summary.csv")

    if not t12p3_scored.empty:
        _save(t12p3_scored, "train12_predict3_patient_summary.csv")

    all_rows = []
    if not loso_scored.empty:
        all_rows.append(loso_scored.assign(source="LOSO"))
    if not t12p3_scored.empty:
        all_rows.append(t12p3_scored.assign(source="Train12_Predict3"))
    if all_rows:
        _save(pd.concat(all_rows, ignore_index=True), "all_patient_prediction_summary.csv")

    if not ranked_table.empty:
        _save(ranked_table, "ranked_suspicious_patients.csv")
    if not provenance_df.empty:
        _save(provenance_df, "provenance_flagged_patients.csv")


def write_excel_workbook(
    loso_scored: pd.DataFrame,
    t12p3_scored: pd.DataFrame,
    ranked_table: pd.DataFrame,
    output_root: Path,
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        log.warning("openpyxl not installed — skipping Excel workbook")
        return

    out = Path(output_root) / "outlier_audit_workbook.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        def _ws(df, sheet):
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)

        _ws(ranked_table, "Ranked_Suspicious")
        if not loso_scored.empty:
            _ws(loso_scored, "All_LOSO")
            _ws(loso_scored[loso_scored["scenario_group"] == "Box12_LOSO"], "Box12_LOSO")
            _ws(loso_scored[loso_scored["scenario_group"] == "Box123_LOSO"], "Box123_LOSO")
            _ws(
                loso_scored[loso_scored["y_true"] == 0].sort_values(
                    "suspicion_score", ascending=False
                ),
                "Suspicious_Controls",
            )
            _ws(
                loso_scored[loso_scored["y_true"] == 1].sort_values(
                    "suspicion_score", ascending=False
                ),
                "Suspicious_DM1",
            )
            qc_cols = [c for c in loso_scored.columns if "QC" in c or "CORAL" in c]
            if qc_cols:
                _ws(
                    loso_scored[["patient_id", "scenario_group", "y_true"] + qc_cols],
                    "QC_CORAL_Effects",
                )
        if not t12p3_scored.empty:
            _ws(t12p3_scored, "Train12_Predict3")

    log.info("Wrote Excel workbook: %s", out)


def write_summary_text(
    loso_scored: pd.DataFrame,
    t12p3_scored: pd.DataFrame,
    ranked_table: pd.DataFrame,
    output_root: Path,
) -> None:
    lines = ["=" * 70, "SERS CLASSIFICATION OUTLIER AUDIT SUMMARY", "=" * 70, ""]

    if not ranked_table.empty:
        lines += ["TOP 5 MOST SUSPICIOUS PATIENTS (combined score)", "-" * 50]
        for _, r in ranked_table.head(5).iterrows():
            lines.append(
                f"  {int(r['rank']):2d}. {r['patient_id']:12s}  "
                f"class={r['class_label']:8s}  "
                f"score={r['suspicion_score']:.2f}  "
                f"source={r['source']}"
            )
        lines.append("")

    if not loso_scored.empty:
        lines += ["LOSO SCENARIOS (Box12_LOSO + Box123_LOSO)", "-" * 50]
        suspicious_controls = loso_scored[
            (loso_scored["y_true"] == 0) & loso_scored["PERSISTENT_MISCLASS"]
        ]
        suspicious_dm1 = loso_scored[
            (loso_scored["y_true"] == 1) & loso_scored["PERSISTENT_MISCLASS"]
        ]
        lines.append(
            f"  Suspicious controls (persistent misclass): {len(suspicious_controls)}"
        )
        for _, r in suspicious_controls.sort_values(
            "misclassification_rate", ascending=False
        ).head(5).iterrows():
            lines.append(
                f"    {r['patient_id']:12s}  "
                f"misclass_rate={r['misclassification_rate']:.2f}  "
                f"scenario={r['scenario_group']}"
            )
        lines.append(
            f"  Suspicious DM1 (persistent misclass): {len(suspicious_dm1)}"
        )
        for _, r in suspicious_dm1.sort_values(
            "misclassification_rate", ascending=False
        ).head(5).iterrows():
            lines.append(
                f"    {r['patient_id']:12s}  "
                f"misclass_rate={r['misclassification_rate']:.2f}  "
                f"scenario={r['scenario_group']}"
            )

        label_suspects = loso_scored[loso_scored["LABEL_SUSPICION"]]
        if not label_suspects.empty:
            lines += ["", "  LABEL SUSPICION (misclass >= 70%, consistently opposite class):"]
            for _, r in label_suspects.iterrows():
                lines.append(
                    f"    {r['patient_id']:12s}  "
                    f"true={'Control' if r['y_true']==0 else 'DM1':8s}  "
                    f"misclass={r['misclassification_rate']:.2f}  "
                    f"mean_score={r['mean_score']:.3f}"
                )

        # QC/CORAL directionality summary
        for label in ("QC", "CORAL", "QC_CORAL"):
            w_col = f"WORSENS_WITH_{label}"
            i_col = f"IMPROVES_WITH_{label}"
            if w_col in loso_scored.columns:
                worsened = loso_scored[loso_scored[w_col]]
                if not worsened.empty:
                    lines += [f"  Patients WORSENED by {label}: {len(worsened)}"]
                    for pid in worsened["patient_id"].unique()[:5]:
                        lines.append(f"    {pid}")
            if i_col in loso_scored.columns:
                improved = loso_scored[loso_scored[i_col]]
                if not improved.empty:
                    lines += [f"  Patients IMPROVED by {label}: {len(improved)}"]
                    for pid in improved["patient_id"].unique()[:5]:
                        lines.append(f"    {pid}")
        lines.append("")

    if not t12p3_scored.empty:
        lines += ["TRAIN12→PREDICT3 (DM1-only external detection)", "-" * 50]
        hard_dm1 = t12p3_scored[t12p3_scored["PERSISTENT_LOW_SCORE_DM1"]]
        lines.append(
            f"  DM1 samples consistently hard to detect externally: {len(hard_dm1)}"
        )
        for _, r in hard_dm1.sort_values("low_score_frac", ascending=False).head(5).iterrows():
            lines.append(
                f"    {r['patient_id']:12s}  "
                f"low_score_frac={r['low_score_frac']:.2f}  "
                f"mean_dm1_score={r['mean_dm1_score']:.3f}"
            )
        lines.append("")

    lines += ["=" * 70]
    text = "\n".join(lines)
    out = Path(output_root) / "outlier_audit_summary.txt"
    out.write_text(text, encoding="utf-8")
    log.info("Wrote summary: %s", out)
    print(text)
