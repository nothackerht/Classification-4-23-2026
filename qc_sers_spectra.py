# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 12:58:50 2026

@author: notha
"""

# qc_sers_spectra.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data  # uses your deterministic loader (1731 points, etc.)
from data_loader_qc import normalize_fn_for_key  # shared key normalization


# ------------------------- helpers -------------------------
def extract_positions_from_txt(path: Path) -> list[int]:
    """
    Read position labels from first column of rows 1..3.
    Expected: -200, 0, 200 (or similar numeric).
    Returns list length 3 in file row order.
    """
    raw = np.genfromtxt(str(path), delimiter="\t", dtype=str)
    raw = np.array([row for row in raw if any(str(cell).strip() for cell in row)], dtype=object)

    if raw.shape[0] != 4:
        raise ValueError(f"{path.name}: Expected 4 rows, got {raw.shape[0]}")
    if raw.shape[1] != 1732:
        raise ValueError(f"{path.name}: Expected 1732 columns, got {raw.shape[1]}")

    pos = []
    for r in raw[1:, 0]:
        try:
            pos.append(int(float(str(r).strip())))
        except Exception as e:
            raise ValueError(f"{path.name}: could not parse position label '{r}' -> {e}") from e

    if len(pos) != 3:
        raise ValueError(f"{path.name}: Expected 3 positions, got {len(pos)}")
    return pos


def robust_z(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust z-score using median & MAD. Returns z ~ (x - med)/(1.4826*MAD)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(x)
    return (x - med) / scale


def parse_mapping_from_filename(fn: str) -> int | None:
    f = fn.lower()
    if "_map1" in f:
        return 1
    if "_map2" in f:
        return 2
    if "_map3" in f:
        return 3
    return None


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size != b.size:
        return np.nan
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(a, b) / denom)


def compute_metrics(w: np.ndarray, y: np.ndarray) -> dict:
    """
    y: (1731,) intensity
    """
    # (1) AUC
    auc = float(np.trapz(y, w))

    # (2) peak-to-baseline ratio
    peak_mask = (w >= 1000) & (w <= 1700)
    base_mask = (w >= 2500) & (w <= 3000)
    peak = float(np.nanmax(y[peak_mask])) if np.any(peak_mask) else np.nan
    base = float(np.nanmedian(y[base_mask])) if np.any(base_mask) else np.nan
    pbr = float(peak / (base + 1e-12)) if np.isfinite(peak) and np.isfinite(base) else np.nan

    # (3) smoothness / curvature: std(second derivative)
    d2 = np.diff(y, n=2)
    smooth = float(np.nanstd(d2))

    return {"auc": auc, "peak_base_ratio": pbr, "smooth_d2_std": smooth}


@dataclass
class Thresholds:
    z_fail: float = 3.5
    z_suspect: float = 2.5
    corr_fail_abs: float = 0.90     # hard fail if correlation to sample-median < this
    corr_suspect_abs: float = 0.95  # suspect if correlation < this


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



# ------------------------- main QC pipeline -------------------------

def qc_box(
    *,
    box_name: str,
    data_dir: Path,
    meta_path: Path,
    out_root: Path,
    include_types: tuple[str, ...] = ("DM1", "Control"),
) -> pd.DataFrame:
    """
    Returns per-spectrum dataframe with metrics + flags.
    """
    w, avg, all_spectra, filenames, meta = load_data(
        data_dir=str(data_dir),
        metadata_path=str(meta_path),
        include_types=include_types,
        return_filenames=True,
        strict=False,
    )

    # meta aligned to sample order (N rows); spectra are (1731, 9N)
    n_samples = meta.shape[0]
    assert all_spectra.shape[1] == 9 * n_samples

    # Build per-spectrum records
    rows = []
    for s_idx in range(n_samples):
        sample_prefix = meta.loc[s_idx, "FilePrefix"]
        typ = meta.loc[s_idx, "Type"]

        # 9 spectra columns for this sample
        cols = slice(s_idx * 9, (s_idx + 1) * 9)
        Y9 = all_spectra[:, cols]  # (1731, 9)
        fn9 = filenames[s_idx * 9 : (s_idx + 1) * 9]

        # sample median spectrum for correlation reference
        sample_med = np.nanmedian(Y9, axis=1)

        # NEW: cache true position labels (-200/0/200) for each mapping file used by this sample
        pos_by_fn = {f: extract_positions_from_txt(Path(data_dir) / f) for f in set(fn9)}

        seen = {}  # filename -> count used so far (0..2) for that file
        for j in range(9):
            y = Y9[:, j]
            fn = fn9[j]
            mapping = parse_mapping_from_filename(fn)

            k = seen.get(fn, 0)   # 0,1,2 in file order
            seen[fn] = k + 1

            position_nm = pos_by_fn[fn][k]   # -200/0/200
            rep_within_file = k + 1          # legacy 1..3

            m = compute_metrics(w, y)
            corr = safe_corr(y, sample_med)

            rows.append({
                "box": box_name,
                "sample": sample_prefix,
                "type": typ,
                "mapping": mapping,
                "filename": fn,

                "rep": rep_within_file,  # keep for now
                "position_nm": position_nm,
                "spectrum_key": f"{box_name}|{sample_prefix}|{normalize_fn_for_key(fn)}|pos{position_nm}",

                "corr_to_sample_median": corr,
                **m,
            })


    df = pd.DataFrame(rows)

    # Robust z-scores across *this box* (keeps it label-free & box-aware)
    df["auc_z"] = robust_z(df["auc"].to_numpy())
    df["pbr_z"] = robust_z(df["peak_base_ratio"].to_numpy())
    df["smooth_z"] = robust_z(df["smooth_d2_std"].to_numpy())
    df["corr_z"] = robust_z(df["corr_to_sample_median"].to_numpy())

    return df


def classify(df: pd.DataFrame, thr: Thresholds) -> pd.DataFrame:
    """
    Adds spectrum_qc = Good/Suspect/Failed + keep boolean.
    """
    df = df.copy()

    # Fail rules (label-free):
    # - AUC too low (strong negative z)
    # - Peak/base too low (strong negative z)
    # - Smoothness too high (strong positive z)
    # - Correlation too low (absolute + z-based)
    fail = (
        (df["auc_z"] < -thr.z_fail) |
        (df["pbr_z"] < -thr.z_fail) |
        (df["smooth_z"] > thr.z_fail) |
        (df["corr_to_sample_median"] < thr.corr_fail_abs) |
        (df["corr_z"] < -thr.z_fail)
    )

    suspect = (
        (~fail) & (
            (df["auc_z"] < -thr.z_suspect) |
            (df["pbr_z"] < -thr.z_suspect) |
            (df["smooth_z"] > thr.z_suspect) |
            (df["corr_to_sample_median"] < thr.corr_suspect_abs) |
            (df["corr_z"] < -thr.z_suspect)
        )
    )

    df["spectrum_qc"] = np.where(fail, "Failed", np.where(suspect, "Suspect", "Good"))

    # Final decision for downstream: discard only Failed by default (defensible, non-cherry-picky)
    df["keep"] = df["spectrum_qc"].isin(["Good", "Suspect"])

    # Mapping-level aggregation: mark mapping_failed if >=2 of 3 replicates failed
    grp = df.groupby(["box", "sample", "mapping"], dropna=False)["spectrum_qc"]
    mapping_failed = grp.apply(lambda s: (s == "Failed").sum() >= 2).rename("mapping_failed").reset_index()
    df = df.merge(mapping_failed, on=["box", "sample", "mapping"], how="left")

    # Sample-level flag: sample_failed if any mapping_failed OR >3 failed spectra total
    sample_failed = (
        df.groupby(["box", "sample"], dropna=False)
          .apply(lambda g: (g["mapping_failed"].any()) or ((g["spectrum_qc"] == "Failed").sum() > 3))
          .rename("sample_failed")
          .reset_index()
    )
    df = df.merge(sample_failed, on=["box", "sample"], how="left")

    return df


def plot_sample_qc(
    *,
    w: np.ndarray,
    Y9: np.ndarray,
    keep_mask: np.ndarray,
    out_path: Path,
    title: str,
    lw_keep: float = 1.6,
    lw_drop: float = 2.2,
) -> None:
    """
    Blue = kept, Red = discarded.
    """
    plt.figure(figsize=(14, 6))
    for j in range(Y9.shape[1]):
        if keep_mask[j]:
            plt.plot(w, Y9[:, j], linewidth=lw_keep, alpha=0.85, color="blue")
        else:
            plt.plot(w, Y9[:, j], linewidth=lw_drop, alpha=0.95, color="red")
    plt.title(title)
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box12_data", required=True, help="Path to Box1-2 data directory (.txt files)")
    ap.add_argument("--box12_meta", required=True, help="Path to Box1-2 metadata CSV")
    ap.add_argument("--box3_data", required=True, help="Path to Box3 data directory (.txt files)")
    ap.add_argument("--box3_meta", required=True, help="Path to Box3 metadata CSV")
    ap.add_argument("--out_dir", required=True, help="Output root directory (e.g., D:\\New modules\\QC)")
    ap.add_argument("--include_types", default="DM1,Control", help="Comma-separated types (default DM1,Control)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    include_types = tuple([t.strip() for t in args.include_types.split(",") if t.strip()])
    # Exclude DM2 and DMD samples (by FilePrefix) so strict loader doesn't choke


    # Run QC for each box
    df12 = qc_box(
        box_name="Box1-2",
        data_dir=Path(args.box12_data),
        meta_path=Path(args.box12_meta),
        out_root=out_root,
        include_types=include_types,
    )
    
    df3 = qc_box(
        box_name="Box3",
        data_dir=Path(args.box3_data),
        meta_path=Path(args.box3_meta),
        out_root=out_root,
        include_types=include_types,
    )


    df = pd.concat([df12, df3], ignore_index=True)
    df = classify(df, Thresholds())

    # Save per-spectrum metrics
    df.to_csv(out_root / "QC_metrics_per_spectrum.csv", index=False)

    # Group summaries (Box × Type)
    summary = (
        df.groupby(["box", "type"])
          .agg(
              n_spectra=("spectrum_qc", "size"),
              n_good=("spectrum_qc", lambda s: (s == "Good").sum()),
              n_suspect=("spectrum_qc", lambda s: (s == "Suspect").sum()),
              n_failed=("spectrum_qc", lambda s: (s == "Failed").sum()),
              n_samples=("sample", "nunique"),
              n_sample_failed=("sample_failed", lambda s: int(pd.Series(s).fillna(False).any()))
          )
          .reset_index()
    )
    summary.to_csv(out_root / "QC_summary_by_group.csv", index=False)

    # Excel workbook with convenient sheets
    xlsx_path = out_root / "QC_flags.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="per_spectrum", index=False)
        summary.to_excel(xw, sheet_name="summary_box_type", index=False)

        # Worst offenders
        worst_auc = df.sort_values("auc_z").head(50)
        worst_corr = df.sort_values("corr_to_sample_median").head(50)
        worst_smooth = df.sort_values("smooth_z", ascending=False).head(50)
        worst_pbr = df.sort_values("pbr_z").head(50)
        worst_auc.to_excel(xw, sheet_name="worst_auc", index=False)
        worst_corr.to_excel(xw, sheet_name="worst_corr", index=False)
        worst_smooth.to_excel(xw, sheet_name="worst_smooth", index=False)
        worst_pbr.to_excel(xw, sheet_name="worst_pbr", index=False)

        # Failed only
        df[df["spectrum_qc"] == "Failed"].to_excel(xw, sheet_name="failed_only", index=False)

    # -------------------- plotting (blue kept / red discarded) --------------------
    # We re-load spectra once per box to create per-sample QC plots (kept vs discarded).
    per_sample_dir = out_root / "PerSample_QC_Plots"
    failed_dir = out_root / "Flagged_Failed_Spectra"
    ensure_dir(per_sample_dir)
    ensure_dir(failed_dir)

    def make_plots_for_box(box_name: str, data_dir: str, meta_path: str):
        w, avg, all_spectra, filenames, meta = load_data(
            data_dir=data_dir,
            metadata_path=meta_path,
            include_types=include_types,
            return_filenames=True,
            strict=False,
        )
        n_samples = meta.shape[0]
        for s_idx in range(n_samples):
            sample = meta.loc[s_idx, "FilePrefix"]
            typ = meta.loc[s_idx, "Type"]

            cols = slice(s_idx * 9, (s_idx + 1) * 9)
            Y9 = all_spectra[:, cols]

            df_s = df[(df["box"] == box_name) & (df["sample"] == sample)].copy()
            df_s = df_s.sort_values(["filename", "position_nm"])


            # Align keep mask to the 9 spectra in the loader order:
            # loader order is [file1]*3 + [file2]*3 + [file3]*3; we can reconstruct by filenames list
            fn9 = filenames[s_idx * 9 : (s_idx + 1) * 9]

            # NEW: cache true position labels for each file in this sample
            pos_by_fn = {f: extract_positions_from_txt(Path(data_dir) / f) for f in set(fn9)}

            seen = {}
            keep_mask = []
            for j in range(9):
                fn = fn9[j]
                k = seen.get(fn, 0)
                seen[fn] = k + 1

                position_nm = pos_by_fn[fn][k]

                r = df_s[(df_s["filename"] == fn) & (df_s["position_nm"] == position_nm)]
                if len(r) != 1:
                    raise RuntimeError(f"QC lookup failed/ambiguous: box={box_name} sample={sample} fn={fn} pos={position_nm} matches={len(r)}")
                keep_mask.append(bool(r["keep"].iloc[0]))


            keep_mask = np.array(keep_mask, dtype=bool)


            title = f"{box_name} | {sample} | Type={typ} | blue=kept red=discarded"
            out_path = per_sample_dir / f"{box_name}__{sample}__{typ}__QC.png"
            plot_sample_qc(w=w, Y9=Y9, keep_mask=keep_mask, out_path=out_path, title=title)

            # If any discarded spectra, also save a focused plot
            if (~keep_mask).any():
                out_path2 = failed_dir / f"{box_name}__{sample}__{typ}__FAILED.png"
                plot_sample_qc(w=w, Y9=Y9, keep_mask=keep_mask, out_path=out_path2,
                               title=title + " (FAILED PRESENT)", lw_keep=1.2, lw_drop=2.6)

    make_plots_for_box("Box1-2", args.box12_data, args.box12_meta)
    make_plots_for_box("Box3", args.box3_data, args.box3_meta)



    print(f"[QC] Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
