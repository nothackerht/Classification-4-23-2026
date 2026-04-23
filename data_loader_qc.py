# -*- coding: utf-8 -*-
"""
Robust, deterministic data loader for SERS grid:
- One canonical, sorted ID list drives reading order
- Metadata is reindexed to exactly that order
- Strong shape/consistency checks at every step
- Provenance: filename -> column mapping
- Alignment report for quick eyeballing
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# QC KEY NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_fn_for_key(fn: str) -> str:
    """
    Canonical filename for use inside a spectrum_key.

    Normalizes environment-specific differences so the same physical file
    produces the same key on Windows and Linux (Athena):

      1. Basename only — strips any accidental leading directory component.
      2. '%' → '_' — Windows path handling replaces '%' with '_' when copying
         files, turning '10%_10Vis' into '10__10Vis'.  This rule makes both
         representations collapse to the same canonical form.

    The two-step mapping is one-to-one within a single patient's three Map
    files, so different files cannot collapse to the same normalized name.
    """
    fn = os.path.basename(fn)
    fn = fn.replace("%", "_")
    return fn


def _normalize_spectrum_key(key: str) -> str:
    """
    Normalize the filename component (field index 2) of a stored spectrum_key.

    spectrum_key format:  box|sample|filename|posN
    Returns the key unchanged if it does not have exactly 4 '|'-separated parts.
    """
    parts = key.split("|")
    if len(parts) == 4:
        parts[2] = normalize_fn_for_key(parts[2])
        return "|".join(parts)
    return key


# ============================== Low-level I/O ==============================

def _extract_wavenumbers_txt(path: Path) -> np.ndarray:
    """
    Read the first line (tab-delimited) and parse wavenumbers.
    Expected: 1732 entries (1 label + 1731 numbers).
    Returns (1731,) float64.
    """
    with path.open("r") as f:
        first_line = f.readline().rstrip("\n").split("\t")
    if len(first_line) != 1732:
        raise ValueError(f"{path.name}: Expected 1732 header columns, got {len(first_line)}")
    try:
        w = np.array([float(x) for x in first_line[1:]], dtype=np.float64)
    except Exception as e:
        raise ValueError(f"{path.name}: failed to parse wavenumbers -> {e}") from e
    if w.shape != (1731,):
        raise ValueError(f"{path.name}: wavenumbers shape {w.shape}, expected (1731,)")
    return w


def _load_txt_block(path: Path) -> np.ndarray:
    """
    Load a single .txt file and return spectra as (1731, 3).
    File format:
      row0: [label, wn1, wn2, ..., wn1731]           (1732 entries)
      row1..3: [label, s(wn1), ..., s(wn1731)]       (1732 entries each)
    """
    raw = np.genfromtxt(str(path), delimiter="\t", dtype=str)

    # drop fully empty rows
    raw = np.array([row for row in raw if any(cell.strip() for cell in row)], dtype=object)

    if raw.shape[0] != 4:
        raise ValueError(f"{path.name}: Expected 4 rows, got {raw.shape[0]}")
    if raw.shape[1] != 1732:
        raise ValueError(f"{path.name}: Expected 1732 columns, got {raw.shape[1]}")

    try:
        spectra = raw[1:, 1:].astype(np.float64)  # (3, 1731)
    except Exception as e:
        raise ValueError(f"{path.name}: numeric cast failed -> {e}") from e

    if spectra.shape != (3, 1731):
        raise ValueError(f"{path.name}: spectra shape {spectra.shape}, expected (3, 1731)")

    return spectra.T  # (1731, 3)


# ============================== ID mapping ==============================

def _file_identifier(fname: str) -> str:
    """
    Canonical identifier = first two underscore-separated tokens.
    e.g., 'DM1_080_Map1_...' -> 'DM1_080', 'AdCo_001_...' -> 'AdCo_001'
    """
    parts = fname.split("_")
    if len(parts) < 2:
        raise ValueError(f"Filename has no two-token prefix: '{fname}'")
    return f"{parts[0]}_{parts[1]}"


def _index_files(data_dir: Path) -> Dict[str, List[str]]:
    """
    Group *.txt by identifier. Filenames within each ID are sorted
    for deterministic order.
    """
    groups: Dict[str, List[str]] = {}
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".txt"):
            continue
        ident = _file_identifier(fn)
        groups.setdefault(ident, []).append(fn)
    for k in groups:
        groups[k].sort()
    return groups


def _canonicalize_fileprefix_column(meta: pd.DataFrame) -> pd.Series:
    """
    Build a robust 'FilePrefix' column that matches filename identifiers.

    Priority:
      1) If Sample_ID already looks like 'DM1_###' or 'AdCo_###', use it.
      2) Else, construct from Type + second token of Sample_ID.
    """
    if "Sample_ID" not in meta or "Type" not in meta:
        raise ValueError("Metadata must contain 'Sample_ID' and 'Type' columns.")

    sample_id = meta["Sample_ID"].astype(str)
    typ = meta["Type"].astype(str)

    looks_prefixed = sample_id.str.match(r"^[A-Za-z0-9]+_[^_]+$")
    fileprefix = sample_id.where(looks_prefixed, None)

    need_construct = fileprefix.isna()
    if need_construct.any():
        parts = sample_id.str.split("_")
        second = parts.str[1].fillna("")
        built = typ.str.strip() + "_" + second.str.strip()
        fileprefix = fileprefix.fillna(built)

    ok = fileprefix.str.match(r"^[A-Za-z0-9]+_[0-9A-Za-z]+$")
    if not ok.all():
        bad = fileprefix[~ok]
        raise ValueError(f"Could not canonicalize FilePrefix for rows:\n{bad}")

    return fileprefix


# ============================== Public API ==============================

def load_metadata(
    csv_path: str | Path,
    include_types: Iterable[str] = ("DM1", "Control"),
) -> pd.DataFrame:
    """
    Load and filter metadata. Returns a copy with a canonical 'FilePrefix'.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Type"].isin(include_types)].copy()
    if df.empty:
        raise ValueError(f"No rows after filtering for include_types={include_types}")
    df["Sample_ID"] = df["Sample_ID"].astype(str)
    df["FilePrefix"] = _canonicalize_fileprefix_column(df)
    return df
def extract_positions_from_txt(txt_path: Path) -> list[int]:
    """
    Reads the row labels (positions) for the 3 spectra inside your mapping .txt.
    Your file format (same as _load_txt_block):
      row0: [label, wn1, ..., wn1731]
      row1..3: [pos_label, s(wn1), ..., s(wn1731)]
    Returns the 3 position labels in file order, e.g. [-200, 0, 200].
    """
    raw = np.genfromtxt(str(txt_path), delimiter="\t", dtype=str)
    raw = np.array([row for row in raw if any(str(cell).strip() for cell in row)], dtype=object)

    if raw.shape[0] != 4:
        raise ValueError(f"{txt_path.name}: Expected 4 rows, got {raw.shape[0]}")
    if raw.shape[1] != 1732:
        raise ValueError(f"{txt_path.name}: Expected 1732 columns, got {raw.shape[1]}")

    pos = []
    for r in raw[1:, 0]:
        try:
            pos.append(int(float(str(r).strip())))
        except Exception as e:
            raise ValueError(f"{txt_path.name}: could not parse position label '{r}' -> {e}") from e

    if len(pos) != 3:
        raise ValueError(f"{txt_path.name}: Expected 3 positions, got {len(pos)}")
    return pos


def build_spectrum_keys_for_sample(box: str, sample: str, fn9: list[str], data_dir: Path) -> list[str]:
    """
    Reconstruct the 9 spectrum_key strings in the SAME column order as all_spectra for that sample:
      [file1]*3 + [file2]*3 + [file3]*3
    and within each file uses occurrence 0/1/2 mapped to the true position_nm labels.
    """
    pos_by_fn = {fn: extract_positions_from_txt(data_dir / fn) for fn in set(fn9)}

    seen = {}
    keys = []
    for fn in fn9:
        k = seen.get(fn, 0)
        seen[fn] = k + 1
        position_nm = pos_by_fn[fn][k]
        keys.append(f"{box}|{sample}|{normalize_fn_for_key(fn)}|pos{position_nm}")
    return keys


def load_data(
    data_dir: str | Path,
    metadata_path: Optional[str | Path] = None,
    include_types: Iterable[str] = ("DM1", "Control"),
    return_filenames: bool = False,
    strict: bool = True,
    report_samples: int = 5,
    mapping_csv_path: Optional[str | Path] = None,
    target_column: Optional[str] = None,
    drop_missing_target: bool = False,
    ignore_prefixes: Optional[Iterable[str]] = None,

    # --- NEW (QC integration) ---
    qc_metrics_path: Optional[str | Path] = None,   # QC_metrics_per_spectrum.csv OR QC_flags.xlsx
    qc_box: Optional[str] = None,                   # "Box12" / "Box3" (whatever you call it in QC)
    min_kept_per_sample: int = 1,                   # drop sample if fewer kept spectra
):


    """
    Deterministic, validated loader.

    Returns:
      wavenumbers:            (1731,)
      averaged_spectra:       (1731, N)        columns aligned to aligned_meta rows
      all_spectra:            (1731, 9N)       9 spectra per sample, same order
      filenames_per_column:   list[str] length 9N (if return_filenames=True)
      aligned_meta:           metadata DataFrame reindexed to this exact order
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    groups = _index_files(data_dir)  # { 'DM1_080': [f1,f2,f3], ... }
    # NEW: ignore certain file groups entirely (e.g., DM2/DMD spectra-only cohorts)
    if ignore_prefixes:
        ignore_prefixes = tuple(ignore_prefixes)
        groups = {k: v for k, v in groups.items() if not str(k).startswith(ignore_prefixes)}
    

    if metadata_path is not None:
        meta = load_metadata(metadata_path, include_types=include_types)
        file_ids = set(groups.keys())
        meta_ids = set(meta["FilePrefix"].tolist())
        common_ids = sorted(file_ids.intersection(meta_ids))

        if strict:
            missing_in_files = sorted(meta_ids - file_ids)
            missing_in_meta = sorted(file_ids - meta_ids)
            if missing_in_files:
                raise FileNotFoundError(
                    "These metadata samples have no matching .txt files: "
                    + ", ".join(missing_in_files)
                )
            if missing_in_meta:
                raise ValueError(
                    "These file groups have no matching metadata rows: "
                    + ", ".join(missing_in_meta)
                )
        if not common_ids:
            raise ValueError("No overlapping sample IDs between files and metadata.")

        aligned_meta = (
            meta.set_index("FilePrefix")
                .loc[common_ids]   # exact order
                .reset_index()
        )
        ordered_groups = {k: groups[k] for k in common_ids}

        # --- NEW: optionally drop samples with non-finite target values ---
        if drop_missing_target and (target_column is not None):
            if target_column not in aligned_meta.columns:
                raise KeyError(
                    f"target_column '{target_column}' not present in metadata columns: "
                    f"{aligned_meta.columns.tolist()}"
                )

            # Coerce to numeric; NaNs will mark non-finite
            t = pd.to_numeric(aligned_meta[target_column], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(t)

            if not mask.any():
                raise ValueError(
                    f"No finite values found in target_column='{target_column}' after filtering."
                )

            # Report what was dropped
            if (~mask).any():
                dropped_ids = aligned_meta.loc[~mask, "FilePrefix"].tolist()
                print(
                    f"[data_loader] Dropping {len(dropped_ids)} sample(s) with missing/non-finite "
                    f"'{target_column}': {', '.join(dropped_ids)}"
                )

            # Keep only finite-target samples
            aligned_meta = aligned_meta.loc[mask].reset_index(drop=True)

            # Also shrink the file groups & common_ids to match
            keep_ids = aligned_meta["FilePrefix"].tolist()
            ordered_groups = {k: ordered_groups[k] for k in keep_ids if k in ordered_groups}
            common_ids = keep_ids

    else:
        # No metadata provided -> deterministic by filename groups only
        common_ids = sorted(groups.keys())
        aligned_meta = pd.DataFrame({"FilePrefix": common_ids})
        ordered_groups = {k: groups[k] for k in common_ids}

    # ---------------- QC keep map (spectrum_key -> keep) ----------------
    keep_by_key = None
    if qc_metrics_path is not None:
        qc_metrics_path = Path(qc_metrics_path)
        if qc_metrics_path.suffix.lower() == ".csv":
            qc_df = pd.read_csv(qc_metrics_path)
        else:
            qc_df = pd.read_excel(qc_metrics_path, sheet_name="per_spectrum")

        if "spectrum_key" not in qc_df.columns or "keep" not in qc_df.columns:
            raise KeyError("QC file must contain columns: spectrum_key, keep")

        _all_box_values = qc_df["box"].unique().tolist() if "box" in qc_df.columns else []
        if qc_box is not None and "box" in qc_df.columns:
            qc_df = qc_df[qc_df["box"] == qc_box].copy()
            if qc_df.empty:
                raise KeyError(
                    f"QC file has 0 rows for box={qc_box!r}. "
                    f"Available box values in file: {_all_box_values}. "
                    "Check --qc-box-train / --qc-box-test arguments."
                )

        # Normalize the filename component of every stored key so that
        # environment-specific differences (e.g. '%' vs '__') are erased
        # before comparison.  The constructed keys (build_spectrum_keys_for_sample)
        # apply the same normalization via normalize_fn_for_key().
        keep_by_key = {
            _normalize_spectrum_key(k): bool(v)
            for k, v in zip(qc_df["spectrum_key"].astype(str), qc_df["keep"].astype(bool))
        }
    # ---------------- Read averaged & unaveraged in the SAME order ----------------

    averaged_cols: List[np.ndarray] = []
    wavenumbers: Optional[np.ndarray] = None
    per_sample_filenames: List[List[str]] = []  # provenance per kept sample

    # NEW: track which samples survive after QC + which spectra columns survive
    kept_meta_rows: List[int] = []
    kept_blocks_for_unavg: List[np.ndarray] = []      # list of (1731, k_i) blocks per kept sample
    kept_filenames_for_unavg: List[str] = []          # column-aligned filenames
    kept_spectrum_keys: List[str] = []                # column-aligned spectrum_key (only if QC used)
    spectrum_sample_ids: List[str] = []               # column-aligned sample ID for grouping downstream

    for s_idx, ident in enumerate(common_ids):
        fns = ordered_groups[ident]
        if strict and len(fns) != 3:
            raise ValueError(f"{ident}: expected 3 files, found {len(fns)} -> {fns}")
        if len(fns) < 3:
            continue

        if wavenumbers is None:
            wavenumbers = _extract_wavenumbers_txt(data_dir / fns[0])

        three_blocks = [_load_txt_block(data_dir / fn) for fn in fns]   # each (1731,3)
        for fn, arr in zip(fns, three_blocks):
            if arr.shape != (1731, 3):
                raise ValueError(f"{fn}: unexpected array shape {arr.shape}")

        arr9 = np.hstack(three_blocks)  # (1731, 9)
        fn9 = [fns[0]]*3 + [fns[1]]*3 + [fns[2]]*3

        # QC-filter this sample if requested
        if keep_by_key is not None:
            if qc_box is None:
                raise ValueError("When qc_metrics_path is provided, you must also pass qc_box")

            keys9 = build_spectrum_keys_for_sample(qc_box, ident, fn9, data_dir)

            missing = [k for k in keys9 if k not in keep_by_key]
            if missing:
                _avail = list(keep_by_key.keys())
                _n = min(3, len(missing))
                raise KeyError(
                    f"QC key mismatch: {len(missing)}/9 spectrum_key(s) for sample "
                    f"'{ident}' not found in QC file after filename normalization.\n"
                    f"  Constructed (missing): {missing[:_n]}\n"
                    f"  Available in QC file : {_avail[:_n]}\n"
                    "Likely causes:\n"
                    "  1. QC file was generated from different files than those in "
                    "data_dir (re-run qc_sers_spectra.py on Athena).\n"
                    "  2. normalize_fn_for_key() does not cover all differences.\n"
                    "  3. Box label mismatch (check --qc-box-train / --qc-box-test).\n"
                    "Run validate_qc_keys() from data_loader_qc.py for a full "
                    "per-sample diagnostic."
                )
            keep9 = [bool(keep_by_key[k]) for k in keys9]

            keep_idx = [i for i, kk in enumerate(keep9) if kk]
            if len(keep_idx) < min_kept_per_sample:
                # drop entire sample
                continue

            arr_keep = arr9[:, keep_idx]
            avg = arr_keep.mean(axis=1)

            kept_blocks_for_unavg.append(arr_keep)
            kept_filenames_for_unavg.extend([fn9[i] for i in keep_idx])
            kept_spectrum_keys.extend([keys9[i] for i in keep_idx])
            spectrum_sample_ids.extend([ident] * len(keep_idx))

        else:
            avg = arr9.mean(axis=1)
            kept_blocks_for_unavg.append(arr9)
            kept_filenames_for_unavg.extend(fn9)
            spectrum_sample_ids.extend([ident] * 9)

        averaged_cols.append(avg)
        per_sample_filenames.append(fn9)

        kept_meta_rows.append(s_idx)

    if wavenumbers is None:
        raise ValueError("No valid files to extract wavenumbers from.")

    averaged_spectra = np.column_stack(averaged_cols)  # (1731, N_kept)

    # Unaveraged data (QC-aware): variable number of spectra per sample
    all_spectra = np.hstack(kept_blocks_for_unavg) if kept_blocks_for_unavg else np.empty((1731, 0))
    filenames_per_column = kept_filenames_for_unavg

    # IMPORTANT: align metadata to kept samples only
    aligned_meta = aligned_meta.iloc[kept_meta_rows].reset_index(drop=True)

    # ---------------- Alignment validations ----------------
    _validate_alignment(
        wavenumbers=wavenumbers,
        averaged_spectra=averaged_spectra,
        all_spectra=all_spectra,
        filenames_per_column=filenames_per_column,
        aligned_meta=aligned_meta,
        strict=strict,
        qc_active=(keep_by_key is not None),
    )

    # ---------------- Alignment report (print & optional CSV) ----------------
    _print_alignment_report(aligned_meta["FilePrefix"].tolist(),
                            per_sample_filenames,
                            max_samples=report_samples)

    if mapping_csv_path is not None:
        _export_alignment_mapping(aligned_meta, per_sample_filenames, mapping_csv_path)

    if return_filenames:
        if keep_by_key is not None:
            return (
                wavenumbers,
                averaged_spectra,
                all_spectra,
                filenames_per_column,
                aligned_meta,
                spectrum_sample_ids,
                kept_spectrum_keys,
            )
        else:
            return (
                wavenumbers,
                averaged_spectra,
                all_spectra,
                filenames_per_column,
                aligned_meta,
                spectrum_sample_ids,
            )
    else:
        if keep_by_key is not None:
            return (
                wavenumbers,
                averaged_spectra,
                all_spectra,
                aligned_meta,
                spectrum_sample_ids,
                kept_spectrum_keys,
            )
        else:
            return (
                wavenumbers,
                averaged_spectra,
                all_spectra,
                aligned_meta,
                spectrum_sample_ids,
            )


# ============================== Validation & Reporting ==============================

def _validate_alignment(
    *,
    wavenumbers: np.ndarray,
    averaged_spectra: np.ndarray,
    all_spectra: np.ndarray,
    filenames_per_column: List[str],
    aligned_meta: pd.DataFrame,
    strict: bool = True,
    qc_active: bool = False,
) -> None:
    """
    Enforce:
      - (1731,) wavenumbers
      - averaged: (1731, N)
      - unaveraged: (1731, 9N)
      - filenames list length = 9N
      - aligned_meta rows = N
      - (strict) each sample’s 9 columns come from exactly 3 files repeated 3×
    """
    if wavenumbers.shape != (1731,):
        raise AssertionError(f"wavenumbers shape {wavenumbers.shape} != (1731,)")

    n_samples = averaged_spectra.shape[1]
    if averaged_spectra.shape[0] != 1731:
        raise AssertionError("averaged_spectra must be (1731, N)")
    if all_spectra.shape[0] != 1731:
        raise AssertionError("all_spectra must be (1731, 9N)")
    if not qc_active:
        if all_spectra.shape[1] != 9 * n_samples:
            raise AssertionError(
                f"all_spectra has {all_spectra.shape[1]} columns, expected {9*n_samples}"
            )

        if len(filenames_per_column) != 9 * n_samples:
            raise AssertionError(
                "filenames_per_column length mismatch with all_spectra columns"
            )
    else:
        if len(filenames_per_column) != all_spectra.shape[1]:
            raise AssertionError(
                "filenames_per_column length must equal all_spectra columns in QC mode"
            )

    if len(aligned_meta) != n_samples:
        raise AssertionError("aligned_meta length must equal averaged_spectra columns")

    if strict and (not qc_active):
        for i in range(n_samples):
        
            block = filenames_per_column[i*9:(i+1)*9]
            uniq, counts = np.unique(block, return_counts=True)
            if not (len(uniq) == 3 and np.all(counts == 3)):
                raise AssertionError(
                    f"Sample {i}: expected three files repeated 3× each, "
                    f"got uniques={uniq}, counts={counts}"
                )


def _print_alignment_report(ids: List[str], per_sample_files: List[List[str]], max_samples: int = 5) -> None:
    """
    Print a concise alignment report for the first few samples:
      SampleID -> nine contributing filenames (in order)
    """
    n = min(max_samples, len(ids))
    if n <= 0:
        return
    print("\n=== Alignment Report (first {} samples) ===".format(n))
    for i in range(n):
        print(f"{ids[i]}:")
        for fn in per_sample_files[i]:
            print(f"   {fn}")
    print("===========================================\n")


def _export_alignment_mapping(aligned_meta: pd.DataFrame,
                              per_sample_files: List[List[str]],
                              out_path: str | Path) -> None:
    """
    Write a CSV mapping each Sample (row) to its 9 filenames (cols).
    """
    rows = []
    for i, sid in enumerate(aligned_meta["FilePrefix"].tolist()):
        row = {"FilePrefix": sid}
        for j, fn in enumerate(per_sample_files[i], start=1):
            row[f"file_{j:02d}"] = fn
        rows.append(row)
    df_map = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_map.to_csv(out_path, index=False)
    print(f"[data_loader] Wrote alignment mapping CSV → {out_path}")


# ============================== (Optional) utilities for modeling ==============================

def build_finite_mask(meta: pd.DataFrame, target_column: str) -> np.ndarray:
    """
    Return a boolean mask over *samples* (rows of meta) where target_column is finite.
    Use this to drop samples with NaN targets before modeling.
    """
    if target_column not in meta.columns:
        raise KeyError(f"target_column '{target_column}' not in metadata.")
    t = pd.to_numeric(meta[target_column], errors="coerce").values.astype(float)
    return np.isfinite(t)


# ══════════════════════════════════════════════════════════════════════════════
# QC KEY VALIDATION UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def validate_qc_keys(
    qc_metrics_path: "str | Path",
    data_dir: "str | Path",
    qc_box: str,
    metadata_path: "str | Path",
    include_types: Iterable[str] = ("DM1", "Control"),
    n_report: int = 5,
) -> bool:
    """
    Diagnostic: compare reconstructed spectrum_keys against QC file keys.

    Prints a per-sample match report and returns True if all keys match,
    False if any are missing.  Run this on Athena BEFORE submitting the
    classification job to verify that the QC file is compatible with the
    data directory.

    Quick usage on Athena
    ---------------------
    python -c "
    import sys; sys.path.insert(0, '.')
    from data_loader_qc import validate_qc_keys
    ok = validate_qc_keys(
        qc_metrics_path='~/sers_project/data/QC_metrics_per_spectrum.csv',
        data_dir='~/sers_project/data/Box12_spectra',
        qc_box='Box1-2',
        metadata_path='~/sers_project/data/Box12_metadata.csv',
    )
    "

    Run once for Box12 (qc_box='Box1-2') and once for Box3 (qc_box='Box3').
    """
    qc_metrics_path = Path(qc_metrics_path).expanduser()
    data_dir = Path(data_dir).expanduser()
    metadata_path = Path(metadata_path).expanduser()

    # ── Load QC file ──────────────────────────────────────────────────────────
    if qc_metrics_path.suffix.lower() == ".csv":
        qc_df = pd.read_csv(qc_metrics_path)
    else:
        qc_df = pd.read_excel(qc_metrics_path, sheet_name="per_spectrum")

    if "spectrum_key" not in qc_df.columns:
        print(f"[validate_qc_keys] ERROR: 'spectrum_key' column not found in {qc_metrics_path}")
        return False

    all_boxes = qc_df["box"].unique().tolist() if "box" in qc_df.columns else []
    print(f"[validate_qc_keys] QC file       : {qc_metrics_path}")
    print(f"[validate_qc_keys] Box values    : {all_boxes}")
    print(f"[validate_qc_keys] Filtering for : box={qc_box!r}")

    if "box" in qc_df.columns:
        qc_df = qc_df[qc_df["box"] == qc_box].copy()

    print(f"[validate_qc_keys] Rows after filter: {len(qc_df)}")

    if qc_df.empty:
        print(
            f"[validate_qc_keys] ERROR: no rows for box={qc_box!r}. "
            f"Box label mismatch? Available: {all_boxes}"
        )
        return False

    keep_by_key: dict = {
        _normalize_spectrum_key(k): bool(v)
        for k, v in zip(qc_df["spectrum_key"].astype(str), qc_df["keep"].astype(bool))
    }
    print(f"[validate_qc_keys] Unique keys loaded: {len(keep_by_key)}")

    # ── Load data index + metadata ────────────────────────────────────────────
    meta = load_metadata(metadata_path, include_types=include_types)
    groups = _index_files(data_dir)
    common_ids = sorted(set(meta["FilePrefix"]) & set(groups.keys()))
    print(f"[validate_qc_keys] Samples in both meta + data dir: {len(common_ids)}")

    # ── Per-sample key check ──────────────────────────────────────────────────
    n_ok = n_all_miss = n_partial = 0
    miss_examples: List[str] = []

    for ident in common_ids:
        fns = groups[ident]
        if len(fns) != 3:
            print(f"[validate_qc_keys]   SKIP {ident}: has {len(fns)} files (expected 3)")
            continue

        fn9 = [fns[0]] * 3 + [fns[1]] * 3 + [fns[2]] * 3
        try:
            keys9 = build_spectrum_keys_for_sample(qc_box, ident, fn9, data_dir)
        except Exception as exc:
            print(f"[validate_qc_keys]   FAIL {ident}: key construction error — {exc}")
            n_all_miss += 1
            continue

        missing = [k for k in keys9 if k not in keep_by_key]

        if not missing:
            n_ok += 1
        elif len(missing) == 9:
            n_all_miss += 1
            if len(miss_examples) < n_report:
                _avail_eg = next(iter(keep_by_key), "NONE")
                miss_examples.append(
                    f"  {ident}: ALL 9 missing\n"
                    f"    constructed : {keys9[0]!r}\n"
                    f"    qc_file_eg  : {_avail_eg!r}"
                )
        else:
            n_partial += 1
            if len(miss_examples) < n_report:
                miss_examples.append(
                    f"  {ident}: {len(missing)}/9 missing — e.g. {missing[0]!r}"
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(common_ids)
    print(f"\n[validate_qc_keys] ── Results for box={qc_box!r} ──")
    print(f"  Fully matched  : {n_ok}/{total}")
    print(f"  Partially miss : {n_partial}/{total}")
    print(f"  All 9 missing  : {n_all_miss}/{total}")

    if miss_examples:
        print(f"\n[validate_qc_keys] First {len(miss_examples)} problem sample(s):")
        for ex in miss_examples:
            print(ex)

    all_ok = (n_all_miss == 0 and n_partial == 0)
    if all_ok:
        print("\n[validate_qc_keys] OK — all keys match. QC file is compatible.")
    else:
        print(
            "\n[validate_qc_keys] FAIL — key mismatches detected.\n"
            "  Most likely: filename encoding difference between environments\n"
            "  (e.g. '%' vs '__').  Check normalize_fn_for_key() in data_loader_qc.py,\n"
            "  or re-generate the QC file directly on Athena with qc_sers_spectra.py."
        )
    return all_ok
