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

    looks_prefixed = sample_id.str.match(r"^(DM1|AdCo)_[^_]+$")
    fileprefix = sample_id.where(looks_prefixed, None)

    need_construct = fileprefix.isna()
    if need_construct.any():
        parts = sample_id.str.split("_")
        second = parts.str[1].fillna("")
        built = typ.str.strip() + "_" + second.str.strip()
        fileprefix = fileprefix.fillna(built)

    ok = fileprefix.str.match(r"^(DM1|AdCo)_[0-9A-Za-z]+$")
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


def load_data(
    data_dir: str | Path,
    metadata_path: Optional[str | Path] = None,
    include_types: Iterable[str] = ("DM1", "Control"),
    return_filenames: bool = False,
    return_rep_idx: bool = False,  # <-- ADD THIS
    strict: bool = True,
    report_samples: int = 5,
    mapping_csv_path: Optional[str | Path] = None,
    # NEW:
    target_column: Optional[str] = None,
    drop_missing_target: bool = False,
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


    # ---------------- Read averaged & unaveraged in the SAME order ----------------

    averaged_cols: List[np.ndarray] = []
    wavenumbers: Optional[np.ndarray] = None
    per_sample_filenames: List[List[str]] = []  # 9 filenames per sample (3 files × 3 spectra each)

    for ident in common_ids:
        fns = ordered_groups[ident]
        if strict and len(fns) != 3:
            raise ValueError(f"{ident}: expected 3 files, found {len(fns)} -> {fns}")
        if len(fns) < 3:
            # skip incomplete groups in non-strict mode
            continue

        if wavenumbers is None:
            wavenumbers = _extract_wavenumbers_txt(data_dir / fns[0])

        three_blocks = [_load_txt_block(data_dir / fn) for fn in fns]   # each (1731,3)
        for fn, arr in zip(fns, three_blocks):
            if arr.shape != (1731, 3):
                raise ValueError(f"{fn}: unexpected array shape {arr.shape}")

        # (1731, 9) for this sample; average across axis=1
        arr9 = np.hstack(three_blocks)
        averaged_cols.append(arr9.mean(axis=1))

        # provenance: list the 9 contributing filenames in order
        per_sample_filenames.append([fns[0]]*3 + [fns[1]]*3 + [fns[2]]*3)

    if wavenumbers is None:
        raise ValueError("No valid files to extract wavenumbers from.")

    averaged_spectra = np.column_stack(averaged_cols)  # (1731, N)

    unavg_blocks: List[np.ndarray] = []
    filenames_per_column: List[str] = []
    rep_idx_per_column: List[int] = []   # <-- ADD THIS
    
    for ident in common_ids:
        fns = ordered_groups[ident]
        if len(fns) < 3:
            continue
        for fn in fns:
            block = _load_txt_block(data_dir / fn)   # (1731,3)
            unavg_blocks.append(block)
            filenames_per_column.extend([fn] * 3)
    
            # IMPORTANT: explicit rep indices for the 3 spectra inside this file
            rep_idx_per_column.extend([0, 1, 2])     # <-- ADD THIS
    
    all_spectra = np.hstack(unavg_blocks)  # (1731, 9N)
    # ---- NEW SAFETY CHECK ----
    if len(rep_idx_per_column) != len(filenames_per_column):
        raise AssertionError(
            "rep_idx_per_column length mismatch with filenames_per_column"
        )
    if all_spectra.shape[1] != len(filenames_per_column):
        raise AssertionError("all_spectra columns mismatch with filenames_per_column")

    # ---------------- Alignment validations ----------------
    _validate_alignment(
        wavenumbers=wavenumbers,
        averaged_spectra=averaged_spectra,
        all_spectra=all_spectra,
        filenames_per_column=filenames_per_column,
        aligned_meta=aligned_meta,
        strict=strict,
    )

    # ---------------- Alignment report (print & optional CSV) ----------------
    _print_alignment_report(aligned_meta["FilePrefix"].tolist(),
                            per_sample_filenames,
                            max_samples=report_samples)

    if mapping_csv_path is not None:
        _export_alignment_mapping(aligned_meta, per_sample_filenames, mapping_csv_path)

    if return_filenames and return_rep_idx:
        return wavenumbers, averaged_spectra, all_spectra, filenames_per_column, rep_idx_per_column, aligned_meta
    elif return_filenames:
        return wavenumbers, averaged_spectra, all_spectra, filenames_per_column, aligned_meta
    else:
        return wavenumbers, averaged_spectra, all_spectra, aligned_meta



# ============================== Validation & Reporting ==============================

def _validate_alignment(
    *,
    wavenumbers: np.ndarray,
    averaged_spectra: np.ndarray,
    all_spectra: np.ndarray,
    filenames_per_column: List[str],
    aligned_meta: pd.DataFrame,
    strict: bool = True,
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
    if all_spectra.shape[1] != 9 * n_samples:
        raise AssertionError(f"all_spectra has {all_spectra.shape[1]} columns, expected {9*n_samples}")

    if len(filenames_per_column) != 9 * n_samples:
        raise AssertionError("filenames_per_column length mismatch with all_spectra columns")

    if len(aligned_meta) != n_samples:
        raise AssertionError("aligned_meta length must equal averaged_spectra columns")

    if strict:
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
