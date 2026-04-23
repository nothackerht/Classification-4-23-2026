# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SERS (Surface-Enhanced Raman Spectroscopy) classification pipeline for DM1 (Myotonic Dystrophy Type 1) vs. Control patient diagnosis. Two patient cohorts: Box 1-2 (training/LOSO) and Box 3 (external test).

## Running the Pipeline

**Full classification sweep (on Athena HPC via SLURM):**
```bash
sbatch run_classification_full_qc_coral.sh
```

**Run the sweep script directly (local/dev):**
```bash
python run_classification_preprocessing_sweep_boxes123_athena.py \
  --train-data <Box12_spectra_dir> \
  --test-data <Box3_spectra_dir> \
  --train-meta <Box12_metadata.csv> \
  --test-meta <Box3_metadata.csv> \
  --qc-metrics <QC_metrics_per_spectrum.csv> \
  --qc-box-train "Box1-2" --qc-box-test "Box3" \
  --output <out_root> \
  --scenarios Box12_LOSO_PATIENT_AVG Train12_Predict3_PATIENT_AVG \
  --variants baseline qc_only coral_only qc_coral \
  --preproc RAW SNV NORM EMSC \
  --models svm plsda
```

**QC pipeline:**
```bash
python qc_sers_spectra.py \
  --box12_data <dir> --box12_meta <csv> \
  --box3_data <dir> --box3_meta <csv> \
  --out_dir <output_dir>
```

**Outlier audit (post-run analysis):**
```bash
python classification_outlier_audit.py --results_root <out_root> --output_dir <audit_dir>
```

**Verify weighted PLS:**
```bash
python verify_weighted_pls.py
```

## Architecture

### Data format
- `.txt` files: 4 rows × 1732 columns (row 0 = wavenumber header; rows 1-3 = 3 spectra at positions -200/0/+200 nm)
- Each patient: 3 Map files (Map1/2/3) × 3 spectra = **9 spectra total**
- **1731 wavenumber points** — hard constraint throughout; shape errors will be raised explicitly
- Arrays are always **feature-first**: `(1731, N)` for spectra matrices, `(1731,)` for wavenumbers
- Patient identifier (FilePrefix) = first two underscore-tokens of filename, e.g., `DM1_080`, `AdCo_001`

### Module responsibilities

| Module | Role |
|---|---|
| `data_loader.py` | Deterministic, validated loader. Outputs `averaged_spectra (1731, N)`, `all_spectra (1731, 9N)`, aligned metadata. Strict alignment checks at every step. |
| `data_loader_qc.py` | Drop-in extension that adds `normalize_fn_for_key()` for cross-platform key normalization (`%` → `_`, basename-only). Used to join loader output with QC flags. |
| `preprocessing.py` | Fit/transform preprocessing (SNV, Normalization, 2nd Derivative, EMSC). CORAL alignment function (`coral_align_target_to_source`). **EMSC reference is always fit on training data only.** |
| `preprocessing_new.py` | Updated preprocessing variant; imported as `PreprocessingNew` in the sweep script when available. |
| `weighted_pls.py` | Custom weighted NIPALS PLS. Replaces all inner products with `⟨a,b⟩_w = aᵀ diag(w) b`. API mirrors sklearn: `fit(X, y, sample_weight)`, `transform(X)`, `predict(X)`. |
| `qc_sers_spectra.py` | Per-spectrum QC: computes AUC, peak-to-baseline ratio, smoothness (d² std), and correlation to sample median. Assigns Good/Suspect/Failed. Outputs `QC_metrics_per_spectrum.csv` and `QC_flags.xlsx`. |
| `run_classification_preprocessing_sweep_boxes123_athena.py` | Main sweep. Iterates all (scenario × mode × variant × preproc × model) combinations in parallel via joblib. Writes results to `<out_root>/<scenario>/<mode>/<variant>/<preproc>/<model>/`. |
| `UNIFIED_EXTERNAL_TEST.py` | External test evaluation with neural-net and sklearn models; parallel CV using joblib. |
| `audit_helpers.py` | Post-hoc analysis utilities: discover runs, load all `predictions.csv`, compute per-patient suspicion scores, generate heatmaps and barplots. |
| `classification_outlier_audit.py` | CLI entry point for the outlier audit pipeline (calls `audit_helpers`). |

### Classification scenarios and variants

**Scenarios:** `Box12_LOSO`, `Box123_LOSO`, `Train12_Predict3`  
**Modes:** `PATIENT_AVG` (average 9 spectra → 1 feature vector), `SPECTRUM_LEVEL` (train on individual spectra, aggregate held-out predictions)  
**Variants:** `baseline`, `qc_only`, `coral_only`, `qc_coral`  
- `coral_only`/`qc_coral` only apply in `Train12_Predict3` (Box3 test spectra aligned to Box12 DM1 space)  

**Output path structure:** `<out_root>/<scenario>/<mode>/<variant>/<preproc>/<model>/`  
Each leaf contains `predictions.csv` (columns: `patient_id`, `y_true`, `y_score`, `y_pred`) and `metrics.csv`.

### Critical invariants — do not break
- **EMSC reference** must be computed from training fold only (never from test/held-out data)
- **Baseline correction** is fold-independent (precomputed once per spectrum before CV)
- In `SPECTRUM_LEVEL` mode, patient weighting is enforced via inverse-frequency sample weights so all patients contribute equally regardless of surviving QC spectrum count
- `audit_helpers.parse_prediction_path` supports both 4-level (old, `variant=baseline`) and 5-level (new) directory structures — adding new path levels will break discovery

### QC flag join key
Spectrum keys have format `box|sample|filename|posN` (e.g., `Box1-2|DM1_080|DM1_080_Map1_...|pos0`). The filename component is normalized via `normalize_fn_for_key` to handle `%` vs `_` differences between Windows and Linux file copies.

## HPC / Athena notes

- Conda env: `sers_env`
- `PYTHONPATH` must include the code directory
- Thread parallelism is always pinned to 1 per worker (`OMP_NUM_THREADS=1`, etc.) to avoid contention with joblib process-level parallelism
- `CV_PHYS_CORES` and `CV_MAX_PROC` env vars control worker counts
