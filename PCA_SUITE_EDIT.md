Please edit the PCA suite so that **every configuration run in the pipeline saves every possible visualization and metric artifact**, for **both `PATIENT_AVG` and `ALL_SPECTRA`** modes.

I want the save behavior changed from selective/winner-oriented output to **full per-run artifact saving**.

## Goal

For **every combination** of:

* spectral window
* preprocessing
* PCA concept / weights definition
* plot mode (`PATIENT_AVG`, `ALL_SPECTRA`)

save **all figures and run metrics** that can be generated for that exact run.

This includes both PCA concepts:

1. `WEIGHTS__BOX12_DM1_ONLY`

   * fit on Cohort 1 DM1 only
   * project Cohort 1 controls + Cohort 2 DM1
2. `WEIGHTS__BOX12_ALL`

   * fit on Cohort 1 DM1 + controls
   * project Cohort 2

## I need saved for EVERY run

For each run and for each mode (`PATIENT_AVG`, `ALL_SPECTRA`), save:

* regular PCA group scatter
* SI gradient PCA scatter
* enhanced PCA scatter
* loadings plot
* spectra + wavenumber importance overlay
* top wavenumbers CSV
* metrics JSON

## Metrics JSON should include, whenever available

* alignment metrics
* separation metrics
* gradient metrics / grad_score
* explained variance
* concept
* preprocessing
* spectrum mode
* spectral window
* any question-routing metadata already used internally

If a metric is not applicable for a given run, keep the key and store `null` / `NaN` rather than omitting it.

## Critical requirements

* Save outputs for BOTH `PATIENT_AVG` and `ALL_SPECTRA`
* Do NOT save loadings/overlay only for `PATIENT_AVG` anymore
* Do NOT restrict visualizations to winners only
* Do NOT change any scientific calculations, PCA fitting, preprocessing, ranking, or metrics logic
* Only change output generation and filename/path handling
* Prevent overwriting by making filenames mode-specific

## Required filename behavior

Within each run folder, expected files should look like:

* `PATIENT_AVG__group.png`
* `PATIENT_AVG__si_gradient.png`
* `PATIENT_AVG__enhanced.png`
* `PATIENT_AVG__loadings.png`
* `PATIENT_AVG__overlay.png`
* `PATIENT_AVG__top_wavenumbers.csv`
* `PATIENT_AVG__metrics.json`
* `ALL_SPECTRA__group.png`
* `ALL_SPECTRA__si_gradient.png`
* `ALL_SPECTRA__enhanced.png`
* `ALL_SPECTRA__loadings.png`
* `ALL_SPECTRA__overlay.png`
* `ALL_SPECTRA__top_wavenumbers.csv`
* `ALL_SPECTRA__metrics.json`

If the current folder structure makes this awkward, a small per-mode subfolder is acceptable, but keep the overall output layout as stable as possible.

## What to inspect and modify

Please inspect:

* `pca_runner_core.py`
* `pca_plotting.py`
* `pca_outputs.py`
  and any helper module needed for path/file naming.

## After editing, report back with:

1. exactly which files were changed
2. what the old save behavior was
3. what the new save behavior is
4. an example of the exact output files expected for:

   * `RAW + FULL_SPECTRUM + WEIGHTS__BOX12_DM1_ONLY`
   * for both `PATIENT_AVG` and `ALL_SPECTRA`
5. whether a full rerun is required, or whether I can rerun only a subset to generate the missing assets

## Important verification

Please confirm that:

* `ALL_SPECTRA` loadings/overlay/top-wavenumbers are generated from the actual `ALL_SPECTRA` PCA model and spectra matrices
* `PATIENT_AVG` assets are generated from the actual `PATIENT_AVG` PCA model and matrices
* no asset is silently reused from the other mode
