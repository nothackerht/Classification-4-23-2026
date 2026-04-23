"""
classification_outlier_audit.py — CLI entry point for the SERS outlier audit pipeline.

Reads saved classification result folders (no retraining) and identifies
patient samples that are persistently hard to predict.

Usage
-----
python classification_outlier_audit.py \
    --results-root "Classification Results" \
    --output-root "Outlier Audit/Results" \
    --data-root "../data" \
    --qc-csv "QC/QC_metrics_per_spectrum.csv" \
    --top-n 10 \
    --misclass-threshold 0.4 \
    --scenarios box12 box123 train12
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import audit_helpers as ah

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SERS classification outlier audit pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-root",
        default="Classification Results",
        help="Root directory to scan recursively for predictions.csv files",
    )
    p.add_argument(
        "--output-root",
        default="Outlier Audit/Results",
        help="Directory where all outputs are written",
    )
    p.add_argument(
        "--data-root",
        default="../data",
        help="Location of Box12_spectra/, Box3_spectra/, metadata CSVs",
    )
    p.add_argument(
        "--qc-csv",
        default="QC/QC_metrics_per_spectrum.csv",
        help="QC metrics CSV with columns: sample, filename, keep",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top-ranked suspicious patients to generate per-patient spectral figures for",
    )
    p.add_argument(
        "--misclass-threshold",
        type=float,
        default=0.4,
        help="Minimum misclassification rate for PERSISTENT_MISCLASS flag",
    )
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        metavar="SCENARIO",
        help="Restrict to subset of scenarios (partial match, case-insensitive). "
             "E.g. --scenarios box12 box123 train12. Default: all.",
    )
    return p


def _resolve(path_str: str, script_dir: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = script_dir / p
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    results_root = _resolve(args.results_root, script_dir)
    output_root = _resolve(args.output_root, script_dir)
    data_root = _resolve(args.data_root, script_dir)
    qc_csv_path = _resolve(args.qc_csv, script_dir)

    log.info("Results root : %s", results_root)
    log.info("Output root  : %s", output_root)
    log.info("Data root    : %s", data_root)
    log.info("QC CSV       : %s", qc_csv_path)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(exist_ok=True)
    (output_root / "figures" / "FlaggedPatients").mkdir(parents=True, exist_ok=True)

    # ── Part 2: Load all predictions ─────────────────────────────────────────
    log.info("Loading predictions from %s …", results_root)
    master_df = ah.load_all_predictions(results_root)

    if master_df.empty:
        log.error("No predictions loaded. Check --results-root. Aborting.")
        sys.exit(1)

    log.info("Loaded %d prediction rows from %d unique runs",
             len(master_df), master_df.groupby(["scenario", "mode", "variant", "preproc", "model"]).ngroups)

    # Filter to requested scenarios
    if args.scenarios:
        filters = [s.lower() for s in args.scenarios]
        mask = master_df["scenario"].str.lower().apply(
            lambda s: any(f in s for f in filters)
        )
        master_df = master_df[mask].copy()
        log.info("After scenario filter: %d rows remain", len(master_df))

    # ── Part 3: Patient-level statistics ─────────────────────────────────────
    log.info("Computing LOSO patient statistics …")
    loso_stats = ah.compute_loso_patient_stats(master_df)

    log.info("Computing Train12→Predict3 patient statistics …")
    t12p3_stats = ah.compute_t12p3_patient_stats(master_df)

    log.info("Computing per-variant breakdown …")
    variant_breakdown = ah.compute_variant_breakdown(master_df)

    # ── Part 4: Flags ─────────────────────────────────────────────────────────
    log.info("Applying LOSO flags (misclass_threshold=%.2f) …", args.misclass_threshold)
    loso_flagged = ah.apply_loso_flags(loso_stats, variant_breakdown,
                                       misclass_threshold=args.misclass_threshold)

    log.info("Applying Train12→Predict3 flags …")
    t12p3_flagged = ah.apply_t12p3_flags(t12p3_stats, variant_breakdown)

    log.info("Applying QC/CORAL directionality flags …")
    loso_flagged, t12p3_flagged = ah.apply_directionality_flags(
        loso_flagged, t12p3_flagged, variant_breakdown
    )

    # ── Part 5: Suspicion scores & ranked table ───────────────────────────────
    log.info("Computing suspicion scores …")
    loso_scored = ah.compute_loso_suspicion_score(loso_flagged)
    t12p3_scored = ah.compute_t12p3_suspicion_score(t12p3_flagged)
    ranked_table = ah.build_ranked_table(loso_scored, t12p3_scored)

    # ── Part 6: Global visualizations ────────────────────────────────────────
    figures_dir = output_root / "figures"
    log.info("Plotting global heatmap …")
    ah.plot_global_heatmap(master_df, figures_dir)

    log.info("Plotting misclassification barplot …")
    ah.plot_misclassification_barplot(loso_scored, t12p3_scored, figures_dir)

    top_flagged_ids = ranked_table["patient_id"].head(args.top_n).tolist()
    log.info("Plotting scenario comparison for top-%d patients …", len(top_flagged_ids))
    ah.plot_scenario_comparison(master_df, top_flagged_ids, figures_dir)

    # ── Part 7–8: Per-patient spectral figures ────────────────────────────────
    log.info("Loading QC data from %s …", qc_csv_path)
    qc_df = ah.load_qc_data(qc_csv_path)

    if top_flagged_ids:
        log.info("Loading class-average spectra for spectral figures …")
        # Determine y_true for each flagged patient from ranked_table
        pid_ytrue = {}
        for _, row in ranked_table.iterrows():
            pid_ytrue[row["patient_id"]] = row.get("y_true", None)

        # Build class averages once (they don't change per patient)
        class_avgs: dict = {}
        for y_true_label in [0, 1]:
            class_avgs[y_true_label] = ah.load_class_avg_spectra(
                y_true_label, master_df, data_root
            )

        for patient_id in top_flagged_ids:
            log.info("  Generating spectral figures for %s …", patient_id)
            patient_dir = output_root / "figures" / "FlaggedPatients" / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            spectra_result = ah.load_patient_spectra(patient_id, data_root)
            if spectra_result is None:
                log.warning("    Spectral data not found for %s — skipping spectral plots", patient_id)
            else:
                all_spectra, wavenumbers = spectra_result
                y_true = pid_ytrue.get(patient_id)

                if y_true is not None:
                    same_class_avgs = class_avgs.get(int(y_true), {})
                    opp_class_avgs = class_avgs.get(1 - int(y_true), {})
                else:
                    same_class_avgs = {}
                    opp_class_avgs = {}

                ah.plot_raw_overlay(
                    patient_id, y_true, all_spectra, wavenumbers,
                    same_class_avgs, opp_class_avgs, patient_dir
                )
                ah.plot_class_mean_comparison(
                    patient_id, y_true, all_spectra, wavenumbers,
                    same_class_avgs, opp_class_avgs, patient_dir
                )
                ah.plot_difference_from_class_mean(
                    patient_id, y_true, all_spectra, wavenumbers,
                    same_class_avgs, opp_class_avgs, patient_dir
                )

                if qc_df is not None:
                    ah.plot_qc_overlay(patient_id, all_spectra, wavenumbers,
                                       qc_df, patient_dir)

            ah.plot_score_distribution(patient_id, master_df, patient_dir)

    # ── Part 9: Provenance ────────────────────────────────────────────────────
    log.info("Building provenance report for flagged patients …")
    flagged_for_prov = ranked_table.head(args.top_n)
    provenance_df = ah.build_provenance_report(flagged_for_prov, data_root, qc_df)

    # ── Part 10: Write outputs ────────────────────────────────────────────────
    log.info("Writing tables …")
    ah.write_tables(loso_scored, t12p3_scored, ranked_table, provenance_df, output_root)

    log.info("Writing Excel workbook …")
    ah.write_excel_workbook(loso_scored, t12p3_scored, ranked_table, output_root)

    log.info("Writing plain-text summary …")
    ah.write_summary_text(loso_scored, t12p3_scored, ranked_table, output_root)

    log.info("Audit complete. Outputs written to: %s", output_root)


if __name__ == "__main__":
    main()
