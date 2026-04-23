#!/bin/bash
#SBATCH --job-name=classification_full_qc_coral
#SBATCH --partition=cpu-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/lustre/home/nothackerht/sers_project/%x_%j.out
#SBATCH --error=/lustre/home/nothackerht/sers_project/%x_%j.err
#SBATCH --chdir=/lustre/home/nothackerht/sers_project

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLAS_NUM_THREADS=1

module load anaconda3
source /opt/anaconda3-2023.09-0/etc/profile.d/conda.sh
conda activate sers_env

export PYTHONPATH="/lustre/home/nothackerht/sers_project/Classification_code:${PYTHONPATH:-}"

CODE_DIR="/lustre/home/nothackerht/sers_project/Classification_code"
SCRIPT_NAME="run_classification_preprocessing_sweep_boxes123_athena.py"

BOX12_DATA_DIR="/lustre/home/nothackerht/sers_project/data/Box12_spectra"
BOX3_DATA_DIR="/lustre/home/nothackerht/sers_project/data/Box3_spectra"

BOX12_META_PATH="/lustre/home/nothackerht/sers_project/data/Box12_metadata.csv"
BOX3_META_PATH="/lustre/home/nothackerht/sers_project/data/Box3_metadata.csv"

QC_METRICS_PATH="/lustre/home/nothackerht/sers_project/data/QC_metrics_per_spectrum.csv"

OUT_ROOT="/lustre/home/nothackerht/sers_project/results/CLASSIFICATION_FULL_QC_CORAL"
mkdir -p "${OUT_ROOT}"

cd "${CODE_DIR}"

echo "============================================================"
echo "RUNNING full classification sweep with QC + CORAL"
echo "  box12_data   = ${BOX12_DATA_DIR}"
echo "  box3_data    = ${BOX3_DATA_DIR}"
echo "  box12_meta   = ${BOX12_META_PATH}"
echo "  box3_meta    = ${BOX3_META_PATH}"
echo "  qc_metrics   = ${QC_METRICS_PATH}"
echo "  out_root     = ${OUT_ROOT}"
echo "============================================================"
python "${SCRIPT_NAME}" \
  --train-data "${BOX12_DATA_DIR}" \
  --test-data "${BOX3_DATA_DIR}" \
  --train-meta "${BOX12_META_PATH}" \
  --test-meta "${BOX3_META_PATH}" \
  --qc-metrics "${QC_METRICS_PATH}" \
  --qc-box-train "Box1-2" \
  --qc-box-test "Box3" \
  --output "${OUT_ROOT}" \
  --scenarios Box12_LOSO_PATIENT_AVG Box12_LOSO_SPECTRUM_LEVEL Train12_Predict3_PATIENT_AVG Train12_Predict3_SPECTRUM_LEVEL Box123_LOSO_PATIENT_AVG Box123_LOSO_SPECTRUM_LEVEL \
  --variants baseline qc_only coral_only qc_coral \
  --preproc RAW SNV NORM EMSC SNV+2DER BASELINE+SNV BASELINE+NORM BASELINE+EMSC BASELINE+SNV+2DER \
  --models svm plsda \
  --max-workers 8 \
  --force

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Results root: ${OUT_ROOT}"
