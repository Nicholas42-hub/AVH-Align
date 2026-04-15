#!/bin/bash
#SBATCH --job-name=prep_favc7030
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/prep_favc7030_%A_%a.out
#SBATCH --error=logs/prep_favc7030_%A_%a.err

set -euo pipefail

# ============================================================
#  Preprocess FakeAVCeleb public 70/30 test split.
#  The split CSV is canonicalized to match local directory names.
# ============================================================

CATEGORIES=(
  RealVideo-RealAudio
  RealVideo-FakeAudio
  FakeVideo-RealAudio
  FakeVideo-FakeAudio
)

CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVHUB_DIR="$BASE/av_hubert/avhubert"
RAW_ROOT=/data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2
METADATA="$BASE/av1m_metadata/favc_7030_test_metadata.csv"
SAVE_PATH="$BASE/data/favc_preprocessed"

echo "============================================================"
echo "  FakeAVCeleb 70/30 preprocessing"
echo "  Category : $CATEGORY"
echo "  Job      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node     : $SLURM_NODELIST"
echo "  Start    : $(date)"
echo "============================================================"

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$AVHUB_DIR"
export PYTHONUNBUFFERED=1
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$SAVE_PATH" "$BASE/logs"

echo ">>> Raw root : $RAW_ROOT"
echo ">>> Metadata : $METADATA"
echo ">>> Output   : $SAVE_PATH"
echo ">>> Workers  : $SLURM_CPUS_PER_TASK"
echo ""

# Must run from av_hubert/avhubert so the preparation module is importable.
cd "$AVHUB_DIR"

$PYTHON "$BASE/deepfake_preprocess.py" \
  --dataset FakeAVCeleb \
  --metadata "$METADATA" \
  --data_path "$RAW_ROOT" \
  --save_path "$SAVE_PATH" \
  --category "$CATEGORY" \
  --max_workers "$SLURM_CPUS_PER_TASK"

echo ""
echo "============================================================"
echo "  Done: $CATEGORY at $(date)"
echo "  ROI count in category: $(find "$SAVE_PATH/$CATEGORY" -name '*_roi.mp4' 2>/dev/null | wc -l)"
echo "============================================================"
