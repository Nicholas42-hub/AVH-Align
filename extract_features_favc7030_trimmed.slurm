#!/bin/bash
#SBATCH --job-name=feat_favc7030_trim
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/feat_favc7030_trim_%A_%a.out
#SBATCH --error=logs/feat_favc7030_trim_%A_%a.err

set -euo pipefail

# ============================================================
#  Extract AV-HuBERT NPZ features for FakeAVCeleb public 70/30
#  test split WITH leading-silence trimming (--trimmed flag).
#  Existing NPZ files are skipped by the extractor.
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
METADATA="$BASE/av1m_metadata/favc_7030_test_metadata.csv"
PREP_PATH="$BASE/data/favc_preprocessed"
SAVE_PATH="$BASE/data/favc_features_trimmed"
CKPT_PATH="$AVHUB_DIR/self_large_vox_433h.pt"

echo "============================================================"
echo "  FakeAVCeleb 70/30 AV-HuBERT feature extraction (TRIMMED)"
echo "  Category : $CATEGORY"
echo "  Job      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node     : $SLURM_NODELIST"
echo "  Start    : $(date)"
echo "============================================================"

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$SAVE_PATH" "$BASE/logs"

echo ">>> Preprocessed : $PREP_PATH"
echo ">>> Metadata     : $METADATA"
echo ">>> Output NPZ   : $SAVE_PATH"
echo ">>> AV-HuBERT    : $CKPT_PATH"
echo ">>> Trimmed      : yes"
echo ""

$PYTHON "$BASE/deepfake_feature_extraction.py" \
  --dataset FakeAVCeleb \
  --metadata "$METADATA" \
  --ckpt_path "$CKPT_PATH" \
  --data_path "$PREP_PATH" \
  --save_path "$SAVE_PATH" \
  --category "$CATEGORY" \
  --trimmed

echo ""
echo "============================================================"
echo "  Done: $CATEGORY at $(date)"
echo "  NPZ count in category: $(find "$SAVE_PATH/$CATEGORY" -name '*.npz' 2>/dev/null | wc -l)"
echo "============================================================"
