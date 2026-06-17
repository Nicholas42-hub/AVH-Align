#!/bin/bash
#SBATCH --job-name=vox2_feat_only
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vox2_feat_only_%j.out
#SBATCH --error=logs/vox2_feat_only_%j.err

# ============================================================
#  Vox2 chunk feature extraction ONLY (no training).
#  Designed for incremental data scaling (e.g., 40k+40k+40k = 120k total).
#
#  Usage:
#    sbatch --export=ALL,START=40001,COUNT=40000,RUN_NAME=vox2_n40k_part2 \
#           slurms/vox2_chunks/extract_vox2_features_only.slurm
#
#    For gpfs target:
#    sbatch --export=ALL,START=80001,COUNT=40000,RUN_NAME=vox2_n40k_part3,\
#SCRATCH_ROOT=/data/projects/punim2637/VoxCeleb2/a6_unsup_chunks \
#           slurms/vox2_chunks/extract_vox2_features_only.slurm
#
#  CLEANUP_RAW=1 is the default here (we don't need raw/preprocessed after features).
# ============================================================

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
ZIP=${ZIP:-/data/scratch/projects/punim2637/VoxCeleb2/raw/Vox2-mp4/vox2_mp4_dev.zip}
SCRATCH_ROOT=${SCRATCH_ROOT:-/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks}

START=${START:-1}
COUNT=${COUNT:-40000}
SEED=${SEED:-44}
RUN_NAME=${RUN_NAME:-start${START}_n${COUNT}_seed${SEED}}
CLEANUP_RAW=${CLEANUP_RAW:-1}

WORK="$SCRATCH_ROOT/$RUN_NAME"
RAW_ROOT="$WORK/raw"
PREP_ROOT="$WORK/preprocessed"
FEAT_ROOT="$WORK/features"
CSV_ROOT="$WORK/csv"
ZIP_LIST="$WORK/vox2_zip_entries.txt"
SELECTED_LIST="$WORK/selected_files.txt"
METADATA="$WORK/metadata.csv"

echo "============================================================"
echo "  Vox2 chunk -> AV-HuBERT features (no training)"
echo "  Job        : ${SLURM_JOB_ID:-local}"
echo "  Node       : ${SLURM_NODELIST:-local}"
echo "  START      : $START"
echo "  COUNT      : $COUNT"
echo "  SEED       : $SEED"
echo "  WORK       : $WORK"
echo "  CLEANUP    : $CLEANUP_RAW"
echo "  Start time : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$WORK" "$RAW_ROOT/train" "$PREP_ROOT" "$FEAT_ROOT" "$CSV_ROOT"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0
module load FFmpeg/4.4.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo ""
echo ">>> Building chunk manifest (entries [$START, $((START+COUNT-1))])"
if [[ ! -f "$ZIP_LIST" ]]; then
  zipinfo -1 "$ZIP" > "$ZIP_LIST"
fi
"$PYTHON" "$BASE/prepare_vox2_chunk_manifest.py" \
  --zip-list "$ZIP_LIST" \
  --start "$START" \
  --count "$COUNT" \
  --selected-list "$SELECTED_LIST" \
  --metadata "$METADATA" \
  --train-labels "$CSV_ROOT/train_labels.csv"

echo ""
echo ">>> Extracting selected Vox2 mp4 files from zip"
7za x "$ZIP" -o"$RAW_ROOT/train" -i@"$SELECTED_LIST" -y > /dev/null
echo "Extracted mp4 count: $(find "$RAW_ROOT/train" -name '*.mp4' | wc -l)"

echo ""
echo ">>> Preprocessing mouth ROI + wav"
AVHUB_DIR="$BASE/av_hubert/avhubert"
export PYTHONPATH="$AVHUB_DIR"
cd "$AVHUB_DIR"
"$PYTHON" "$BASE/deepfake_preprocess.py" \
  --dataset AV1M \
  --split train \
  --metadata "$METADATA" \
  --data_path "$RAW_ROOT" \
  --save_path "$PREP_ROOT" \
  --max_workers 8
echo "ROI count: $(find "$PREP_ROOT/train" -name '*_roi.mp4' 2>/dev/null | wc -l)"

echo ""
echo ">>> Extracting AV-HuBERT features"
cd "$BASE"
"$PYTHON" "$BASE/deepfake_feature_extraction.py" \
  --dataset AV1M \
  --split train \
  --metadata "$METADATA" \
  --ckpt_path "$BASE/av_hubert/avhubert/self_large_vox_433h.pt" \
  --data_path "$PREP_ROOT" \
  --save_path "$FEAT_ROOT"
NPZ_COUNT=$(find "$FEAT_ROOT/train" -name '*.npz' 2>/dev/null | wc -l)
echo "NPZ count: $NPZ_COUNT"

if [[ "$CLEANUP_RAW" == "1" ]]; then
  echo ""
  echo ">>> Cleaning raw/preprocessed intermediates"
  rm -rf "$RAW_ROOT" "$PREP_ROOT"
fi

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Features at: $FEAT_ROOT"
echo "  NPZ count:   $NPZ_COUNT (target: $COUNT)"
echo "  Disk usage:  $(du -sh "$WORK" | cut -f1)"
echo "============================================================"
