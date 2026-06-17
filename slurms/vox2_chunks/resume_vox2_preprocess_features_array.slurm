#!/bin/bash
#SBATCH --job-name=vox2_feat_arr
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/vox2_feat_arr_%A_%a.out
#SBATCH --error=logs/vox2_feat_arr_%A_%a.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
WORK=/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40000_seed44_e20

SHARDS=${SHARDS:-4}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

METADATA="$WORK/metadata.csv"
SHARD_DIR="$WORK/shards"
SHARD_METADATA="$SHARD_DIR/metadata_shard_${TASK_ID}_of_${SHARDS}.csv"
RAW_ROOT="$WORK/raw"
PREP_ROOT="$WORK/preprocessed"
FEAT_ROOT="$WORK/features"

echo "============================================================"
echo "  Vox2 n40000 - sharded preprocess + feature extraction"
echo "  Job      : ${SLURM_ARRAY_JOB_ID:-local}.${TASK_ID}"
echo "  Node     : ${SLURM_NODELIST:-local}"
echo "  Shard    : ${TASK_ID}/${SHARDS}"
echo "  Work     : $WORK"
echo "  Start    : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p "$SHARD_DIR" "$PREP_ROOT" "$FEAT_ROOT" logs

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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo ""
echo ">>> Building metadata shard"
"$PYTHON" "$BASE/make_metadata_shard.py" \
  --input "$METADATA" \
  --output "$SHARD_METADATA" \
  --shard-id "$TASK_ID" \
  --num-shards "$SHARDS"

echo ""
echo ">>> Preprocessing shard (already done globally: $(find "$PREP_ROOT/train" -name '*_roi.mp4' 2>/dev/null | wc -l))"
AVHUB_DIR="$BASE/av_hubert/avhubert"
export PYTHONPATH="$AVHUB_DIR"
cd "$AVHUB_DIR"
"$PYTHON" "$BASE/deepfake_preprocess.py" \
  --dataset AV1M \
  --split train \
  --metadata "$SHARD_METADATA" \
  --data_path "$RAW_ROOT" \
  --save_path "$PREP_ROOT" \
  --max_workers 8

echo "ROI count after shard: $(find "$PREP_ROOT/train" -name '*_roi.mp4' 2>/dev/null | wc -l)"
echo "WAV count after shard: $(find "$PREP_ROOT/train" -name '*.wav' 2>/dev/null | wc -l)"

echo ""
echo ">>> Extracting AV-HuBERT features for shard (already done globally: $(find "$FEAT_ROOT" -name '*.npz' 2>/dev/null | wc -l))"
cd "$BASE"
"$PYTHON" "$BASE/deepfake_feature_extraction.py" \
  --dataset AV1M \
  --split train \
  --metadata "$SHARD_METADATA" \
  --ckpt_path "$BASE/av_hubert/avhubert/self_large_vox_433h.pt" \
  --data_path "$PREP_ROOT" \
  --save_path "$FEAT_ROOT"

echo "NPZ count after shard: $(find "$FEAT_ROOT/train" -name '*.npz' 2>/dev/null | wc -l)"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Features: $FEAT_ROOT"
echo "============================================================"
