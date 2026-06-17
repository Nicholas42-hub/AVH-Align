#!/bin/bash
#SBATCH --job-name=vox2_shard
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/vox2_shard_%A_%a.out
#SBATCH --error=logs/vox2_shard_%A_%a.err

# ============================================================
#  Vox2 sharded extraction: zip → preprocess → feat per shard.
#  Run as array (e.g. --array=0-3%4 for 4 parallel shards).
#
#  Env:
#    SHARDS=4
#    RUN_NAME=vox2_n40k_part2
#    SCRATCH_ROOT=...    (parent dir; default scratch path)
#    METADATA=$WORK/metadata.csv  (must exist before submit; built by
#                                  prepare_vox2_chunk_manifest.py)
# ============================================================

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
ZIP=${ZIP:-/data/scratch/projects/punim2637/VoxCeleb2/raw/Vox2-mp4/vox2_mp4_dev.zip}
SCRATCH_ROOT=${SCRATCH_ROOT:-/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks}
RUN_NAME=${RUN_NAME:-vox2_chunk}
SHARDS=${SHARDS:-4}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

WORK="$SCRATCH_ROOT/$RUN_NAME"
RAW_ROOT="$WORK/raw"
PREP_ROOT="$WORK/preprocessed"
FEAT_ROOT="$WORK/features"
SHARD_DIR="$WORK/shards"
SHARD_METADATA="$SHARD_DIR/metadata_shard_${TASK_ID}_of_${SHARDS}.csv"
SHARD_LIST="$SHARD_DIR/selected_shard_${TASK_ID}_of_${SHARDS}.txt"
GLOBAL_METADATA="$WORK/metadata.csv"

echo "============================================================"
echo "  Vox2 SHARDED extraction"
echo "  Job        : ${SLURM_ARRAY_JOB_ID:-local}.${TASK_ID}"
echo "  Node       : ${SLURM_NODELIST:-local}"
echo "  RUN_NAME   : $RUN_NAME"
echo "  Shard      : ${TASK_ID}/${SHARDS}"
echo "  WORK       : $WORK"
echo "  Start      : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$RAW_ROOT/train" "$PREP_ROOT" "$FEAT_ROOT" "$SHARD_DIR"

if [[ ! -f "$GLOBAL_METADATA" ]]; then
  echo "ERROR: $GLOBAL_METADATA not found. Build it first with prepare_vox2_chunk_manifest.py"
  exit 1
fi

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
echo ">>> Building shard ${TASK_ID}/${SHARDS} metadata"
"$PYTHON" "$BASE/make_metadata_shard.py" \
  --input "$GLOBAL_METADATA" \
  --output "$SHARD_METADATA" \
  --shard-id "$TASK_ID" \
  --num-shards "$SHARDS"

# extract path column → shard_list.txt for 7za
awk -F, 'NR>1 {print $1}' "$SHARD_METADATA" > "$SHARD_LIST"
echo "Shard rows: $(wc -l < "$SHARD_LIST")"

echo ""
echo ">>> Unzipping shard $TASK_ID files from zip"
7za x "$ZIP" -o"$RAW_ROOT/train" -i@"$SHARD_LIST" -y > /dev/null
echo "Raw mp4 in shard: $(find "$RAW_ROOT/train" -name '*.mp4' | wc -l)"

echo ""
echo ">>> Preprocessing shard $TASK_ID (mouth ROI + wav)"
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

echo ""
echo ">>> Extracting AV-HuBERT features for shard $TASK_ID"
cd "$BASE"
"$PYTHON" "$BASE/deepfake_feature_extraction.py" \
  --dataset AV1M \
  --split train \
  --metadata "$SHARD_METADATA" \
  --ckpt_path "$BASE/av_hubert/avhubert/self_large_vox_433h.pt" \
  --data_path "$PREP_ROOT" \
  --save_path "$FEAT_ROOT"

echo ""
echo ">>> Cleaning shard's raw + preprocessed (keep features only)"
# Delete only files listed in this shard's selected_list (per-shard cleanup)
while read -r p; do
  rm -f "$RAW_ROOT/train/$p" 2>/dev/null
  base="${p%.mp4}"
  rm -f "$PREP_ROOT/train/${base}_roi.mp4" "$PREP_ROOT/train/${base}.wav" 2>/dev/null
done < "$SHARD_LIST"

NPZ_TOTAL=$(find "$FEAT_ROOT/train" -name '*.npz' 2>/dev/null | wc -l)
echo ""
echo "============================================================"
echo "  Done shard ${TASK_ID}/${SHARDS} at $(date)"
echo "  Total NPZ in $FEAT_ROOT now: $NPZ_TOTAL"
echo "============================================================"
