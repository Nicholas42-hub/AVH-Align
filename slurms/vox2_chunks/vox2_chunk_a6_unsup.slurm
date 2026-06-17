#!/bin/bash
#SBATCH --job-name=vox2_a6_unsup
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vox2_a6_unsup_%j.out
#SBATCH --error=logs/vox2_a6_unsup_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
ZIP=${ZIP:-/data/scratch/projects/punim2637/VoxCeleb2/raw/Vox2-mp4/vox2_mp4_dev.zip}
SCRATCH_ROOT=${SCRATCH_ROOT:-/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks}

# Override these with sbatch --export=ALL,START=1,COUNT=200,...
START=${START:-1}
COUNT=${COUNT:-200}
SEED=${SEED:-44}
EPOCHS=${EPOCHS:-20}
FRAMES_PER_CLIP=${FRAMES_PER_CLIP:-4}
RUN_NAME=${RUN_NAME:-start${START}_n${COUNT}_seed${SEED}}
CLEANUP_RAW=${CLEANUP_RAW:-0}

WORK="$SCRATCH_ROOT/$RUN_NAME"
RAW_ROOT="$WORK/raw"
PREP_ROOT="$WORK/preprocessed"
FEAT_ROOT="$WORK/features"
CSV_ROOT="$WORK/csv"
ZIP_LIST="$WORK/vox2_zip_entries.txt"
SELECTED_LIST="$WORK/selected_files.txt"
METADATA="$WORK/metadata.csv"
OUT="$BASE/avh_sup/outputs_a6_unsup_vox2_${RUN_NAME}"

echo "============================================================"
echo "  Vox2 chunk -> AV-HuBERT features -> A6 unsup sync"
echo "  Job        : ${SLURM_JOB_ID:-local}"
echo "  Node       : ${SLURM_NODELIST:-local}"
echo "  START      : $START"
echo "  COUNT      : $COUNT"
echo "  SEED       : $SEED"
echo "  EPOCHS     : $EPOCHS"
echo "  WORK       : $WORK"
echo "  OUT        : $OUT"
echo "  Start time : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$WORK" "$RAW_ROOT/train" "$PREP_ROOT" "$FEAT_ROOT" "$CSV_ROOT" "$OUT"

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
echo ">>> Building chunk manifest"
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
echo ">>> Extracting selected Vox2 mp4 files"
7za x "$ZIP" -o"$RAW_ROOT/train" -i@"$SELECTED_LIST" -y
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
echo "WAV count: $(find "$PREP_ROOT/train" -name '*.wav' 2>/dev/null | wc -l)"

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
echo "NPZ count: $(find "$FEAT_ROOT/train" -name '*.npz' 2>/dev/null | wc -l)"

echo ""
echo ">>> Training A6 unsup sync on Vox2 chunk features"
cd "$BASE"
"$PYTHON" avh_sup/train_eval_triroute_unsup_sync.py \
  --output_dir "$OUT" \
  --train_features "$FEAT_ROOT" \
  --train_csv_root "$CSV_ROOT" \
  --av1m_features "$BASE/data/avh_features" \
  --av1m_csv_root "$BASE/avh_sup/csv_metadata/av1m" \
  --favc_features "$BASE/data/favc_features" \
  --favc_csv_root "$BASE/avh_sup/csv_metadata/favc_7030_canonical" \
  --epochs "$EPOCHS" \
  --frames_per_clip "$FRAMES_PER_CLIP" \
  --batch_size 64 \
  --num_workers 4 \
  --tau 15 \
  --lr 1.0e-4 \
  --seed "$SEED"

if [[ "$CLEANUP_RAW" == "1" ]]; then
  echo ""
  echo ">>> Cleaning raw/preprocessed intermediates"
  rm -rf "$RAW_ROOT" "$PREP_ROOT"
fi

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OUT"
echo "============================================================"
