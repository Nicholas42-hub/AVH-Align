#!/bin/bash
#SBATCH --job-name=vox2_resume
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vox2_resume_%j.out
#SBATCH --error=logs/vox2_resume_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
WORK=/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40000_seed44_e20

METADATA="$WORK/metadata.csv"
RAW_ROOT="$WORK/raw"
PREP_ROOT="$WORK/preprocessed"
FEAT_ROOT="$WORK/features"

echo "============================================================"
echo "  Vox2 n40000 – resume preprocess + feature extraction"
echo "  Job   : ${SLURM_JOB_ID:-local}"
echo "  Node  : ${SLURM_NODELIST:-local}"
echo "  Start : $(date)"
echo "============================================================"

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

mkdir -p "$PREP_ROOT" "$FEAT_ROOT" logs

# ── Step 1: Preprocess (skip already-done clips via existing _roi.mp4 check) ──
echo ""
echo ">>> Preprocessing mouth ROI + wav  (already done: $(find "$PREP_ROOT/train" -name '*_roi.mp4' | wc -l))"
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
echo "ROI count after: $(find "$PREP_ROOT/train" -name '*_roi.mp4' | wc -l)"
echo "WAV count after: $(find "$PREP_ROOT/train" -name '*.wav'     | wc -l)"

# ── Step 2: Feature extraction ────────────────────────────────────────────────
echo ""
echo ">>> Extracting AV-HuBERT features  (already done: $(find "$FEAT_ROOT" -name '*.npz' | wc -l))"
cd "$BASE"
"$PYTHON" "$BASE/deepfake_feature_extraction.py" \
  --dataset AV1M \
  --split train \
  --metadata "$METADATA" \
  --ckpt_path "$BASE/av_hubert/avhubert/self_large_vox_433h.pt" \
  --data_path "$PREP_ROOT" \
  --save_path "$FEAT_ROOT"
echo "NPZ count after: $(find "$FEAT_ROOT/train" -name '*.npz' | wc -l)"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Features: $FEAT_ROOT"
echo "============================================================"
