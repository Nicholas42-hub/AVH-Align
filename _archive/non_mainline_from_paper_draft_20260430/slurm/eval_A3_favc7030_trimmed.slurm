#!/bin/bash
#SBATCH --job-name=eval_A3_7030_trim
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:24113664
#SBATCH --output=logs/eval_A3_7030_trim_%j.out
#SBATCH --error=logs/eval_A3_7030_trim_%j.err

set -euo pipefail

# ============================================================
#  Evaluate A3 (Two-way + GRL, single seed) on TRIMMED
#  FakeAVCeleb 70/30 public test split.
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A3"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_trimmed"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features_trimmed"
CKPT="$OUTPUT_DIR/ckpts/model-epoch=10.ckpt"

echo "============================================================"
echo "  TRIMMED 70/30 eval | A3 (single seed, epoch=10)"
echo "  Results : $RESULTS_DIR"
echo "  Start   : $(date)"
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

mkdir -p "$RESULTS_DIR" logs

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: Checkpoint not found: $CKPT" >&2
  exit 1
fi
echo "  Ckpt: $CKPT"

cd avh_sup
$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model ablation \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: A3 trimmed 70/30 at $(date)"
echo "============================================================"
