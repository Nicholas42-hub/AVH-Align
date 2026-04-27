#!/bin/bash
#SBATCH --job-name=A6_stable_train
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/A6_stable_train_%A_%a.out
#SBATCH --error=logs/A6_stable_train_%A_%a.err

# ============================================================
#  Retrain fcd_a6 seeds 44/45 with stability fixes:
#    - GRL alpha warmup (DANN schedule, 0 → 1.0 over epochs)
#    - gradient_clip_val=1.0
#    - lr: 3e-4 (reduced from 1e-3)
#
#  Array index → SEED:
#    0: 44
#    1: 45
# ============================================================

SEEDS=(44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_seed${SEED}"

echo "============================================================"
echo "  Train fcd_a6 stable | seed=${SEED}"
echo "  Config  : avh_sup/configs/A6_stable.yaml"
echo "  Output  : $OUTPUT_DIR"
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

mkdir -p logs
mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results}

cd avh_sup

"$PYTHON" train_test_fcd_a6.py \
  --config_path "configs/A6_stable.yaml" \
  --mode train \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Done: fcd_a6 stable seed=${SEED} at $(date)"
echo "============================================================"
