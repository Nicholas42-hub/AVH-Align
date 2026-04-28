#!/bin/bash
#SBATCH --job-name=baselines_50ep
#SBATCH --partition=gpu-l40s
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/baselines_50ep_%A_%a.out
#SBATCH --error=logs/baselines_50ep_%A_%a.err

# ============================================================
#  Baseline retrain — 50 epochs, seed 44, patience 50
#  Array index → model:
#    0  A2   Two-way baseline
#    1  DANN Two-way + DANN/GRL  (UDA)
#    2  CORAL Two-way + CORAL    (UDA)
#    3  A5   Three-way, no push-pull
#    4  A3   Three-way UDA       (real FAVC domain signal)
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
SEED=44

MODELS=(A2 DANN CORAL A5 A3)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  Baseline 50ep seed${SEED} | model=${MODEL}"
echo "  Array task : $SLURM_ARRAY_TASK_ID"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  Start      : $(date)"
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

OUT="$BASE/avh_sup/outputs_${MODEL}_50ep_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results} logs

cd avh_sup

case "$MODEL" in
  A2)
    "$PYTHON" train_test_causal_ablation.py \
      --config_path "configs/A2_50ep_seed44.yaml" \
      --mode train
    ;;
  A3)
    "$PYTHON" train_test_causal_ablation.py \
      --config_path "configs/A3_50ep_seed44.yaml" \
      --mode train
    ;;
  A5)
    "$PYTHON" train_test_fcd.py \
      --config_path "configs/A5_50ep_seed44.yaml" \
      --mode train
    ;;
  DANN)
    "$PYTHON" train_da_baselines.py \
      --method       dann \
      --config_path  "configs/DANN_50ep_seed44.yaml" \
      --seed         $SEED \
      --output_dir   "$OUT"
    ;;
  CORAL)
    "$PYTHON" train_da_baselines.py \
      --method       coral \
      --config_path  "configs/CORAL_50ep_seed44.yaml" \
      --seed         $SEED \
      --output_dir   "$OUT"
    ;;
esac

echo ""
echo "============================================================"
echo "  Done: ${MODEL} 50ep seed${SEED} at $(date)"
echo "============================================================"
