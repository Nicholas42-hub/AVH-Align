#!/bin/bash
#SBATCH --job-name=pdv2_s43
#SBATCH --partition=gpu-l40s-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/retrain_A6_pdv2_50ep_seed43_resume_%j.out
#SBATCH --error=logs/retrain_A6_pdv2_50ep_seed43_resume_%j.err

# ============================================================
#  Retrain A6-AV1M-PseudoDomain-v2 50ep — seed 43 ONLY
#  Resume from last.ckpt (epoch=4), complete to epoch 50.
#  Uses gpu-a100 (non-preempt) with 8h to avoid timeout.
# ============================================================

SEED=43
BASE=/data/projects/punim2637/nnliang/AVH-Align
LAST_CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed43/ckpts/last.ckpt"

echo "============================================================"
echo "  A6-pdv2-50ep resume | seed=${SEED}"
echo "  Resume from: $LAST_CKPT"
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

OUT="avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
mkdir -p "${OUT}"/{logs,ckpts,results} logs

cd avh_sup

$PYTHON train_test_fcd_a6_av1m_pseudodomain.py \
  --config_path configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml

echo ""
echo "============================================================"
echo "  Done: seed=${SEED} at $(date)"
echo "============================================================"
