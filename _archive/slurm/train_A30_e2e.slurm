#!/bin/bash
#SBATCH --job-name=train_A30_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A30_e2e_%j.out
#SBATCH --error=logs/train_A30_e2e_%j.err

# ============================================================
#  A30_e2e — Whitened mean alignment (Cohen's d² loss)
#
#  A29 stalled: ep1→ep2 only 1.2× gap reduction because
#  MSE gradient ∝ gap (self-diminishing). Within-class std
#  collapsed to 0.0019, effect size still 9.5σ at ep2.
#
#  A30 fix: normalize mean diff by within-class std (detached):
#    loss = mean(((mu0-mu1)/std_within.detach())^2) = Cohen's d²
#  Gradient ∝ effect size (scale-invariant). Starts at d²≈90,
#  should converge to d≈0 in 1-2 epochs.
#  lambda=1.0 (loss already ~90, no scaling needed).
# ============================================================

echo "============================================================"
echo "  A30_e2e — Whitened mean alignment (Cohen's d²)"
echo "  lambda_mean_align=1.0 (d² loss, starts ~90)"
echo "  Finetune from A9_e2e"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date +"%a %d %b %Y %T %Z")"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

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

mkdir -p avh_sup/outputs_A30_e2e/ckpts avh_sup/outputs_A30_e2e/results avh_sup/outputs_A30_e2e/logs logs

cd avh_sup

echo ">>> Starting A30_e2e training (whitened mean alignment)..."
$PYTHON train_e2e_a30.py --config_path configs/A30_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
