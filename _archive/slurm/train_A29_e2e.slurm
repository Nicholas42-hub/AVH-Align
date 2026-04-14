#!/bin/bash
#SBATCH --job-name=train_A29_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A29_e2e_%j.out
#SBATCH --error=logs/train_A29_e2e_%j.err

# ============================================================
#  A29_e2e — A28 + kill noisy GRL + boost mean_align 50x
#
#  DIAGNOSIS (A28 metrics.csv):
#    loss_mean_align: ep0=2.76e-3, ep1=6.07e-5, ep2-6=~4e-5 STUCK
#    loss_zsync_dann: 0.6932 every epoch = ln(2) = random binary CE
#    → DANN on CENTERED z_sync is at chance (CENTERED AUC=0.5 always)
#    → generates pure noise gradients via GRL (lambda=2 → 1.39 total)
#    → these noisy gradients oscillate and neutralize mean_align progress
#
#  FIX:
#    lambda_zsync_dann = 0   (kill noise source)
#    lambda_mean_align = 500 (50× boost: ep0→1.4, ep1+→0.02 self-regulating)
#
#  Expected: mean_align converges to ~0 without oscillation → z_sync RAW AUC drops
# ============================================================

echo "============================================================"
echo "  A29_e2e — lambda_mean_align=500, lambda_zsync_dann=0"
echo "  Finetune from A9_e2e; kill noisy centered-DANN; boost mean-align"
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

mkdir -p avh_sup/outputs_A29_e2e/ckpts avh_sup/outputs_A29_e2e/results avh_sup/outputs_A29_e2e/logs logs

cd avh_sup

echo ">>> Starting A29_e2e training..."
$PYTHON train_e2e_a29.py --config_path configs/A29_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
