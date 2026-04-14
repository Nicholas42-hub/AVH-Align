#!/bin/bash
#SBATCH --job-name=train_A26_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A26_e2e_%j.out
#SBATCH --error=logs/train_A26_e2e_%j.err

# ============================================================
#  A26_e2e — Fine-tune from A9_e2e (task AUC=1.0)
#             + Direct Mean Alignment (lambda=10) on z_sync_pool
#             + Light DANN on z_sync (lambda=2)
#
#  Root cause fix for A20-A25 failures:
#    - All task AUCs = 0.50 because lambda_enc_dann=10-15 was
#      10-15x stronger than task gradient → encoder collapsed
#    - A25 z_sync=0.5 was a PROBE ARTIFACT (centering in probe)
#
#  A26 approach:
#    - Start from A9_e2e epoch-8 ckpt (val_auc_causal=1.0000)
#    - Direct MSE(mu_0, mu_1) on z_sync_pool — stable, targeted
#    - NO enc_dann — don't destabilise the already-good encoder
#    - lambda_zsync_dann=2 (very light adversarial for covariance)
#    - lambda_ddis=1.0 (route domain info to z_app)
#    - lr_encoder=2e-7 (protect task from being disrupted)
#    - DUAL PROBE: RAW (true invariance) + CENTERED (residual)
# ============================================================

echo "============================================================"
echo "  A26_e2e — Finetune from A9_e2e + Direct Mean Alignment"
echo "  dual DANN enc+zsync, lambda_ddis=0 (no conflict)"
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

mkdir -p avh_sup/outputs_A26_e2e/ckpts avh_sup/outputs_A26_e2e/results avh_sup/outputs_A26_e2e/logs logs

cd avh_sup

echo ">>> Starting A26_e2e training (finetune A9_e2e + mean alignment lambda=10)..."
$PYTHON train_e2e_a26.py --config_path configs/A26_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
