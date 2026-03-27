#!/bin/bash
#SBATCH --job-name=train_A28_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A28_e2e_%j.out
#SBATCH --error=logs/train_A28_e2e_%j.err

# ============================================================
#  A28_e2e — A26 with lr_encoder raised to 5e-6 (25x)
#
#  A26 ep0: task AUC=0.9999 ✅ but z_sync RAW stayed at 0.9988
#  Root cause: lr_encoder=2e-7 too low — encoder barely moves,
#  domain means don't shift per epoch.
#
#  A28 fix: lr_encoder=5e-6 (same as A9_e2e training).
#  Mean-align gradient is gentle (non-adversarial), task
#  gradient is near-zero (confident at start) → encoder update
#  dominated by domain mean alignment → means converge fast.
# ============================================================

echo "============================================================"
echo "  A28_e2e — Mean-align lambda=10, lr_enc=5e-6 (A26 x25)"
echo "  Finetune from A9_e2e + direct MSE(mu_0, mu_1)"
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

mkdir -p avh_sup/outputs_A28_e2e/ckpts avh_sup/outputs_A28_e2e/results avh_sup/outputs_A28_e2e/logs logs

cd avh_sup

echo ">>> Starting A28_e2e training (mean-align from A9_e2e, lr_enc=5e-6)..."
$PYTHON train_e2e_a28.py --config_path configs/A28_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
