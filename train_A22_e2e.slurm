#!/bin/bash
#SBATCH --job-name=train_A22_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A22_e2e_%j.out
#SBATCH --error=logs/train_A22_e2e_%j.err

# ============================================================
#  A22_e2e — BN on BOTH enc_feats [2048] AND z_sync_pool [512]
#             + Strong Dual DANN (lambda_enc=15, lambda_zsync=15)
#
#  Hypothesis: if domain signal in enc_feats propagates through the
#  SAD projection into z_sync, BN at enc level breaks that path.
#  BN at z_sync level is a final catch-all.
#  lambda=15 provides enough gradient to move the 325M-param encoder.
# ============================================================

echo "============================================================"
echo "  A22_e2e — BN-enc + BN-zsync + Dual DANN lambda=15"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
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

mkdir -p avh_sup/outputs_A22_e2e/ckpts avh_sup/outputs_A22_e2e/results avh_sup/outputs_A22_e2e/logs logs

cd avh_sup

echo ">>> Starting A22_e2e training (BN-enc + BN-zsync + dual DANN lambda=15)..."

$PYTHON train_e2e_a22.py --config_path configs/A22_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
