#!/bin/bash
#SBATCH --job-name=train_A16_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A16_e2e_%j.out
#SBATCH --error=logs/train_A16_e2e_%j.err

# ============================================================
#  A16_e2e — SAD + LayerNorm + CORAL + Balanced Batching
#  Key change vs A10: CORAL applied directly on AV-HuBERT output
# ============================================================

echo "============================================================"
echo "  A16_e2e — SAD + LayerNorm on Encoder Output + CORAL, Fully E2E"
echo "  LayerNorm on encoder output + CORAL (belt+suspenders)"
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

mkdir -p avh_sup/outputs_A16_e2e/ckpts avh_sup/outputs_A16_e2e/results avh_sup/outputs_A16_e2e/logs logs

cd avh_sup

echo ">>> Starting A16_e2e training (LayerNorm + CORAL + BalancedDomainSampler)..."

$PYTHON train_e2e_a16.py --config_path configs/A16_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
