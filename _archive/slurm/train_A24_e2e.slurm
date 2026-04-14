#!/bin/bash
#SBATCH --job-name=train_A24_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A24_e2e_%j.out
#SBATCH --error=logs/train_A24_e2e_%j.err

# ============================================================
#  A24_e2e — CDAN (Conditional Domain-Adversarial Network)
#             at z_sync_pool level + enc DANN (lambda=10)
#
#  Vanilla DANN converges to Nash where domain signal survives in
#  classifier-blind subspace. CDAN (Long et al. 2018) conditions
#  the domain classifier on task predictions: domain clf input =
#  z_sync_pool * sigmoid(task_score). This forces domain alignment
#  in the task-relevant feature dimensions specifically.
# ============================================================

echo "============================================================"
echo "  A24_e2e — CDAN(z_sync, task_prob) + enc DANN lambda=10"
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

mkdir -p avh_sup/outputs_A24_e2e/ckpts avh_sup/outputs_A24_e2e/results avh_sup/outputs_A24_e2e/logs logs

cd avh_sup

echo ">>> Starting A24_e2e training (CDAN + enc DANN lambda=10)..."

$PYTHON train_e2e_a24.py --config_path configs/A24_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
