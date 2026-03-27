#!/bin/bash
#SBATCH --job-name=train_A23_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A23_e2e_%j.out
#SBATCH --error=logs/train_A23_e2e_%j.err

# ============================================================
#  A23_e2e — RFF-MMD directly on z_sync_pool [512]
#             + Enc-level DANN (lambda=10)
#
#  Key insight: A15 applies RFF-MMD on enc_feats [2048] upstream.
#  If the SAD head projection re-introduces domain signal into z_sync,
#  A15 cannot fix it. A23 applies MMD directly on z_sync_pool,
#  targeting the EXACT representation the domain probe measures.
#  RBF kernel = no rank constraints (unlike CORAL which needs n > d=512).
# ============================================================

echo "============================================================"
echo "  A23_e2e — RFF-MMD on z_sync_pool + enc DANN lambda=10"
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

mkdir -p avh_sup/outputs_A23_e2e/ckpts avh_sup/outputs_A23_e2e/results avh_sup/outputs_A23_e2e/logs logs

cd avh_sup

echo ">>> Starting A23_e2e training (RFF-MMD on z_sync + enc DANN lambda=10)..."

$PYTHON train_e2e_a23.py --config_path configs/A23_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
