#!/bin/bash
#SBATCH --job-name=train_A9
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A9_%j.out
#SBATCH --error=logs/train_A9_%j.err

# ============================================================
#  A9 — SAD + Multi-Kernel MMD Domain Alignment
#
#  Diagnosis from A1–A8 domain probes:
#    Raw AV-HuBERT features: domain AUC = 0.999 (extremely strong signal)
#    GRL-based adversarial (all A3/A6/A7/A8 variants): Z_c AUC = 0.985–0.999
#    → GRL saddle-points when domain signal is overwhelmingly strong.
#
#  Fix: Replace L_dadv [GRL BCE] with L_mmd [multi-kernel RBF MMD²]:
#    MMD²(Z_sync|AV1M, Z_sync|FAVC)  — no adversarial game, always
#    informative gradient, provably bounds domain generalisation error.
#
#  Key changes from A8:
#    • No domain_head_c / no GRL on Z_sync
#    • lambda_mmd = 10.0 (strong — raw signal is 0.999)
#    • batch_size = 16 (needed for per-batch MMD estimation)
#    • FAVC oversampled ×14 for balanced mini-batches
# ============================================================

echo "============================================================"
echo "  A9 Training — SAD + Multi-Kernel MMD Domain Alignment"
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

mkdir -p avh_sup/outputs_A9/{logs,ckpts,results} logs

cd avh_sup

$PYTHON train_test_sad_a9.py --config_path configs/A9.yaml --mode train

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A9/ckpts/
echo "============================================================"
