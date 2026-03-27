#!/bin/bash
#SBATCH --job-name=train_A7
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A7_%j.out
#SBATCH --error=logs/train_A7_%j.err

# ============================================================
#  A7 — FCD + Domain Adversarial + Label Adversarial on Z_s
#
#  Key difference from A6:
#    + L_ladv: GRL(Z_s) → label_head_s → BCE(fake/real)
#      Forces Z_s to NOT encode fake/real label information.
#      Fixes Z_s > Z_c label-predictability inversion from A6.
#    + lambda_dadv increased 1.0 → 3.0 for stronger domain
#      disentanglement.
#  Wall time: 24h (A5/A6 were killed at epochs 41/22 at 12h)
# ============================================================

echo "============================================================"
echo "  A7 Training — FCD + Domain Adv + Label Adv on Z_s"
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

mkdir -p avh_sup/outputs_A7/{logs,ckpts,results} logs

cd avh_sup

$PYTHON train_test_fcd_a7.py --config_path configs/A7.yaml --mode train

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A7/ckpts/
echo "============================================================"
