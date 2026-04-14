#!/bin/bash
#SBATCH --job-name=train_A32_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A32_e2e_%j.out
#SBATCH --error=logs/train_A32_e2e_%j.err

# ============================================================
#  A32_e2e — Null-space gradient projection + L1-d alignment
#
#  Core idea: project z_sync onto null-space of domain direction
#  before task head. Task gradient is orthogonal to domain
#  direction by construction → no conflict possible.
#
#  d_unit = normalize(mu_FAVC - mu_AV1M) [per-epoch, probe batch]
#  z_task = z_sync - (z_sync @ d_unit) * d_unit
#  + L1-d alignment (λ=0.02) to actively close the gap
# ============================================================

echo "============================================================"
echo "  A32_e2e — Null-space projection + L1-d alignment"
echo "  d_unit per epoch from probe batch (200+200)"
echo "  lambda_mean_align=0.02  lambda_ladv=0.0"
echo "  gradient_clip_val=2.0"
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

mkdir -p avh_sup/outputs_A32_e2e/ckpts avh_sup/outputs_A32_e2e/results avh_sup/outputs_A32_e2e/logs logs

cd avh_sup

echo ">>> Starting A32_e2e training (null-space projection)..."
$PYTHON train_e2e_a32.py --config_path configs/A32_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
