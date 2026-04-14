#!/bin/bash
#SBATCH --job-name=train_A27_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A27_e2e_%j.out
#SBATCH --error=logs/train_A27_e2e_%j.err

# ============================================================
#  A27_e2e — CDAN + fine-tune from A9_e2e (task AUC=1.0)
#
#  Combines the two best signals from A20-A26:
#    A24: CDAN was the only genuine adversarial movement (0.9187)
#    A26: fine-tune from A9_e2e protects task AUC (already 1.0)
#
#  With task AUC=1.0 at start, CDAN's task_prob is meaningful
#  from epoch 0 → CDAN targets exactly the task-relevant
#  directions of z_sync where domain information is harmful.
#
#  No enc_dann, lambda_cdan=5, lr_encoder=5e-7.
# ============================================================

echo "============================================================"
echo "  A27_e2e — CDAN + finetune from A9_e2e (task AUC=1.0)"
echo "  lambda_cdan=5, lambda_ddis=1.0, lr_enc=5e-7, no enc_dann"
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

mkdir -p avh_sup/outputs_A27_e2e/ckpts avh_sup/outputs_A27_e2e/results avh_sup/outputs_A27_e2e/logs logs

cd avh_sup

echo ">>> Starting A27_e2e training (CDAN + finetune from A9_e2e)..."
$PYTHON train_e2e_a27.py --config_path configs/A27_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
