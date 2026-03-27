#!/bin/bash
#SBATCH --job-name=train_A2_mine
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A2_mine_%j.out
#SBATCH --error=logs/train_A2_mine_%j.err

# ============================================================
#  Ablation A2_mine — Adversarial + Stop Gradient + MINE
#
#  Same as A2 but replaces per-sample orth loss with MINE:
#    MINE directly minimises MI(Z_c; Z_s) across the batch,
#    giving a stronger statistical independence guarantee than
#    per-sample dot-product orthogonality.
#
#  Component 1: GRL adversarial loss                          ✓
#  Component 2: Stop gradient / Z_c-only classification       ✓
#  Component 4: MINE independence loss (replaces orth)        ✓
#
#  Alternating optimisation per step:
#    mine_net: Adam lr=1e-4  (maximise MI estimate)
#    encoder : Adam lr=1e-3  (minimise MI estimate)
# ============================================================

echo "============================================================"
echo "  AVH-Align Ablation A2_mine — Adversarial + StopGrad + MINE"
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

mkdir -p avh_sup/outputs_A2_mine/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_ablation.py \
    --config_path configs/A2_mine.yaml \
    --mode train

echo ""
echo "============================================================"
echo "  A2_mine training done: $(date)"
echo "============================================================"
