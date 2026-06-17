#!/bin/bash
#SBATCH --job-name=pdv2_s45_fresh
#SBATCH --partition=gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/retrain_A6_pdv2_50ep_seed45_fresh_%j.out
#SBATCH --error=logs/retrain_A6_pdv2_50ep_seed45_fresh_%j.err

# ============================================================
#  Retrain A6-AV1M-PseudoDomain-v2 50ep — seed 45 (FRESH)
#
#  Backs up existing checkpoints so training starts from epoch 0.
#  Uses train_test_fcd_a6_av1m_pseudodomain_v2.py which enables:
#    1. last.ckpt           — final epoch (Lightning default)
#    2. model-epoch=XX.ckpt — best val_auc_causal (source val)
#    3. best-favc-fvra-*    — best FV-RA AUC (checked every 5 epochs)
# ============================================================

SEED=45
BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}/ckpts"

echo "============================================================"
echo "  A6-pdv2-50ep FRESH retrain | seed=${SEED}"
echo "  Start: $(date)"
echo "============================================================"

cd "$BASE"

# Back up existing epoch checkpoints only if old model-epoch=*.ckpt files are present
if ls "$CKPT_DIR"/model-epoch=*.ckpt 2>/dev/null | grep -q .; then
    BAK="${CKPT_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Old epoch ckpts found — backing up: $CKPT_DIR -> $BAK"
    mv "$CKPT_DIR" "$BAK"
else
    echo "No old epoch ckpts found — starting fresh or resuming from last.ckpt"
fi
mkdir -p "$CKPT_DIR"
mkdir -p "avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"/{logs,results} logs

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

cd avh_sup

$PYTHON train_test_fcd_a6_av1m_pseudodomain_v2.py \
    --config_path configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml

echo ""
echo "============================================================"
echo "  Done: seed=${SEED} at $(date)"
echo "============================================================"
