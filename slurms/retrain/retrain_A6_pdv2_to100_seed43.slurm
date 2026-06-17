#!/bin/bash
#SBATCH --job-name=pdv2_s43_to100
#SBATCH --partition=gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/retrain_A6_pdv2_to100_seed43_%j.out
#SBATCH --error=logs/retrain_A6_pdv2_to100_seed43_%j.err

# ============================================================
#  Resume A6-AV1M-PseudoDomain-v2 -> 100ep | seed 43
#
#  Auto-discovers latest resume checkpoint:
#    1. ckpts/last.ckpt                            (cleanest)
#    2. ckpts/favc_fvra/best-favc-fvra-epoch=*.ckpt (FV-RA spike state)
#    3. ckpts/model-epoch=*.ckpt                    (best source-val)
#
#  If none found, starts fresh.
# ============================================================

SEED=43
BASE=/data/projects/punim2637/nnliang/AVH-Align
OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
CKPT_DIR="$OUT/ckpts"

echo "============================================================"
echo "  A6-pdv2 RESUME -> 100ep | seed=${SEED}"
echo "  Start: $(date)"
echo "============================================================"

# Discover resume checkpoint
RESUME_CKPT=""
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
    RESUME_CKPT="$CKPT_DIR/last.ckpt"
    echo "Resume from last.ckpt: $RESUME_CKPT"
elif ls "$CKPT_DIR"/favc_fvra/best-favc-fvra-epoch=*.ckpt 2>/dev/null | grep -q .; then
    RESUME_CKPT=$(ls -1 "$CKPT_DIR"/favc_fvra/best-favc-fvra-epoch=*.ckpt | sort | tail -n1)
    echo "Resume from latest FV-RA callback ckpt: $RESUME_CKPT"
elif ls "$CKPT_DIR"/model-epoch=*.ckpt 2>/dev/null | grep -q .; then
    RESUME_CKPT=$(ls -1 "$CKPT_DIR"/model-epoch=*.ckpt | sort | tail -n1)
    echo "Resume from best source-val ckpt: $RESUME_CKPT"
else
    echo "No checkpoint found — starting fresh from epoch 0"
fi

mkdir -p "$OUT"/{logs,results} "$BASE/logs"

cd "$BASE"

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

if [[ -n "$RESUME_CKPT" ]]; then
    $PYTHON train_test_fcd_a6_av1m_pseudodomain_v2.py \
        --config_path configs/A6_av1m_pseudodomain_v2_to100_seed${SEED}.yaml \
        --ckpt_path "$RESUME_CKPT"
else
    $PYTHON train_test_fcd_a6_av1m_pseudodomain_v2.py \
        --config_path configs/A6_av1m_pseudodomain_v2_to100_seed${SEED}.yaml
fi

echo ""
echo "============================================================"
echo "  Done: seed=${SEED} at $(date)"
echo "============================================================"
