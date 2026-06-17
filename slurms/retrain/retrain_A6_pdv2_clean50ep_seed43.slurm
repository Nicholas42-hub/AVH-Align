#!/bin/bash
#SBATCH --job-name=pdv2_clean_s43
#SBATCH --partition=gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/retrain_A6_pdv2_clean50ep_seed43_%j.out
#SBATCH --error=logs/retrain_A6_pdv2_clean50ep_seed43_%j.err

# ============================================================
#  POLICY-CLEAN A6-pdv2 retrain | 50 epochs | dense per-epoch save
#  Output: outputs_A6_pdv2_clean50ep_seed43/
#  No FAVC callback. SWA window will be defined by source val_auc plateau.
# ============================================================

THIS_SEED=43
BASE=/data/projects/punim2637/nnliang/AVH-Align
OUT="$BASE/avh_sup/outputs_A6_pdv2_clean50ep_seed${THIS_SEED}"

echo "============================================================"
echo "  pdv2 CLEAN 50ep | seed=${THIS_SEED} | Start: $(date)"
echo "============================================================"

mkdir -p "$OUT"/{logs,results,ckpts/dense} "$BASE/logs"

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

# Auto-resume if last.ckpt exists (preempt-safe)
RESUME_CKPT=""
if [[ -f "$OUT/ckpts/last.ckpt" ]]; then
    RESUME_CKPT="$OUT/ckpts/last.ckpt"
    echo "Resume from: $RESUME_CKPT"
fi

cd avh_sup

if [[ -n "$RESUME_CKPT" ]]; then
    $PYTHON train_test_fcd_a6_av1m_pseudodomain_v2_clean.py \
        --config_path configs/A6_av1m_pseudodomain_v2_clean50ep_seed${THIS_SEED}.yaml \
        --ckpt_path "$RESUME_CKPT"
else
    $PYTHON train_test_fcd_a6_av1m_pseudodomain_v2_clean.py \
        --config_path configs/A6_av1m_pseudodomain_v2_clean50ep_seed${THIS_SEED}.yaml
fi

echo "============================================================"
echo "  Done: seed=${THIS_SEED} at $(date)"
echo "============================================================"
