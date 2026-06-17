#!/bin/bash
#SBATCH --job-name=ts_unsup_s43
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/ts_unsup_s43_trimmed_%j.out
#SBATCH --error=logs/ts_unsup_s43_trimmed_%j.err

# ============================================================
#  Table 8 row: TriRoute unsup (Vox2) — seed43 best ckpt
#  Two passes:
#    1. UNTRIMMED (default features) → results_table8_untrimmed/
#    2. TRIMMED  (leading-silence trimmed features) → results_table8_trimmed/
#  Each pass outputs FAVC 70/30 + AV1M val AUC.
#
#  Untrimmed FAVC eval already exists in results_favc_7030/ but rerun for
#  consistent table8 output dir layout.
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
CKPT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/triroute_unsup_sync_best_auc.pt"

UNTRIMMED_OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/results_table8_untrimmed"
TRIMMED_OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/results_table8_trimmed"

# Untrimmed features (defaults inside the script):
#   FAVC : $BASE/data/favc_features
#   AV1M : $BASE/data/avh_features  (val/ symlink)
# Trimmed features:
FAVC_TRIM="$BASE/data/favc_features_trimmed"
AV1M_TRIM="/scratch/punim2637/nnliang/avd1m_features_trimmed"

echo "============================================================"
echo "  Table 8 unsup eval | seed=43 | $(date)"
echo "  Ckpt: $CKPT"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$UNTRIMMED_OUT" "$TRIMMED_OUT"

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

cd avh_sup

echo ""
echo "============================================================"
echo "  Pass 1/2 : UNTRIMMED  (FAVC original + AV1M original)"
echo "============================================================"
$PYTHON train_eval_triroute_unsup_sync.py \
    --eval_only \
    --ckpt "$CKPT" \
    --output_dir "$UNTRIMMED_OUT" \
    --num_workers 4

echo ""
echo "============================================================"
echo "  Pass 2/2 : TRIMMED  (FAVC trimmed + AV1M trimmed)"
echo "============================================================"
$PYTHON train_eval_triroute_unsup_sync.py \
    --eval_only \
    --ckpt "$CKPT" \
    --output_dir "$TRIMMED_OUT" \
    --favc_features "$FAVC_TRIM" \
    --av1m_features "$AV1M_TRIM" \
    --num_workers 4

echo ""
echo "============================================================"
echo "  Done at $(date)"
echo ""
echo "  Untrimmed:"
echo "    FAVC : $UNTRIMMED_OUT/results_favc_7030/eval_results.txt"
echo "    AV1M : $UNTRIMMED_OUT/results_av1m_val/eval_results.txt"
echo "  Trimmed:"
echo "    FAVC : $TRIMMED_OUT/results_favc_7030/eval_results.txt"
echo "    AV1M : $TRIMMED_OUT/results_av1m_val/eval_results.txt"
echo "============================================================"
