#!/bin/bash
#SBATCH --job-name=favc_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/favc_eval_%j.out
#SBATCH --error=logs/favc_eval_%j.err

# ============================================================
#  Cross-dataset evaluation: AV1M-trained model → FakeAVCeleb
#
#  Runs eval_favc.py for both the baseline (AVH-Sup) and
#  the causal model (AVH-Causal), evaluating on the FakeAVCeleb
#  test split using per-clip NPZ features.
#
#  Results written to:
#    avh_sup/results/favc_baseline/eval_results.txt
#    avh_sup/results/favc_causal/eval_results.txt
#
#  Usage:
#    sbatch favc_eval.slurm
#  Prerequisite:
#    favc_extract_features.slurm must have completed (NPZs in data/favc_features/)
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Cross-Dataset Evaluation"
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
echo "Python: $(which python) — $(python --version)"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

FEATURES=/data/projects/punim2637/nnliang/AVH-Align/data/favc_features
CSV_DIR=/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc
SPLIT=test

BASELINE_CKPT=/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs/ckpts/model-epoch=05.ckpt
CAUSAL_CKPT=/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_causal/ckpts/model-epoch=09.ckpt

# Sanity: check features exist
NPZ_COUNT=$(find "$FEATURES" -name "*.npz" 2>/dev/null | wc -l)
echo "NPZ feature files in $FEATURES: $NPZ_COUNT"
if [ "$NPZ_COUNT" -eq 0 ]; then
    echo "ERROR: No feature files found. Run favc_extract_features.slurm first."
    exit 1
fi

mkdir -p logs avh_sup/results

cd avh_sup

# ── Baseline evaluation ────────────────────────────────────────
if [ -f "$BASELINE_CKPT" ]; then
    echo ""
    echo "─── Baseline (AVH-Sup) evaluation ───"
    python eval_favc.py \
        --ckpt          "$BASELINE_CKPT" \
        --model         baseline \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_DIR" \
        --split         "$SPLIT" \
        --output_dir    results/favc_baseline
    echo ""
    echo "Baseline results:"
    cat results/favc_baseline/eval_results.txt
else
    echo "WARNING: Baseline checkpoint not found: $BASELINE_CKPT"
fi

# ── Causal evaluation ──────────────────────────────────────────
if [ -f "$CAUSAL_CKPT" ]; then
    echo ""
    echo "─── Causal (AVH-Causal) evaluation ───"
    python eval_favc.py \
        --ckpt          "$CAUSAL_CKPT" \
        --model         causal \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_DIR" \
        --split         "$SPLIT" \
        --output_dir    results/favc_causal
    echo ""
    echo "Causal results:"
    cat results/favc_causal/eval_results.txt
else
    echo "WARNING: Causal checkpoint not found: $CAUSAL_CKPT"
fi

echo ""
echo "============================================================"
echo "  Evaluation done: $(date)"
echo "============================================================"
