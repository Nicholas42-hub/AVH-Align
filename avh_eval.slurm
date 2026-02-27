#!/bin/bash
#SBATCH --job-name=avh_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/avh_eval_%j.out
#SBATCH --error=logs/avh_eval_%j.err

# ============================================================
#  Step 3: Evaluate AVH-Align pretrained model on AV1M test set
#
#  Uses pretrained checkpoints/AVH-Align_AV1M.pt
#  Expected paper result: AUC ~0.83, AP ~0.88 (from paper)
#
#  Usage:
#    sbatch avh_eval.slurm
#    CKPT=checkpoints/my_trained.pt sbatch avh_eval.slurm  # custom ckpt
# ============================================================

echo "============================================================"
echo "  AVH-Align Evaluation"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

CKPT=${CKPT:-checkpoints/AVH-Align_AV1M.pt}
FEATURES_PATH=data/avh_features/val        # test split stores in val/
METADATA=av1m_metadata/test_metadata.csv
OUTPUT_DIR=results/avh_align_test

echo "Checkpoint     : $CKPT"
echo "Features path  : $FEATURES_PATH"
echo "Metadata       : $METADATA"
echo ""

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

NFEAT=$(find "$FEATURES_PATH" -name "*.npz" 2>/dev/null | wc -l)
NEXPECTED=$(tail -n +2 "$METADATA" | wc -l)
echo "Feature files  : $NFEAT / $NEXPECTED expected"
if [ "$NFEAT" -eq 0 ]; then
    echo "ERROR: No feature files found — run avh_extract_features.slurm first!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python eval.py \
    --checkpoint_path "$CKPT" \
    --features_path   "$FEATURES_PATH" \
    --metadata        "$METADATA" \
    --dataset         AV1M \
    2>&1 | tee "$OUTPUT_DIR/eval_results.txt"

echo ""
echo "============================================================"
echo "  Evaluation done: $(date)"
echo "  Results → $OUTPUT_DIR/eval_results.txt"
echo "============================================================"
