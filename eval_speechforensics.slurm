#!/bin/bash
#SBATCH --job-name=speechforensics_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/speechforensics_eval_%j.out
#SBATCH --error=logs/speechforensics_eval_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT="$BASE/checkpoints/large_vox_iter5.pt"
RESULTS_UNTRIMMED="$BASE/avh_sup/results_speechforensics_favc"
RESULTS_TRIMMED="$BASE/avh_sup/results_speechforensics_favc_trimmed"

echo "============================================================"
echo "  SpeechForensics eval on FakeAVCeleb test"
echo "  Start: $(date)"
echo "============================================================"

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

# ── Step 1: Download large_vox_iter5.pt if not present ─────────────────────
if [ ! -f "$CKPT" ]; then
    echo "Downloading large_vox_iter5.pt ..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt \
        -O "$CKPT"
    echo "Download complete."
else
    echo "Checkpoint already present: $CKPT"
fi

# ── Step 2: Evaluate on untrimmed favc_preprocessed ────────────────────────
echo ""
echo "--- Untrimmed evaluation ---"
python eval_speechforensics_favc.py \
    --ckpt          "$CKPT" \
    --csv           avh_sup/csv_metadata/favc/test_split.csv \
    --preprocessed_dir data/favc_preprocessed \
    --output_dir    "$RESULTS_UNTRIMMED" \
    --max_length    50

# ── Step 3: Evaluate on trimmed favc_preprocessed ──────────────────────────
# The trimmed split uses favc_features_trimmed; check if trimmed mouth crops exist
TRIMMED_DIR="$BASE/data/favc_preprocessed_trimmed"
if [ -d "$TRIMMED_DIR" ]; then
    echo ""
    echo "--- Trimmed evaluation ---"
    python eval_speechforensics_favc.py \
        --ckpt          "$CKPT" \
        --csv           avh_sup/csv_metadata/favc/test_split.csv \
        --preprocessed_dir "$TRIMMED_DIR" \
        --output_dir    "$RESULTS_TRIMMED" \
        --max_length    50
else
    echo "Trimmed preprocessed dir not found ($TRIMMED_DIR), skipping trimmed eval."
fi

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
