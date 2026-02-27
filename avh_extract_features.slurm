#!/bin/bash
#SBATCH --job-name=avh_features
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/avh_features_%j.out
#SBATCH --error=logs/avh_features_%j.err

# ============================================================
#  Step 2: Extract AV-HuBERT features for AVH-Align
#
#  Reads preprocessed ROI videos + wavs from data/avh_preprocessed/
#  Saves .npz feature files to data/avh_features/
#
#  Usage:
#    sbatch avh_extract_features.slurm           # test split (default)
#    SPLIT=train sbatch avh_extract_features.slurm
# ============================================================

echo "============================================================"
echo "  AVH-Align Feature Extraction"
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
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SPLIT=${SPLIT:-test}   # 'test' → processes val/ subset; 'train' → train/

if [ "$SPLIT" == "test" ]; then
    METADATA=av1m_metadata/test_metadata.csv
else
    METADATA=av1m_metadata/train_metadata.csv
fi
# The extraction script appends /val or /train to data_path and save_path
DATA_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/avh_preprocessed
SAVE_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/avh_features
CKPT=self_large_vox_433h.pt

echo "Split      : $SPLIT"
echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Checkpoint : $CKPT"
echo ""

# Verify checkpoint is present
if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: Checkpoint not found: av_hubert/avhubert/$CKPT"
    echo "       Run setup_avh_align.sh first!"
    exit 1
fi

mkdir -p "$SAVE_PATH" logs

# Run from avhubert directory (required for relative imports of hubert modules)
cd av_hubert/avhubert

python deepfake_feature_extraction.py \
    --dataset   AV1M \
    --split     "$SPLIT" \
    --metadata  /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH"

echo ""
echo "============================================================"
echo "  Feature extraction done: $(date)"
DONE=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
echo "  Feature files ready: $DONE"
echo "============================================================"
