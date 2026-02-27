#!/bin/bash
#SBATCH --job-name=favc_full_feat
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/favc_full_features_%j.out
#SBATCH --error=logs/favc_full_features_%j.err

# ============================================================
#  Extract Features for FULL FakeAVCeleb Dataset
#  
#  Prerequisite: favc_full_preprocess.slurm completed
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb FULL Feature Extraction"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

# Use conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Set environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

METADATA=av1m_metadata/favc_metadata.csv
DATA_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/favc_preprocessed
SAVE_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/favc_features
CKPT=self_large_vox_433h.pt

echo ""
echo "Configuration:"
echo "  Metadata: $METADATA"
echo "  Data path: $DATA_PATH"
echo "  Save path: $SAVE_PATH"
echo "  Checkpoint: $CKPT"
echo ""

# Count preprocessed videos
PREP_COUNT=$(find "$DATA_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Preprocessed videos: $PREP_COUNT"

if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: Checkpoint not found: av_hubert/avhubert/$CKPT"
    exit 1
fi

mkdir -p "$SAVE_PATH" logs

cd av_hubert/avhubert

echo ""
echo "============================================================"
echo "Starting feature extraction..."
echo "============================================================"
echo ""

python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH"

echo ""
echo "============================================================"
echo "Feature extraction completed: $(date)"
DONE=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
echo "Total feature files: $DONE"
echo "============================================================"
