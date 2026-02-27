#!/bin/bash
#SBATCH --job-name=avh_feat_45k
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/avh_features_train_45k_%j.out
#SBATCH --error=logs/avh_features_train_45k_%j.err

# ============================================================
#  Step 2: Extract AV-HuBERT features for FULL TRAINING SET (45k videos)
#
#  Uses av1m_metadata/train_metadata.csv (45k clips from train/)
#  Reads from data/avh_preprocessed/train/
#  Saves .npz to data/avh_features/train/
# ============================================================

echo "============================================================"
echo "  AVH-Align Feature Extraction - FULL TRAINING SET (45k)"
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

METADATA=av1m_metadata/train_metadata.csv
DATA_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/avh_preprocessed
SAVE_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/avh_features
CKPT=self_large_vox_433h.pt

echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Clips in metadata: $(tail -n +2 $METADATA | wc -l)"

if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: Checkpoint not found: av_hubert/avhubert/$CKPT"
    exit 1
fi

mkdir -p "$SAVE_PATH" logs

cd av_hubert/avhubert

python deepfake_feature_extraction.py \
    --dataset   AV1M \
    --split     train \
    --metadata  /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH"

echo ""
echo "============================================================"
echo "  Feature extraction (full training set) done: $(date)"
DONE=$(find "$SAVE_PATH/train" -name "*.npz" 2>/dev/null | wc -l)
echo "  Total feature files in train/: $DONE"
echo "============================================================"
