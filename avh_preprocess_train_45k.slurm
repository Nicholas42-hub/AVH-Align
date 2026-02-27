#!/bin/bash
#SBATCH --job-name=avh_prep_45k
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/avh_preprocess_train_45k_%j.out
#SBATCH --error=logs/avh_preprocess_train_45k_%j.err

# ============================================================
#  Step 1: Preprocess AV1M FULL TRAINING SET (45k videos)
#
#  Uses av1m_metadata/train_metadata.csv (45k clips)
#  which live in the train/ directory.
#  SPLIT=train so deepfake_preprocess.py reads from data_path/train/
#  and saves to avh_preprocessed/train/
# ============================================================

echo "============================================================"
echo "  AVH-Align Preprocess - FULL TRAINING SET (45k videos)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py 2>/dev/null || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Use train_metadata.csv with 45k clips from train/ directory
METADATA=av1m_metadata/train_metadata.csv
DATA_PATH=/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M
SAVE_PATH=data/avh_preprocessed

echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Clips in metadata: $(tail -n +2 $METADATA | wc -l)"
echo ""

mkdir -p "$SAVE_PATH" logs

cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset    AV1M \
    --split      train \
    --metadata   /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path  "$DATA_PATH" \
    --save_path  /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH \
    --max_workers 16

echo ""
echo "============================================================"
echo "  Preprocessing done: $(date)"
DONE=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH/train -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "  Total ROI videos ready in train/: $DONE"
echo "============================================================"
