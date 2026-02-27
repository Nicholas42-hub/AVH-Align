#!/bin/bash
#SBATCH --job-name=avh_preprocess
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/avh_preprocess_%j.out
#SBATCH --error=logs/avh_preprocess_%j.err

# ============================================================
#  Step 1: Preprocess AV1M val videos for AVH-Align
#
#  Runs deepfake_preprocess.py from av_hubert/avhubert/ for
#  the test split (10000 videos from val/) to reproduce
#  paper results with the pretrained AVH-Align_AV1M.pt.
#
#  Usage:
#    sbatch avh_preprocess.slurm
#    SPLIT=train sbatch avh_preprocess.slurm   # for training
# ============================================================

echo "============================================================"
echo "  AVH-Align Preprocess"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

# Patch ffmpeg path dynamically
FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py 2>/dev/null || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SPLIT=${SPLIT:-test}   # 'test' uses val/ data for paper verification; 'train' uses train/

if [ "$SPLIT" == "test" ]; then
    METADATA=av1m_metadata/test_metadata.csv
else
    METADATA=av1m_metadata/train_metadata.csv
fi
# Note: the preprocess script appends /val or /train to DATA_PATH internally
DATA_PATH=/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M
SAVE_PATH=data/avh_preprocessed

echo "Split      : $SPLIT"
echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Videos in metadata: $(tail -n +2 $METADATA | wc -l)"
echo ""

mkdir -p "$SAVE_PATH" logs

# Run from avhubert directory (required for relative imports)
cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset    AV1M \
    --split      "$SPLIT" \
    --metadata   /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path  "$DATA_PATH" \
    --save_path  /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH \
    --max_workers 16

echo ""
echo "============================================================"
echo "  Preprocessing done: $(date)"
DONE=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "  ROI videos ready: $DONE"
echo "============================================================"
