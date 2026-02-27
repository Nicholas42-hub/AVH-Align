#!/bin/bash
#SBATCH --job-name=avd1m_scratch_prep
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/avd1m_scratch_preprocess_%j.out
#SBATCH --error=logs/avd1m_scratch_preprocess_%j.err

# ============================================================
#  Preprocess AVDeepfake1M Validation Set using SCRATCH
#  
#  Source:  /data/scratch/.../AVDeepfake1M/val (57,340 videos)
#  Output:  /data/scratch/.../avd1m_preprocessed (temporary)
#  Final:   NPZ files will be copied to projects after feature extraction
# ============================================================

echo "============================================================"
echo "  AVDeepfake1M Preprocessing (Using Scratch)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2
module load CMake/3.24.3
# dlib is installed in conda environment

# Use conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo "FFmpeg: $(which ffmpeg)"

# Set environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

# Update FFmpeg path
FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py 2>/dev/null || true

# Use SCRATCH paths
SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
DATA_PATH=$SCRATCH_ROOT/Datasets/AVDeepfake1M
SAVE_PATH=$SCRATCH_ROOT/avd1m_preprocessed

METADATA=av1m_metadata/val_metadata_full.csv

# Verify dataset in scratch
TOTAL_VIDS=$(find "$DATA_PATH/val" -name "*.mp4" 2>/dev/null | wc -l)
echo ""
echo "Dataset Status:"
echo "  Source (SCRATCH): $DATA_PATH/val"
echo "  Total videos: $TOTAL_VIDS"
echo "  Output (SCRATCH): $SAVE_PATH"
echo ""

# Check metadata
if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata not found: $METADATA"
    echo "Please ensure val_metadata_full.csv exists"
    exit 1
fi

METADATA_COUNT=$(tail -n +2 $METADATA | wc -l)
echo "Metadata entries: $METADATA_COUNT"
echo ""

mkdir -p "$SAVE_PATH" logs

# Check existing preprocessed files
EXISTING=$(find "$SAVE_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Already preprocessed: $EXISTING / $TOTAL_VIDS"
echo ""

# Run preprocessing
echo "============================================================"
echo "Starting preprocessing (32 workers)..."
echo "============================================================"
echo ""

cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset AVDeepfake1M \
    --metadata /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --split val \
    --max_workers 32

echo ""
echo "============================================================"
echo "Preprocessing completed: $(date)"
DONE=$(find "$SAVE_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Total preprocessed videos: $DONE / $TOTAL_VIDS"
echo "Completion rate: $(awk "BEGIN {printf \"%.1f%%\", $DONE/$TOTAL_VIDS*100}")"
echo ""
echo "Scratch usage:"
check_scratch_usage
echo "============================================================"
echo ""
echo "Next step: Run feature extraction with:"
echo "  sbatch avd1m_scratch_features.slurm"
echo "============================================================"
