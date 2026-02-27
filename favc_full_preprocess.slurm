#!/bin/bash
#SBATCH --job-name=favc_full_prep
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/favc_full_preprocess_%j.out
#SBATCH --error=logs/favc_full_preprocess_%j.err

# ============================================================
#  Preprocess FULL FakeAVCeleb Dataset (21,544 videos)
#  
#  Current: 2,703 preprocessed (12.5%)
#  Target:  21,544 total (100%)
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb FULL Dataset Preprocessing"
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
module load dlib/19.24.2

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

FAVC_ROOT=/data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2
METADATA=av1m_metadata/favc_metadata.csv
SAVE_PATH=data/favc_preprocessed

# Verify dataset
TOTAL_VIDS=$(find "$FAVC_ROOT" -name "*.mp4" | wc -l)
echo ""
echo "Dataset Status:"
echo "  Source: $FAVC_ROOT"
echo "  Total videos: $TOTAL_VIDS"
echo "  Output: $SAVE_PATH"
echo ""

# Check metadata
if [ ! -f "$METADATA" ]; then
    echo "Generating metadata CSV..."
    python generate_favc_metadata.py \
        --split all \
        --csv_dir avh_sup/csv_metadata/favc \
        --out "$METADATA"
fi

METADATA_COUNT=$(tail -n +2 $METADATA | wc -l)
echo "Metadata entries: $METADATA_COUNT"
echo ""

mkdir -p "$SAVE_PATH" logs

# Run preprocessing
echo "============================================================"
echo "Starting preprocessing (32 workers)..."
echo "============================================================"
echo ""

cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset FakeAVCeleb \
    --metadata /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path "$FAVC_ROOT" \
    --save_path /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH \
    --max_workers 32

echo ""
echo "============================================================"
echo "Preprocessing completed: $(date)"
DONE=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Total preprocessed videos: $DONE / $TOTAL_VIDS"
echo "Success rate: $(python3 -c "print(f'{$DONE/$TOTAL_VIDS*100:.1f}%')")"
echo "============================================================"
