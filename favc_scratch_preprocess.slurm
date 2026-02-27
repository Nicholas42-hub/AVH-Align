#!/bin/bash
#SBATCH --job-name=favc_scratch_prep
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/favc_scratch_preprocess_%j.out
#SBATCH --error=logs/favc_scratch_preprocess_%j.err

# ============================================================
#  Preprocess FakeAVCeleb using SCRATCH space
#  
#  Source:  /data/scratch/.../FakeAVCeleb_v1.2 (21,544 videos)
#  Output:  /data/scratch/.../favc_preprocessed (temporary)
#  Final:   NPZ files will be copied to projects after feature extraction
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Preprocessing (Using Scratch)"
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

# Use SCRATCH paths
SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
FAVC_ROOT=$SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2
SAVE_PATH=$SCRATCH_ROOT/favc_preprocessed

METADATA=av1m_metadata/favc_metadata_corrected.csv

# Verify dataset in scratch
TOTAL_VIDS=$(find "$FAVC_ROOT" -name "*.mp4" | wc -l)
echo ""
echo "Dataset Status:"
echo "  Source (SCRATCH): $FAVC_ROOT"
echo "  Total videos: $TOTAL_VIDS"
echo "  Output (SCRATCH): $SAVE_PATH"
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
    --save_path "$SAVE_PATH" \
    --max_workers 32

echo ""
echo "============================================================"
echo "Preprocessing completed: $(date)"
DONE=$(find "$SAVE_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Total preprocessed videos: $DONE / $TOTAL_VIDS"
echo ""
echo "Scratch usage:"
check_scratch_usage
echo "============================================================"
echo ""
echo "Next step: Run feature extraction with:"
echo "  sbatch favc_scratch_features.slurm"
echo "============================================================"
