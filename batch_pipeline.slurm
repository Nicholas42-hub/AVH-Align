#!/bin/bash
#SBATCH --job-name=batch_avd1m
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=logs/batch_avd1m_%j.out
#SBATCH --error=logs/batch_avd1m_%j.err

# ============================================================
#  Batch Download & Feature Extraction Pipeline
#  
#  Downloads training videos in batches, extracts features,
#  then deletes source files to conserve disk space.
#  
#  Processes 254 archive parts in batches of 20
#  (~13 batches total, ~12 hours per batch = ~7 days)
# ============================================================

echo "============================================================"
echo "  AVH-Align Batch Pipeline"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

# Install p7zip if needed
echo ""
echo "Ensuring p7zip is available..."
conda install -y -c conda-forge p7zip 2>/dev/null || echo "p7zip already installed"

# Fix ffmpeg path in preprocessing script
FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py 2>/dev/null || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Check initial disk space
echo ""
echo "Initial disk space:"
df -h /data/gpfs

echo ""
echo "============================================================"
echo "  Starting Batch Pipeline"
echo "  This will process ~254 archive parts in batches of 20"
echo "  Estimated time: 5-7 days"
echo "============================================================"
echo ""

mkdir -p logs data/avh_features/train

# Run the batch pipeline
python batch_download_extract_features.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Pipeline finished: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"

# Show results
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Success! Checking results..."
    echo ""
    echo "Features extracted:"
    FEATURE_COUNT=$(find data/avh_features/train -name "*.npz" 2>/dev/null | wc -l)
    echo "  Total .npz files: $FEATURE_COUNT"
    
    echo ""
    echo "Feature directory size:"
    du -sh data/avh_features/train
    
    echo ""
    echo "Final disk space:"
    df -h /data/gpfs
    
    echo ""
    echo "Next step: Train the model"
    echo "  python train.py --name=AVH-Align_45k --data_root_path=data/avh_features/"
else
    echo ""
    echo "✗ Pipeline failed. Check logs above."
    echo ""
    echo "Progress saved in batch_progress.json"
    echo "You can resume by resubmitting this job"
fi

echo ""
echo "============================================================"
