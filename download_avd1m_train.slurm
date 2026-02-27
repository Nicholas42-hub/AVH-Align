#!/bin/bash
#SBATCH --job-name=dl_avd1m_train
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/download_avd1m_train_%j.out
#SBATCH --error=logs/download_avd1m_train_%j.err

# ============================================================
#  Download AV-Deepfake1M Training Set from Hugging Face
# ============================================================

echo "============================================================"
echo "  AV-Deepfake1M Training Set Download"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

# Install 7z for extracting multi-volume archives
echo ""
echo "Installing p7zip for archive extraction..."
conda install -y -c conda-forge p7zip || pip install py7zr

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

# Check available disk space
echo ""
echo "Disk space check:"
df -h /data/projects/punim2637/nnliang/Datasets

echo ""
echo "Starting download..."
echo "This may take 24-72 hours for the full dataset"
echo ""

mkdir -p logs

python download_avd1m_train.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Download script finished: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Checking downloaded content..."
    ls -lh /data/projects/punim2637/nnliang/Datasets/AVDeepfake1M/
    
    TRAIN_DIR="/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M/train"
    if [ -d "$TRAIN_DIR" ]; then
        echo ""
        echo "Training directory exists!"
        echo "Video count (this may take a while):"
        find "$TRAIN_DIR" -name "*.mp4" -type f | wc -l
    else
        echo ""
        echo "⚠ Training directory not found"
        echo "Dataset might not include training videos on Hugging Face"
    fi
else
    echo "Download failed. Check the error messages above."
fi

echo ""
echo "============================================================"
