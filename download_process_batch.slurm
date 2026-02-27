#!/bin/bash
#SBATCH --job-name=avd1m_batch_004-006
#SBATCH --partition=cascade
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/batch_004-006_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/batch_004-006_%j.err

# AV-Deepfake1M Batch Download and Process (files 004-006)
# This job downloads archives, extracts, preprocesses, and cleans up

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Setup environment
cd /data/projects/punim2637/nnliang/AVH-Align

# Load required modules
module load GCCcore/11.3.0
module load FFmpeg/4.4.2
module load CMake/3.24.3
module load dlib/19.24.2

# Set Python path to use the conda environment
export PATH="/data/projects/punim2637/nnliang/avh_env/bin:$PATH"
export PYTHONPATH="/data/projects/punim2637/nnliang/avh_env/lib/python3.9/site-packages:$PYTHONPATH"

# Show environment info
echo ""
echo "Python: $(which python3)"
echo "FFmpeg: $(which ffmpeg)"
echo "Working directory: $(pwd)"
echo ""

# Run batch processing pipeline
python3 download_and_process_batch.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
