#!/bin/bash
#SBATCH --job-name=avd1m_job2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/batch_avd1m_job2_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/batch_avd1m_job2_%j.err

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"  
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Set environment PATH (includes Python, 7z, FFmpeg, and all dependencies)
export PATH=/data/projects/punim2637/nnliang/avh_env/bin:/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin:$PATH
export PYTHONPATH=/data/projects/punim2637/nnliang/AVH-Align:$PYTHONPATH
export PYTHONNOUSERSITE=1  # Ignore user site-packages to avoid conflicts

# Navigate to working directory
cd /data/projects/punim2637/nnliang/AVH-Align

# Run the batch pipeline (Job 2: Batches 7-12)
echo "Starting batch pipeline Job 2 (Batches 7-12)..."
python batch_download_extract_features_job2.py

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
