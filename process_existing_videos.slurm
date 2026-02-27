#!/bin/bash
#SBATCH --job-name=process_existing
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/process_existing_%j.out
#SBATCH --error=logs/process_existing_%j.err

module purge
module load GCCcore/11.3.0
module load FFmpeg/4.4.2
export PATH="/data/projects/punim2637/nnliang/avh_env/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /data/projects/punim2637/nnliang/AVH-Align

echo "=========================================="
echo "Processing Existing Extracted Videos"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

python process_existing_videos.py

echo ""
echo "End: $(date)"
