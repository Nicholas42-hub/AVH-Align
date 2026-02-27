#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/extract_features_%j.out
#SBATCH --error=logs/extract_features_%j.err

module purge
module load GCCcore/11.3.0
module load FFmpeg/4.4.2
export PATH="/data/projects/punim2637/nnliang/avh_env/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /data/projects/punim2637/nnliang/AVH-Align

echo "=========================================="
echo "FEATURE EXTRACTION ONLY - PART 2 OF 2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

python extract_features_only.py

echo ""
echo "End: $(date)"
