#!/bin/bash
#SBATCH --job-name=preprocess_only
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_only_%j.out
#SBATCH --error=logs/preprocess_only_%j.err

module purge
module load GCCcore/11.3.0
module load FFmpeg/4.4.2
export PATH="/data/projects/punim2637/nnliang/avh_env/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /data/projects/punim2637/nnliang/AVH-Align

echo "=========================================="
echo "PREPROCESSING ONLY - PART 1 OF 2"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo ""

python preprocess_only.py

echo ""
echo "End: $(date)"
