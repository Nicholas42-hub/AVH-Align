#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

export PATH=/data/projects/punim2637/nnliang/avh_env/bin:/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin:$PATH
export PYTHONNOUSERSITE=1
cd /data/projects/punim2637/nnliang/AVH-Align

# Just check quota
df -h /data/projects/punim2637/nnliang
ls -la batch_metadata_job2_temp.csv
head -5 batch_metadata_job2_temp.csv

