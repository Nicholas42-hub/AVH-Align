#!/bin/bash
#SBATCH --job-name=avad_favc_shard
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/avad_favc_shard_%A_%a.out
#SBATCH --error=logs/avad_favc_shard_%A_%a.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVAD_DIR="$BASE/external/audio-visual-forensics"
PROTO_DIR="$BASE/submission_prep/avad_matched_baseline"
SHARD_ID=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate avad_env

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

cd "$AVAD_DIR"

python detect.py \
    --test_video_path "$PROTO_DIR/shards/shard_${SHARD_ID}_videos.txt" \
    --device cuda:0 \
    --max-len 50 \
    --n_workers 4 \
    --bs 1 \
    --lam 0 \
    --output_dir "$AVAD_DIR/outputs_favc_7030_shards/shard_${SHARD_ID}"
