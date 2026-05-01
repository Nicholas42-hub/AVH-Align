#!/bin/bash
#SBATCH --job-name=silence_rob_A6s43
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/silence_robustness_A6_seed43_%j.out
#SBATCH --error=logs/silence_robustness_A6_seed43_%j.err

set -e

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup

python eval_leading_silence_robustness.py \
    --ckpt      outputs_A6_seed43/ckpts/model-epoch=15.ckpt \
    --model     fcd_a6 \
    --features_path   /data/projects/punim2637/nnliang/AVH-Align/data/favc_features \
    --csv_root_path   csv_metadata/favc \
    --split     test \
    --trim_frames 1 3 5 10 \
    --output_dir outputs_A6_seed43/results_silence_rob

echo "Done: A6 seed43 leading silence robustness"
