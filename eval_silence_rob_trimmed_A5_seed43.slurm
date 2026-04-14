#!/bin/bash
#SBATCH --job-name=sil_trim_A5s43
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/silence_rob_trimmed_A5_seed43_%j.out
#SBATCH --error=logs/silence_rob_trimmed_A5_seed43_%j.err

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

# Use truly-trimmed features (Smeu-style: raw waveform trimmed, re-extracted AV-HuBERT)
# trim_frames=0 means no additional zero-padding on top of already-trimmed features
python eval_leading_silence_robustness.py \
    --ckpt      outputs_A5_seed43/ckpts/model-epoch=17.ckpt \
    --model     fcd \
    --features_path   /data/projects/punim2637/nnliang/AVH-Align/data/favc_features_trimmed \
    --csv_root_path   csv_metadata/favc_npz_matched \
    --split     test \
    --trim_frames 0 \
    --output_dir outputs_A5_seed43/results_silence_rob_trimmed

echo "Done: A5 seed43 true-trimmed silence robustness"
