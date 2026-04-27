#!/bin/bash
#SBATCH --job-name=tr_DANN_alpha0p3_seed44
#SBATCH --partition=gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,deeplearn,feit-gpu-a100,gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.err

set -euo pipefail
cd /data/projects/punim2637/nnliang/AVH-Align
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_DANN_alpha0p3_seed44/logs /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_DANN_alpha0p3_seed44/ckpts /data/projects/punim2637/nnliang/AVH-Align/logs
cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
python train_da_baselines.py             --method dann             --config_path "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/alpha_sweep/DANN_alpha0p3_seed44.yaml"             --seed 44             --output_dir "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_DANN_alpha0p3_seed44"
