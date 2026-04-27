#!/bin/bash
#SBATCH --job-name=ev_DANN_alpha0p5_seed45
#SBATCH --partition=gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,gpu-a100-short,gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
CKPT=$(ls "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_DANN_alpha0p5_seed45/ckpts"/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "No checkpoint found for DANN_alpha0p5_seed45" >&2
    exit 2
fi

python eval_favc.py             --ckpt "$CKPT"             --model dann             --features_path "/data/projects/punim2637/nnliang/AVH-Align/data/favc_features"             --csv_root_path "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical"             --split test             --output_dir "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_DANN_alpha0p5_seed45/results_favc_7030_full"
