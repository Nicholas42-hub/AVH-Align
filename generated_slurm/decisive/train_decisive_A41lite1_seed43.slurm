#!/bin/bash
#SBATCH --job-name=tr_decisive_A41lite1_43
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.err

cd /data/projects/punim2637/nnliang/AVH-Align
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p /data/projects/punim2637/nnliang/AVH-Align/logs
cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
echo ">>> tr_decisive_A41lite1_43 starting on $(hostname) at $(date)"
$PYTHON train_e2e_a41.py --config_path "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/decisive/decisive_A41lite1_seed43.yaml"
echo ">>> tr_decisive_A41lite1_43 finished at $(date)"
