#!/bin/bash
#SBATCH --job-name=A6_tvr_s44
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/A6_trainvalreal_seed44_%j.out
#SBATCH --error=logs/A6_trainvalreal_seed44_%j.err

echo "============================================================"
echo "  A6-trainvalreal seed44"
echo "  Start: $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
cd "$BASE"

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

mkdir -p avh_sup/outputs_A6_trainvalreal_seed44/{logs,ckpts,results} logs

cd avh_sup
$PYTHON train_test_fcd_a6_trainvalreal.py \
  --config_path configs/A6_trainvalreal_seed44.yaml \
  --mode train

echo "Done: $(date)"
