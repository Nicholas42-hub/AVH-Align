#!/bin/bash
#SBATCH --job-name=DAOF_tvr_s43
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/DAOF_trainvalreal_seed43_%j.out
#SBATCH --error=logs/DAOF_trainvalreal_seed43_%j.err

echo "============================================================"
echo "  DAOF — Dual-Axis Orthogonal Factorization"
echo "  Two-level decomposition: modality axis + domain-task axis"
echo "  Domain support: FAVC train+val real"
echo "  Seed: 43"
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

mkdir -p avh_sup/outputs_DAOF_trainvalreal_seed43/{logs,ckpts,results} logs

cd avh_sup
$PYTHON train_test_fcd_daof.py \
  --config_path configs/DAOF_trainvalreal_seed43.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
