#!/bin/bash
#SBATCH --job-name=DAOFv2_tvr
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/DAOF_v2_trainvalreal_seed%a_%A.out
#SBATCH --error=logs/DAOF_v2_trainvalreal_seed%a_%A.err

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  DAOF v2 — ct_dim=1024, cs_dim=256, reduced regularization"
echo "  Seed: ${SEED}"
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

mkdir -p avh_sup/outputs_DAOF_v2_trainvalreal_seed${SEED}/{logs,ckpts,results} logs

cd avh_sup
$PYTHON train_test_fcd_daof.py \
  --config_path configs/DAOF_v2_trainvalreal_seed${SEED}.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
