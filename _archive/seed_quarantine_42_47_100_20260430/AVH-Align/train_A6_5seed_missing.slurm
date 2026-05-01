#!/bin/bash
#SBATCH --job-name=A6_5seed_train
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/A6_5seed_train_%A_%a.out
#SBATCH --error=logs/A6_5seed_train_%A_%a.err

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

mkdir -p logs

VARIANTS=(A6 A6 A6_udelta_trainvalreal)
SEEDS=(46 47 46)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  Train missing 5-seed member | ${VARIANT} seed=${SEED}"
echo "  Start: $(date)"
echo "============================================================"

cd avh_sup

if [[ "$VARIANT" == "A6" ]]; then
  OUTPUT_DIR="$BASE/avh_sup/outputs_A6_seed${SEED}"
  mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results}
  "$PYTHON" train_test_fcd_a6.py \
    --config_path configs/A6_legacy.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
else
  mkdir -p "$BASE/avh_sup/outputs_A6_udelta_trainvalreal_seed${SEED}"/{logs,ckpts,results}
  "$PYTHON" train_test_fcd_a6_trainvalreal.py \
    --config_path "configs/A6_udelta_trainvalreal_seed${SEED}.yaml" \
    --mode train
fi

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed=${SEED} at $(date)"
echo "============================================================"

