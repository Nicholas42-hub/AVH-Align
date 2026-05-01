#!/bin/bash
#SBATCH --job-name=A6v_ms
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/A6v_ms_%A_%a.out
#SBATCH --error=logs/A6v_ms_%A_%a.err

# ============================================================
#  A6 variant stability sweep:
#    - A6_lite_trainreal      × seeds 43,44,45
#    - A6_sharedonly_trainreal × seeds 43,44,45
# ============================================================

VARIANTS=(A6_lite_trainreal A6_lite_trainreal A6_lite_trainreal A6_sharedonly_trainreal A6_sharedonly_trainreal A6_sharedonly_trainreal)
SEEDS=(43 44 45 43 44 45)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${VARIANT}_seed${SEED}"

echo "============================================================"
echo "  A6 variant seed sweep | ${VARIANT} seed=${SEED}"
echo "  Job ID : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node   : $SLURM_NODELIST"
echo "  Output : $OUTPUT_DIR"
echo "  Start  : $(date)"
echo "============================================================"

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

mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup

if [[ "$VARIANT" == "A6_lite_trainreal" ]]; then
  python train_test_fcd_a6_lite.py \
    --config_path configs/A6_lite_trainreal.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
elif [[ "$VARIANT" == "A6_sharedonly_trainreal" ]]; then
  python train_test_fcd_a6_sharedonly.py \
    --config_path configs/A6_sharedonly_trainreal.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
else
  echo "Unknown variant: $VARIANT" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed=${SEED} at $(date)"
echo "============================================================"
