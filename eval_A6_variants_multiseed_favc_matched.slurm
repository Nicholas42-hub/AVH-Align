#!/bin/bash
#SBATCH --job-name=A6v_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/A6v_eval_%A_%a.out
#SBATCH --error=logs/A6v_eval_%A_%a.err

# ============================================================
#  Matched FakeAVCeleb evaluation for A6 variant seed sweeps.
# ============================================================

VARIANTS=(A6_lite_trainreal A6_lite_trainreal A6_lite_trainreal A6_sharedonly_trainreal A6_sharedonly_trainreal A6_sharedonly_trainreal)
SEEDS=(43 44 45 43 44 45)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${VARIANT}_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_matched"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  A6 variant eval | ${VARIANT} seed=${SEED}"
echo "  Output  : $OUTPUT_DIR"
echo "  Results : $RESULTS_DIR"
echo "  Start   : $(date)"
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

mkdir -p "$RESULTS_DIR" logs
CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi

cd avh_sup

if [[ "$VARIANT" == "A6_lite_trainreal" ]]; then
  MODEL_FLAG="fcd_a6_lite"
elif [[ "$VARIANT" == "A6_sharedonly_trainreal" ]]; then
  MODEL_FLAG="fcd_a6_sharedonly"
else
  echo "Unknown variant: $VARIANT" >&2
  exit 1
fi

python eval_favc.py \
  --ckpt "$CKPT" \
  --model "$MODEL_FLAG" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Eval done: ${VARIANT} seed=${SEED} at $(date)"
echo "============================================================"
