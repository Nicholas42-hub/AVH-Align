#!/bin/bash
#SBATCH --job-name=A6_5seed_eval
#SBATCH --partition=gpu-l40s-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/A6_5seed_eval_%A_%a.out
#SBATCH --error=logs/A6_5seed_eval_%A_%a.err

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

VARIANTS=(A6 A6 A6_udelta_trainvalreal A6_udelta_trainvalreal)
SEEDS=(46 47 46 47)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

if [[ "$VARIANT" == "A6" ]]; then
  OUTPUT_DIR="$BASE/avh_sup/outputs_A6_seed${SEED}"
else
  OUTPUT_DIR="$BASE/avh_sup/outputs_A6_udelta_trainvalreal_seed${SEED}"
fi
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_full"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

mkdir -p "$RESULTS_DIR" logs

CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi

echo "============================================================"
echo "  Full FAVC 70/30 eval | ${VARIANT} seed=${SEED}"
echo "  Ckpt   : $CKPT"
echo "  Results: $RESULTS_DIR"
echo "  Start  : $(date)"
echo "============================================================"

cd avh_sup
"$PYTHON" eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed=${SEED} at $(date)"
echo "============================================================"

