#!/bin/bash
#SBATCH --job-name=A6_route_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-14%5
#SBATCH --output=logs/A6_routing_eval_%A_%a.out
#SBATCH --error=logs/A6_routing_eval_%A_%a.err

set -euo pipefail

VARIANTS=(noudelta expose_residual ddis_only dadv_only udelta100)
SEEDS=(43 44 45)

VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_trainvalreal_${VARIANT}_seed${SEED}"
FEATURES="$BASE/data/favc_features"
MATCHED_CSV="$BASE/avh_sup/csv_metadata/favc_npz_matched"
FULL_CSV="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

echo "============================================================"
echo "  A6 routing ablation eval"
echo "  Variant : ${VARIANT}"
echo "  Seed    : ${SEED}"
echo "  Output  : ${OUTPUT_DIR}"
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

mkdir -p logs

CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi
echo "  Ckpt: $CKPT"

cd avh_sup

$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$MATCHED_CSV" \
  --split test \
  --output_dir "$OUTPUT_DIR/results_favc_matched"

$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$FULL_CSV" \
  --split test \
  --output_dir "$OUTPUT_DIR/results_favc_7030"

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed${SEED} at $(date)"
echo "============================================================"
