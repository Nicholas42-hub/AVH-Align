#!/bin/bash
#SBATCH --job-name=eval_7030_trim
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --dependency=afterok:24113664
#SBATCH --output=logs/eval_7030_trim_%A_%a.out
#SBATCH --error=logs/eval_7030_trim_%A_%a.err

set -euo pipefail

# ============================================================
#  Evaluate A2 / A5 / A6-pseudodomain_v2 on TRIMMED FakeAVCeleb
#  70/30 public test split.  Runs after trimmed feature extraction
#  (job 24113631).
#
#  Array indices:
#   0-2: A2  seeds 43,44,45   model=ablation
#   3-5: A5  seeds 43,44,45   model=fcd
#   6-8: A6_av1m_pseudodomain_v2 seeds 43,44,45  model=fcd_a6
# ============================================================

VARIANTS=( \
  A2  A2  A2 \
  A5  A5  A5 \
  A6_av1m_pseudodomain_v2  A6_av1m_pseudodomain_v2  A6_av1m_pseudodomain_v2 \
)
SEEDS=(43 44 45  43 44 45  43 44 45)
MODELS=(ablation ablation ablation  fcd fcd fcd  fcd_a6 fcd_a6 fcd_a6)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${VARIANT}_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_trimmed"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features_trimmed"

echo "============================================================"
echo "  TRIMMED 70/30 eval | ${VARIANT} seed=${SEED} model=${MODEL}"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$RESULTS_DIR" logs

CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi
echo "  Ckpt: $CKPT"

cd avh_sup
$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model "$MODEL" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed${SEED} trimmed 70/30 at $(date)"
echo "============================================================"
