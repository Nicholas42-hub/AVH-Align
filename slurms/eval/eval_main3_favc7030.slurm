#!/bin/bash
#SBATCH --job-name=eval_7030
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/eval_7030_%A_%a.out
#SBATCH --error=logs/eval_7030_%A_%a.err

# ============================================================
#  Eval A2 / A5 / A6 on FakeAVCeleb 70-30 public split
#  Array: 9 = 3 models x 3 seeds (43,44,45)
# ============================================================

MODELS=(A2 A2 A2 A5 A5 A5 A6 A6 A6)
MODEL_TYPES=(ablation ablation ablation fcd fcd fcd fcd_a6 fcd_a6 fcd_a6)
SEEDS=(43 44 45 43 44 45 43 44 45)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${MODEL}_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  Eval ${MODEL} seed${SEED} on FakeAVCeleb 70-30 split"
echo "  Model type : ${MODEL_TYPE}"
echo "  Job        : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node       : $SLURM_NODELIST"
echo "  Start      : $(date)"
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
  --model "$MODEL_TYPE" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: ${MODEL} seed${SEED} at $(date)"
echo "============================================================"
