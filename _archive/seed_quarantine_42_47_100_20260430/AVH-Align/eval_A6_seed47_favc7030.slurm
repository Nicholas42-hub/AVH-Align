#!/bin/bash
#SBATCH --job-name=A6_eval_s47
#SBATCH --partition=deeplearn
#SBATCH --account=punim2637
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/A6_eval_seed47_%A_%a.out
#SBATCH --error=logs/A6_eval_seed47_%A_%a.err

# ============================================================
#  Full FakeAVCeleb 70/30 evaluation (N=6,667) for seed47
#
#  Array index → VARIANT:
#    0: A6_trainvalreal       seed47
#    1: A6_udelta_trainvalreal seed47
# ============================================================

VARIANTS=(A6_trainvalreal A6_udelta_trainvalreal)
MODELS=(fcd_a6 fcd_a6)

VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=47

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${VARIANT}_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_full"

CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  Full FAVC eval (N=6667) | ${VARIANT} seed=${SEED}"
echo "  Model   : $MODEL"
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
  echo "ERROR: No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi
echo "  Ckpt    : $CKPT"

cd avh_sup

python eval_favc.py \
  --ckpt "$CKPT" \
  --model "$MODEL" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: ${VARIANT} seed=${SEED} at $(date)"
echo "============================================================"
