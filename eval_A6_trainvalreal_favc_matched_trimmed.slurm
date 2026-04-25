#!/bin/bash
#SBATCH --job-name=tvr_trim
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/A6_trainvalreal_favc_matched_trimmed_%A_%a.out
#SBATCH --error=logs/A6_trainvalreal_favc_matched_trimmed_%A_%a.err

# ============================================================
#  Trimmed matched-FAVC eval for A6-trainvalreal (Option B)
#  Uses favc_features_trimmed (raw-waveform leading silence removed)
#  to confirm the FV-RA gain is not a leading-silence artifact.
# ============================================================

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_trainvalreal_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_matched_trimmed"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features_trimmed"

echo "============================================================"
echo "  A6-trainvalreal TRIMMED matched-FAVC | seed=${SEED}"
echo "  Features: $FEATURES"
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
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: A6-trainvalreal seed${SEED} trimmed matched at $(date)"
echo "============================================================"
