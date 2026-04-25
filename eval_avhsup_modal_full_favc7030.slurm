#!/bin/bash
#SBATCH --job-name=avhsup_modal_full
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/avhsup_modal_full_%A_%a.out
#SBATCH --error=logs/avhsup_modal_full_%A_%a.err

set -euo pipefail

COMBOS=(
  "both 43"
  "both 44"
  "both 45"
  "audio 43"
  "audio 44"
  "audio 45"
  "video 43"
  "video 44"
  "video 45"
)

read -r INPUT_TYPE SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUT="$BASE/avh_sup/outputs_avhsup_av1m_${INPUT_TYPE}_seed${SEED}"
RESULTS="$OUT/results_favc_7030_full"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  Plain AVH_Sup modal baseline | FULL FAVC 70/30"
echo "  Input type : $INPUT_TYPE"
echo "  Seed       : $SEED"
echo "  Results    : $RESULTS"
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

mkdir -p "$RESULTS" logs

CKPT="$(ls -t "$OUT"/ckpts/*.ckpt | head -1)"
echo "Checkpoint: $CKPT"

cd avh_sup
python eval_favc.py \
  --ckpt "$CKPT" \
  --model baseline \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS"

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
