#!/bin/bash
#SBATCH --job-name=A6pdv2_ev
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/A6_av1m_pseudodomain_v2_eval_%A_%a.out
#SBATCH --error=logs/A6_av1m_pseudodomain_v2_eval_%A_%a.err

# ============================================================
#  Matched FakeAVCeleb evaluation for A6-AV1M-PseudoDomain-v2.
#  Fixed ablation: all AV1M clips (real+fake), speaker-group domain labels.
#  Seeds 43, 44, 45.
# ============================================================

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_matched"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  A6-AV1M-PseudoDomain-v2 eval | seed=${SEED}"
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
  echo "No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi
echo "Using checkpoint: $CKPT"

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
echo "  Done: A6-AV1M-PseudoDomain-v2 seed=${SEED} at $(date)"
echo "============================================================"
