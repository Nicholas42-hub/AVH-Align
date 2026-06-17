#!/bin/bash
#SBATCH --job-name=unsup_avh
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/unsup_avh_%A_%a.out
#SBATCH --error=logs/unsup_avh_%A_%a.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
CKPT="$BASE/checkpoints/AVH-Align_AV1M.pt"

DATASETS=(favc favc av1m)
FEATURES=(
  "$BASE/data/favc_features"
  "$BASE/data/favc_features_trimmed"
  "$BASE/data/avh_features"
)
CSV_ROOTS=(
  "$BASE/avh_sup/csv_metadata/favc_7030_canonical"
  "$BASE/avh_sup/csv_metadata/favc_7030_canonical"
  "$BASE/avh_sup/csv_metadata/av1m"
)
SPLITS=(test test val)
OUTS=(
  "$BASE/avh_sup/outputs_unsup_avhalign/results_favc_7030"
  "$BASE/avh_sup/outputs_unsup_avhalign/results_favc_7030_trimmed"
  "$BASE/avh_sup/outputs_unsup_avhalign/results_av1m_val"
)

IDX=${SLURM_ARRAY_TASK_ID}
DATASET=${DATASETS[$IDX]}
FEATURE_PATH=${FEATURES[$IDX]}
CSV_ROOT=${CSV_ROOTS[$IDX]}
SPLIT=${SPLITS[$IDX]}
OUT=${OUTS[$IDX]}

echo "============================================================"
echo "  Unsupervised AVH-Align eval | task=$IDX"
echo "  Dataset : $DATASET"
echo "  Ckpt    : $CKPT"
echo "  Features: $FEATURE_PATH"
echo "  CSV     : $CSV_ROOT"
echo "  Split   : $SPLIT"
echo "  Out     : $OUT"
echo "  Node    : $SLURM_NODELIST"
echo "  Start   : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$OUT"

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

cd avh_sup

$PYTHON eval_unsup_avhalign.py \
  --checkpoint_path "$CKPT" \
  --dataset "$DATASET" \
  --features_path "$FEATURE_PATH" \
  --csv_root_path "$CSV_ROOT" \
  --split "$SPLIT" \
  --output_dir "$OUT"

echo ""
echo "============================================================"
echo "  Done: task=$IDX at $(date)"
echo "============================================================"
