#!/bin/bash
#SBATCH --job-name=anomaly_sup_A6
#SBATCH --partition=gpu-a100-preempt,gpu-l40s-preempt,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/anomaly_sup_A6_%A_%a.out
#SBATCH --error=logs/anomaly_sup_A6_%A_%a.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

CKPT=$(ls "$BASE/avh_sup/outputs_A6_real_domain_100ep_seed${SEED}/ckpts/"model-epoch=*.ckpt 2>/dev/null | sort | tail -1)
OUT="$BASE/avh_sup/anomaly_results_sup_A6_seed${SEED}"

echo "============================================================"
echo "  Anomaly scores | model=sup_A6 seed=${SEED}"
echo "  ckpt : $CKPT"
echo "  out  : $OUT"
echo "  Start: $(date)"
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

$PYTHON avh_sup/eval_anomaly_scores.py \
    --model_type sup \
    --ckpt_path  "$CKPT" \
    --output_dir "$OUT" \
    --knn_k 5 \
    --alpha 1.0 \
    --num_workers 4

echo ""
echo "============================================================"
echo "  Done: sup_A6 seed=${SEED} at $(date)"
echo "============================================================"
