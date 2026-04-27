#!/bin/bash
#SBATCH --job-name=dann_eval_rerun
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/dann_eval_rerun_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/dann_eval_rerun_%j.err

set -euo pipefail

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

BASE="/data/projects/punim2637/nnliang/AVH-Align"
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

cd "$BASE/avh_sup"

for SEED in 43 44; do
    OUT="$BASE/avh_sup/outputs_DANN_seed${SEED}"
    CKPT=$(ls "$OUT/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "ERROR: no checkpoint for DANN seed${SEED}" >&2
        continue
    fi
    echo "=== DANN seed=${SEED} ==="
    python eval_favc.py \
        --ckpt          "$CKPT" \
        --model         dann \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split         test \
        --output_dir    "$OUT/results_favc_7030_full"
    echo "Done DANN seed${SEED}: $(date)"
done
