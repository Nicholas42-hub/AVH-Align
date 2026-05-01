#!/bin/bash
#SBATCH --job-name=ablation_udelta
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_head_ablation_udelta_%j.out
#SBATCH --error=logs/eval_head_ablation_udelta_%j.err

echo "============================================================"
echo "  A6 u_delta head ablation (mask_udelta / mask_uv / mask_ua)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
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

FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"

mkdir -p logs

cd avh_sup

# Run ablation on seed42 u_delta checkpoint (already trained)
CKPT=$(ls outputs_A6_udelta_trainvalreal/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: No checkpoint found in outputs_A6_udelta_trainvalreal/ckpts/" >&2
    exit 1
fi
echo "Checkpoint: $CKPT"

OUT_DIR="outputs_A6_udelta_trainvalreal/results_ablation"
mkdir -p "$OUT_DIR"

python eval_head_ablation_udelta.py \
    --ckpt           "$CKPT" \
    --features_path  "$FEATURES" \
    --csv_root_path  "$CSV_ROOT" \
    --split          test \
    --output_dir     "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"

echo ""
cat "$OUT_DIR/ablation_results.txt" 2>/dev/null
