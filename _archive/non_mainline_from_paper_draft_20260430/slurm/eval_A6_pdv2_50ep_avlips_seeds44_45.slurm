#!/bin/bash
#SBATCH --job-name=eval_pdv2_avl
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/eval_pdv2_50ep_avlips_%A_%a.out
#SBATCH --error=logs/eval_pdv2_50ep_avlips_%A_%a.err

SEEDS=(44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  A6 pseudo-domain v2 50ep eval on AVLips | seed=${SEED}"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

AVLIPS_FEAT="$BASE/data/avlips_features"
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
BEST_FILE="$OUTPUT_DIR/ckpts/favc_fvra/best_favc_fvra.txt"
OUT_DIR="$OUTPUT_DIR/results_avlips_pdv2_50ep"

mkdir -p "$OUT_DIR" logs

if [[ ! -f "$BEST_FILE" ]]; then
  echo "Missing best checkpoint pointer: $BEST_FILE"
  exit 1
fi

CKPT=$(awk '/^checkpoint=/{sub(/^checkpoint=/, ""); print}' "$BEST_FILE")
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "Invalid checkpoint from $BEST_FILE: $CKPT"
  exit 1
fi

echo "Checkpoint: $CKPT"
echo "Output    : $OUT_DIR"

cd avh_sup

"$PYTHON" eval_avlips.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$AVLIPS_FEAT" \
  --output_dir "$OUT_DIR" \
  --device cuda

echo ""
echo "============================================================"
echo "  Done: A6 pseudo-domain v2 50ep AVLips seed=${SEED} at $(date)"
echo "============================================================"
cat "$OUT_DIR/eval_results.txt"
