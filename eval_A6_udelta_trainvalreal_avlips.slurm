#!/bin/bash
#SBATCH --job-name=eval_A6udelta_avlips
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_A6_udelta_trainvalreal_avlips_%j.out
#SBATCH --error=logs/eval_A6_udelta_trainvalreal_avlips_%j.err

echo "============================================================"
echo "  A6-udelta-trainvalreal eval on AVLips"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

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

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVLIPS_FEAT="$BASE/data/avlips_features"
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_udelta_trainvalreal"
OUT_DIR="$OUTPUT_DIR/results_avlips"

mkdir -p "$OUT_DIR" logs

# save_top_k=1: only one ckpt exists at a time and it is always the best
CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | grep -v last | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "No checkpoint found under $OUTPUT_DIR/ckpts" >&2
    exit 1
fi
echo "Using checkpoint: $CKPT"

cd avh_sup

python eval_avlips.py \
    --ckpt          "$CKPT" \
    --model         fcd_a6 \
    --features_path "$AVLIPS_FEAT" \
    --output_dir    "$OUT_DIR" \
    --device        cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"

echo ""
cat "$OUT_DIR/eval_results.txt" 2>/dev/null || echo "(no results yet)"
