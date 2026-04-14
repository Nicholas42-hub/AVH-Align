#!/bin/bash
#SBATCH --job-name=eval_avlips_A6
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_avlips_A6_%j.out
#SBATCH --error=logs/eval_avlips_A6_%j.err

echo "============================================================"
echo "  A6 Evaluation — AVLips (seeds 42, 43, 44)"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVLIPS_FEAT="$BASE/data/avlips_features"

mkdir -p "$BASE/logs"

cd avh_sup

for SEED in 42 43 44; do
    CKPT=$(ls outputs_A6_seed${SEED}/ckpts/*.ckpt 2>/dev/null | grep -v last | sort | tail -1)
    CKPT="${CKPT:-$(ls outputs_A6_seed${SEED}/ckpts/last.ckpt 2>/dev/null)}"

    if [[ -z "$CKPT" ]]; then
        echo ">>> [seed${SEED}] No checkpoint found, skipping."
        continue
    fi

    OUT_DIR="outputs_A6_seed${SEED}/results_avlips"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "=== A6 | AVLips | seed${SEED} ==="
    echo ">>> Checkpoint: $CKPT"

    $PYTHON eval_avlips.py \
        --ckpt          "$CKPT" \
        --model         fcd_a6 \
        --features_path "$AVLIPS_FEAT" \
        --output_dir    "$OUT_DIR" \
        --device        cuda

    echo ">>> Done seed${SEED}: $(date)"
done

echo ""
echo "============================================================"
echo "  A6 AVLips eval complete: $(date)"
echo "============================================================"

for SEED in 42 43 44; do
    echo ""
    echo "--- A6 seed${SEED} ---"
    cat "outputs_A6_seed${SEED}/results_avlips/eval_results.txt" 2>/dev/null || echo "(no results)"
done
