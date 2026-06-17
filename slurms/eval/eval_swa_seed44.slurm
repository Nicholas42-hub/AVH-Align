#!/bin/bash
#SBATCH --job-name=swa_s44_eval
#SBATCH --partition=gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/eval_swa_seed44_%j.out
#SBATCH --error=logs/eval_swa_seed44_%j.err

# ============================================================
#  FAVC 7030_full eval on SWA(epochs 40-50) ckpt for seed 44
#
#  Input ckpt:  outputs_A6_av1m_pseudodomain_v2_50ep_seed44_SWA/ckpts/swa_e40_50.ckpt
#  Output:      outputs_A6_av1m_pseudodomain_v2_50ep_seed44_SWA/results_favc_7030_full/
#
#  Does NOT touch any existing outputs_A6_av1m_pseudodomain_v2_50ep_seed44/ files.
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
SWA_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed44_SWA"
CKPT="$SWA_DIR/ckpts/swa_e40_50.ckpt"
OUT_DIR="$SWA_DIR/results_favc_7030_full"

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

cd avh_sup

mkdir -p "$OUT_DIR"

$PYTHON eval_favc.py \
    --ckpt "$CKPT" \
    --model fcd_a6 \
    --features_path "$BASE/data/favc_features" \
    --csv_root_path "csv_metadata/favc_7030_canonical" \
    --split test \
    --output_dir "$OUT_DIR"

echo ""
echo "============================================================"
echo "  Eval done. Results: $OUT_DIR/eval_results.txt"
echo "============================================================"
