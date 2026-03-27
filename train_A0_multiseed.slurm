#!/bin/bash
#SBATCH --job-name=A0_ms
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/A0_ms_%A_%a.out
#SBATCH --error=logs/A0_ms_%A_%a.err

# A0 re-run with fixed loss (full_head only, no secondary causal_head)
# Array: 0→seed42  1→seed43  2→seed44

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A0_seed${SEED}"

echo "============================================================"
echo "  A0 (fixed loss) — seed=${SEED}"
echo "  Job ID    : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node      : $SLURM_NODELIST"
echo "  Output    : $OUTPUT_DIR"
echo "  Start     : $(date)"
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

mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_ablation.py \
    --config_path "configs/A0.yaml" \
    --mode        train \
    --seed        "$SEED" \
    --output_dir  "$OUTPUT_DIR"

echo "============================================================"
echo "  A0 seed=${SEED} done: $(date)"
echo "============================================================"
