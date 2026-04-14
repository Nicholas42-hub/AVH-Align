#!/bin/bash
#SBATCH --job-name=ablation_ms
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/ablation_ms_%A_%a.out
#SBATCH --error=logs/ablation_ms_%A_%a.err

# ============================================================
#  Multi-seed ablation re-training: A0, A1, A2 × seeds 42,43,44
#
#  Array task → (model, seed):
#    0→(A0,42)  1→(A0,43)  2→(A0,44)
#    3→(A1,42)  4→(A1,43)  5→(A1,44)
#    6→(A2,42)  7→(A2,43)  8→(A2,44)
# ============================================================

MODELS=(A0 A0 A0 A1 A1 A1 A2 A2 A2)
SEEDS=(42 43 44 42 43 44 42 43 44)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_${MODEL}_seed${SEED}"

echo "============================================================"
echo "  Ablation ${MODEL} — seed=${SEED}"
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
    --config_path "configs/${MODEL}.yaml" \
    --mode        train \
    --seed        "$SEED" \
    --output_dir  "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  ${MODEL} seed=${SEED} training done: $(date)"
echo "============================================================"
