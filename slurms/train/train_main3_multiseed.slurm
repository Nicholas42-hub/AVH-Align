#!/bin/bash
#SBATCH --job-name=main3_ms
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/main3_ms_%A_%a.out
#SBATCH --error=logs/main3_ms_%A_%a.err

# ============================================================
#  Main frozen-model seed sweep: A2 / A5 / A6 × seeds 43,44,45
# ============================================================

MODELS=(A2 A2 A2 A5 A5 A5 A6 A6 A6)
SEEDS=(43 44 45 43 44 45 43 44 45)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_${MODEL}_seed${SEED}"

echo "============================================================"
echo "  Main3 seed sweep | ${MODEL} seed=${SEED}"
echo "  Job ID : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node   : $SLURM_NODELIST"
echo "  Output : $OUTPUT_DIR"
echo "  Start  : $(date)"
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

mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup

if [[ "$MODEL" == "A2" ]]; then
  python train_test_causal_ablation.py \
    --config_path configs/A2.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
elif [[ "$MODEL" == "A5" ]]; then
  python train_test_fcd.py \
    --config_path configs/A5.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
elif [[ "$MODEL" == "A6" ]]; then
  python train_test_fcd_a6.py \
    --config_path configs/A6.yaml \
    --mode train \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"
else
  echo "Unknown model: $MODEL" >&2
  exit 1
fi

echo ""
echo "============================================================"
echo "  Done: ${MODEL} seed=${SEED} at $(date)"
echo "============================================================"
