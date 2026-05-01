#!/bin/bash
#SBATCH --job-name=infer_testB_smoke
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/data/projects/punim2637/nnliang/logs/infer_testB_smoke_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/logs/infer_testB_smoke_%j.err

echo "============================================================"
echo "  Smoke test: inference on first 100 testB clips"
echo "  Job  : $SLURM_JOB_ID"
echo "  Node : $SLURM_NODELIST"
echo "  Start: $(date)"
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

CKPT="$BASE/avh_sup/outputs_A6_e2e/ckpts/model-epoch=11-val_auc_causal=1.0000.ckpt"
OUTPUT="$BASE/avh_sup/outputs_A6_e2e/results/prediction_testB_smoke.txt"
mkdir -p "$(dirname $OUTPUT)"

$PYTHON "$BASE/infer_testB.py" \
    --ckpt         "$CKPT" \
    --config       "$BASE/avh_sup/configs/A6_e2e.yaml" \
    --preprocessed_dir "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus/testB/preprocessed" \
    --output       "$OUTPUT" \
    --batch_size   8 \
    --num_workers  4 \
    --limit        100

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Lines in smoke output: $(wc -l < $OUTPUT 2>/dev/null || echo 0)"
echo "  First 5 lines:"
head -5 "$OUTPUT" 2>/dev/null
echo "============================================================"
