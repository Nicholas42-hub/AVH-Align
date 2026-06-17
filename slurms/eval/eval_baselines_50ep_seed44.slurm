#!/bin/bash
#SBATCH --job-name=eval_bl50s44
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-4
#SBATCH --output=logs/eval_bl50s44_%A_%a.out
#SBATCH --error=logs/eval_bl50s44_%A_%a.err

# ============================================================
#  FakeAVCeleb 70/30 full eval for 5 baselines retrained at
#  50 epochs / seed 44.
#  Array: 0=A2, 1=DANN, 2=CORAL, 3=A5, 4=A3
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

MODELS=(A2 DANN CORAL A5 A3)
MODEL_TYPES=(ablation dann coral fcd ablation)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MTYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="$BASE/avh_sup/outputs_${MODEL}_50ep_seed44/ckpts"
CKPT=$(ls "$CKPT_DIR"/*.ckpt | sort | tail -1)
RESULTS_DIR="$BASE/avh_sup/outputs_${MODEL}_50ep_seed44/results_favc7030_full"

echo "============================================================"
echo "  Eval: $MODEL (--model $MTYPE)"
echo "  Ckpt: $CKPT"
echo "  Out : $RESULTS_DIR"
echo "  Job : $SLURM_JOB_ID  Array: $SLURM_ARRAY_TASK_ID"
echo "  Start: $(date)"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$RESULTS_DIR" logs

cd avh_sup

$PYTHON eval_favc.py \
    --ckpt          "$CKPT" \
    --model         "$MTYPE" \
    --features_path "$FEATURES" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: $MODEL at $(date)"
echo "============================================================"
