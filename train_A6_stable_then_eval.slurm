#!/bin/bash
#SBATCH --job-name=A6_stbl_eval
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/A6_stable_then_eval_%A_%a.out
#SBATCH --error=logs/A6_stable_then_eval_%A_%a.err

# ============================================================
#  Retrain A6 stable (seeds 44/45) and immediately run full
#  FAVC 70/30 evaluation using the best FV-RA checkpoint
#  saved by FAVCFvraCallback.
#
#  Array index → seed:
#    0 → seed=44  (FV-RA peak observed at epoch 34, AUC=0.8014)
#    1 → seed=45  (FV-RA peak observed at epoch 49, AUC=0.8060)
# ============================================================

SEEDS=(44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_full_bestfvra"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  A6 stable retrain + eval | seed=${SEED}"
echo "  Output  : $OUTPUT_DIR"
echo "  Results : $RESULTS_DIR"
echo "  Start   : $(date)"
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

mkdir -p logs
mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results}
# Clear existing checkpoints so we get a clean deterministic run
rm -f "$OUTPUT_DIR"/ckpts/model-epoch=*.ckpt
rm -rf "$OUTPUT_DIR"/ckpts/favc_fvra
mkdir -p "$OUTPUT_DIR/ckpts/favc_fvra"

echo ""
echo "=== Phase 1: Training ==="
cd avh_sup

"$PYTHON" train_test_fcd_a6.py \
  --config_path "configs/A6_stable.yaml" \
  --mode train \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=== Training done at $(date) ==="

# Pick the best FV-RA checkpoint saved by FAVCFvraCallback
FVRA_CKPT=$(ls "$OUTPUT_DIR/ckpts/favc_fvra/best-favc-fvra-epoch="*.ckpt 2>/dev/null | sort | tail -1)

if [[ -z "$FVRA_CKPT" ]]; then
  echo "WARNING: No best-FV-RA checkpoint found under ckpts/favc_fvra/"
  echo "Falling back to best val_auc_causal checkpoint..."
  FVRA_CKPT=$(ls "$OUTPUT_DIR"/ckpts/model-epoch=*.ckpt 2>/dev/null | sort | tail -1)
fi

if [[ -z "$FVRA_CKPT" ]]; then
  echo "ERROR: No checkpoint found at all" >&2
  exit 1
fi

echo "Best FV-RA checkpoint: $FVRA_CKPT"

echo ""
echo "=== Phase 2: Full FAVC 70/30 Evaluation ==="
mkdir -p "$RESULTS_DIR"

"$PYTHON" eval_favc.py \
  --ckpt "$FVRA_CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: seed=${SEED} at $(date)"
echo "  Results: $RESULTS_DIR"
echo "============================================================"
