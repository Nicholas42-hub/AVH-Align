#!/bin/bash
#SBATCH --job-name=A6_nosda_s42
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/A6_trainvalreal_nosda_seed42_%j.out
#SBATCH --error=logs/A6_trainvalreal_nosda_seed42_%j.err

# ============================================================
#  A6-trainvalreal, lambda_sda=0 ablation, seed 42
#
#  Identical to A6_trainvalreal_seed42 except lambda_sda=0.
#  Purpose: isolate whether the SDA (alignment) loss contributes
#  to FV-RA gain in the three-way setting.
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_trainvalreal_nosda_seed42"
RESULTS_DIR="$OUTPUT_DIR/results_favc_matched"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  A6-trainvalreal nosda seed42 — train + eval"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup

echo ""
echo "--- Training ---"
$PYTHON train_test_fcd_a6_trainvalreal.py \
  --config_path configs/A6_trainvalreal_nosda_seed42.yaml

echo ""
echo "--- Eval on FAVC matched ---"
mkdir -p "$RESULTS_DIR"
CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi

$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
