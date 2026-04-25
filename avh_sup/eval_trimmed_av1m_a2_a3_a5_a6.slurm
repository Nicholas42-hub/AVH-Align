#!/bin/bash
#SBATCH --job-name=trim_av1m_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/trim_av1m_eval_%j.out
#SBATCH --error=logs/trim_av1m_eval_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
SUP_DIR="$BASE/avh_sup"
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
AVHUB_CKPT="$BASE/av_hubert/avhubert/self_large_vox_433h.pt"

AV1M_PREP_ROOT=/scratch/punim2637/nnliang/avd1m_preprocessed
AV1M_TRIMMED_ROOT=/scratch/punim2637/nnliang/avd1m_features_trimmed
AV1M_UNTRIMMED_ROOT="$BASE/data/avh_features"
AV1M_CSV_ROOT="$SUP_DIR/csv_metadata/av1m"
AV1M_VAL_METADATA="$AV1M_CSV_ROOT/val_labels.csv"

mkdir -p "$BASE/logs" "$AV1M_TRIMMED_ROOT"

echo "============================================================"
echo "  Trimmed AV1M evaluation for A2 / A3 / A5 / A6"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
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

echo ""
echo ">>> Step 1/3: extract trimmed AV1M val features"
echo "    raw preprocessed root : $AV1M_PREP_ROOT"
echo "    metadata             : $AV1M_VAL_METADATA"
echo "    output root          : $AV1M_TRIMMED_ROOT"
echo ""

$PYTHON "$BASE/deepfake_feature_extraction.py" \
  --dataset AV1M \
  --split test \
  --metadata "$AV1M_VAL_METADATA" \
  --ckpt_path "$AVHUB_CKPT" \
  --data_path "$AV1M_PREP_ROOT" \
  --save_path "$AV1M_TRIMMED_ROOT" \
  --trimmed

echo ""
echo ">>> Trimmed NPZ count: $(find "$AV1M_TRIMMED_ROOT/val" -name '*.npz' | wc -l)"
echo ""

run_eval () {
  local model_id="$1"
  local model_type="$2"
  local ckpt_path="$3"
  local features_root="$4"
  local suffix="$5"

  echo "------------------------------------------------------------"
  echo "Model   : $model_id"
  echo "Type    : $model_type"
  echo "Ckpt    : $ckpt_path"
  echo "Features: $features_root"
  echo "Suffix  : $suffix"
  echo "------------------------------------------------------------"

  $PYTHON "$SUP_DIR/eval_av1m.py" \
    --ckpt "$ckpt_path" \
    --model "$model_type" \
    --features_path "$features_root" \
    --csv_root_path "$AV1M_CSV_ROOT" \
    --split val \
    --output_dir "$SUP_DIR/outputs_${model_id}/results_av1m_val_${suffix}" \
    --device cuda
}

echo ">>> Step 2/3: evaluate original AV1M val features"
run_eval A2 ablation "$SUP_DIR/outputs_A2/ckpts/model-epoch=09.ckpt" "$AV1M_UNTRIMMED_ROOT" untrimmed
run_eval A3 ablation "$SUP_DIR/outputs_A3/ckpts/model-epoch=10.ckpt" "$AV1M_UNTRIMMED_ROOT" untrimmed
run_eval A5 fcd      "$SUP_DIR/outputs_A5/ckpts/model-epoch=17.ckpt" "$AV1M_UNTRIMMED_ROOT" untrimmed
run_eval A6 fcd_a6   "$SUP_DIR/outputs_A6/ckpts/model-epoch=15.ckpt" "$AV1M_UNTRIMMED_ROOT" untrimmed

echo ""
echo ">>> Step 3/3: evaluate trimmed AV1M val features"
run_eval A2 ablation "$SUP_DIR/outputs_A2/ckpts/model-epoch=09.ckpt" "$AV1M_TRIMMED_ROOT" trimmed
run_eval A3 ablation "$SUP_DIR/outputs_A3/ckpts/model-epoch=10.ckpt" "$AV1M_TRIMMED_ROOT" trimmed
run_eval A5 fcd      "$SUP_DIR/outputs_A5/ckpts/model-epoch=17.ckpt" "$AV1M_TRIMMED_ROOT" trimmed
run_eval A6 fcd_a6   "$SUP_DIR/outputs_A6/ckpts/model-epoch=15.ckpt" "$AV1M_TRIMMED_ROOT" trimmed

echo ""
echo "============================================================"
echo "  Done at $(date)"
echo "  Result files:"
echo "    $SUP_DIR/outputs_A2/results_av1m_val_untrimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A2/results_av1m_val_trimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A3/results_av1m_val_untrimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A3/results_av1m_val_trimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A5/results_av1m_val_untrimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A5/results_av1m_val_trimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A6/results_av1m_val_untrimmed/eval_results.txt"
echo "    $SUP_DIR/outputs_A6/results_av1m_val_trimmed/eval_results.txt"
echo "============================================================"
