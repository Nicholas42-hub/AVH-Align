#!/bin/bash
#SBATCH --job-name=pdv2_50ep_pure
#SBATCH --partition=gpu-l40s
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/pdv2_50ep_pure_%A_%a.out
#SBATCH --error=logs/pdv2_50ep_pure_%A_%a.err

# ============================================================
#  TriRoute (A6_av1m_pseudodomain_v2) 50-epoch retrain.
#
#  Pure source-val ckpt selection — no favc_eval_info in config,
#  so no FAVC test/val data is touched during training. Compliant
#  with paper-stated protocol (ckpt by AV1M val_auc_causal).
#
#  Array index → seed:
#    0 → 43
#    1 → 44
#    2 → 45
# ============================================================

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  pdv2 50ep PURE | seed=${SEED}"
echo "  Array task : $SLURM_ARRAY_TASK_ID"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  Start      : $(date)"
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

OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results} logs

FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

cd avh_sup

# --- Train (50 epochs, source-val ckpt selection only) ---
"$PYTHON" train_test_fcd_a6_av1m_pseudodomain_v2.py \
  --config_path "configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml"

# --- Eval on FAVC 7030 full test (N=6667), source-val ckpt ---
RESULTS_DIR="$OUT/results_favc_7030_full"
mkdir -p "$RESULTS_DIR"

CKPT=$(ls "$OUT/ckpts/model-epoch="*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no source-val checkpoint under $OUT/ckpts" >&2
  exit 1
fi
echo "  Eval ckpt (source-val): $CKPT"

"$PYTHON" eval_favc.py \
  --ckpt          "$CKPT" \
  --model         "fcd_a6" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split         test \
  --output_dir    "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: pdv2 50ep PURE seed=${SEED} at $(date)"
echo "============================================================"
