#!/bin/bash
#SBATCH --job-name=A3_50ep_s4345
#SBATCH --partition=gpu-l40s
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/A3_50ep_s4345_%A_%a.out
#SBATCH --error=logs/A3_50ep_s4345_%A_%a.err

# ============================================================
#  A3 (Two-way + GRL) retrain — 50 epochs, seeds 43/45.
#  seed 44 already done in outputs_A3_50ep_seed44/.
#  Backfills error bars for the dagger row of Tab 1 in the paper.
#
#  Array index → seed:
#    0 -> 43
#    1 -> 45
# ============================================================

SEEDS=(43 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  A3 50ep | seed=${SEED}"
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

OUT="$BASE/avh_sup/outputs_A3_50ep_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results} logs

FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

cd avh_sup

# --- Train ---
"$PYTHON" train_test_causal_ablation.py \
  --config_path "configs/A3_50ep_seed${SEED}.yaml" \
  --mode train

# --- Eval on FAVC 7030 full test (N=6667) ---
RESULTS_DIR="$OUT/results_favc7030_full"
mkdir -p "$RESULTS_DIR"

CKPT=$(ls "$OUT/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint under $OUT/ckpts" >&2
  exit 1
fi
echo "  Eval ckpt: $CKPT"

"$PYTHON" eval_favc.py \
  --ckpt          "$CKPT" \
  --model         "ablation" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split         test \
  --output_dir    "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: A3 50ep seed=${SEED} at $(date)"
echo "============================================================"
