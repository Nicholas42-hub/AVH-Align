#!/bin/bash
#SBATCH --job-name=baselines_50ep_s4345
#SBATCH --partition=gpu-l40s
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/baselines_50ep_s4345_%A_%a.out
#SBATCH --error=logs/baselines_50ep_s4345_%A_%a.err

# ============================================================
#  Baseline retrain — 50 epochs, seeds 43/45 (seed 44 already
#  done). Backfills error bars for fig_appendix_table1.
#
#  Array index → (model, seed):
#    0 A2    seed43      4 A2    seed45
#    1 A5    seed43      5 A5    seed45
#    2 DANN  seed43      6 DANN  seed45
#    3 CORAL seed43      7 CORAL seed45
# ============================================================

MODELS=(A2 A5 DANN CORAL A2 A5 DANN CORAL)
SEEDS=(43 43 43 43 45 45 45 45)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  Baseline 50ep | model=${MODEL} seed=${SEED}"
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

OUT="$BASE/avh_sup/outputs_${MODEL}_50ep_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results} logs

FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

cd avh_sup

# --- Train ---
case "$MODEL" in
  A2)
    "$PYTHON" train_test_causal_ablation.py \
      --config_path "configs/A2_50ep_seed${SEED}.yaml" \
      --mode train
    EVAL_MODEL="ablation"
    ;;
  A5)
    "$PYTHON" train_test_fcd.py \
      --config_path "configs/A5_50ep_seed${SEED}.yaml" \
      --mode train
    EVAL_MODEL="fcd"
    ;;
  DANN)
    "$PYTHON" train_da_baselines.py \
      --method       dann \
      --config_path  "configs/DANN_50ep_seed${SEED}.yaml" \
      --seed         $SEED \
      --output_dir   "$OUT"
    EVAL_MODEL="dann"
    ;;
  CORAL)
    "$PYTHON" train_da_baselines.py \
      --method       coral \
      --config_path  "configs/CORAL_50ep_seed${SEED}.yaml" \
      --seed         $SEED \
      --output_dir   "$OUT"
    EVAL_MODEL="coral"
    ;;
esac

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
  --model         "$EVAL_MODEL" \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split         test \
  --output_dir    "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: ${MODEL} 50ep seed=${SEED} at $(date)"
echo "============================================================"
