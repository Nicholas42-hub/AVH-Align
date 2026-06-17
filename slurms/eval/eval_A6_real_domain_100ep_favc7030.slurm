#!/bin/bash
#SBATCH --job-name=A6_realeval
#SBATCH --partition=deeplearn
#SBATCH --account=punim2637
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=43,44,45
#SBATCH --output=logs/A6_real_eval_s%a_%A.out
#SBATCH --error=logs/A6_real_eval_s%a_%A.err

# ============================================================
#  Eval A6 real-target-domain 100ep on FAVC 70/30 canonical (N=6667).
#  Picks the best-favc-fvra ckpt to match pseudodomain_v2_100ep protocol.
# ============================================================

SEED=$SLURM_ARRAY_TASK_ID

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_real_domain_100ep_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
FEATURES="$BASE/data/favc_features"

echo "============================================================"
echo "  A6 real-domain 100ep FAVC eval | seed=${SEED}"
echo "  Output  : $OUTPUT_DIR"
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

mkdir -p "$RESULTS_DIR" logs

# Prefer FAVC-FV-RA-best ckpt; fall back to last training ckpt.
CKPT=$(ls "$OUTPUT_DIR"/ckpts/favc_fvra/best-favc-fvra-epoch=*.ckpt 2>/dev/null | sort -t= -k3 -n | tail -1)
if [[ -z "$CKPT" ]]; then
  CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
fi
if [[ -z "$CKPT" ]]; then
  echo "ERROR: No checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi
echo "  Ckpt    : $CKPT"

cd avh_sup

python eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Done: A6 real-domain 100ep seed=${SEED} at $(date)"
echo "============================================================"
