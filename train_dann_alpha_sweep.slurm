#!/bin/bash
#SBATCH --job-name=dann_alpha_sweep
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:50:00
#SBATCH --array=0-4
#SBATCH --output=logs/dann_alpha_sweep_%A_%a.out
#SBATCH --error=logs/dann_alpha_sweep_%A_%a.err

# ────────────────────────────────────────────────────────────────────
#  DANN GRL-α sensitivity sweep (single seed = 44).
#
#  Goal: show that no α in {0.1, 0.3, 1.0, 3.0, 10.0} closes the
#  AV1M→FAVC FV-RA gap to TriRoute (0.838 AUC).
#
#  Each array task: 1 α value, 20 epochs, seed 44.
#  Output: avh_sup/outputs_DANN_alpha${A}_seed44/
# ────────────────────────────────────────────────────────────────────

ALPHAS=(0.1 0.3 1.0 3.0 10.0)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
SEED=44
METHOD=dann
CONFIG="configs/DANN_alpha${ALPHA}.yaml"
OUT="$BASE/avh_sup/outputs_DANN_alpha${ALPHA}_seed${SEED}"

echo "=== DANN α=${ALPHA} seed=${SEED} on $(hostname) ==="
echo "Config: ${CONFIG}"
echo "Out:    ${OUT}"
echo "Start:  $(date)"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

cd avh_sup
python train_da_baselines.py \
    --method       "$METHOD" \
    --config_path  "$CONFIG" \
    --seed         "$SEED" \
    --output_dir   "$OUT"

CKPT=$(ls "$OUT/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then echo "ERROR: no checkpoint for α=${ALPHA}" >&2; exit 1; fi

python eval_favc.py \
    --ckpt          "$CKPT" \
    --model         "$METHOD" \
    --features_path "$FEATURES" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$OUT/results_favc_7030"

echo "=== α=${ALPHA} done: $(date) ==="
echo "--- FV-RA summary ---"
grep -E "AUC|FakeVideo-RealAudio|Overall" "$OUT/results_favc_7030/eval_results.txt" | head -20 || true
