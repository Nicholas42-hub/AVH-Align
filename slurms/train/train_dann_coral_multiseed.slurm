#!/bin/bash
#SBATCH --job-name=dann_coral_train
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dann_coral_train_%j.out
#SBATCH --error=logs/dann_coral_train_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"

echo "============================================================"
echo "  DANN-UDA + CORAL-UDA  (seeds 43-45)"
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

cd avh_sup

# ── Helper: find best checkpoint ────────────────────────────────────────────
find_ckpt() {
    local dir="$1"
    ls "$dir"/ckpts/*.ckpt 2>/dev/null | sort | tail -1
}

# ── Helper: eval on FAVC ────────────────────────────────────────────────────
eval_favc() {
    local ckpt="$1"
    local model_type="$2"   # dann or coral
    local results_dir="$3"
    mkdir -p "$results_dir"
    python eval_favc.py \
        --ckpt          "$ckpt" \
        --model         "$model_type" \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split         test \
        --output_dir    "$results_dir"
}

# ════════════════════════════════════════════════════════════════
#  DANN-UDA  (seeds 43, 44, 45)
# ════════════════════════════════════════════════════════════════
for SEED in 43 44 45; do
    OUT="$BASE/avh_sup/outputs_DANN_seed${SEED}"
    echo ""
    echo "--- DANN seed=${SEED} ---"
    python train_da_baselines.py \
        --method       dann \
        --config_path  configs/DANN.yaml \
        --seed         "$SEED" \
        --output_dir   "$OUT"

    CKPT=$(find_ckpt "$OUT")
    if [[ -z "$CKPT" ]]; then
        echo "ERROR: No DANN checkpoint for seed $SEED" >&2
        continue
    fi
    eval_favc "$CKPT" "dann" "$OUT/results_favc_7030"
    echo "  DANN seed=$SEED done."
done

# ════════════════════════════════════════════════════════════════
#  CORAL-UDA  (seeds 43, 44, 45)
# ════════════════════════════════════════════════════════════════
for SEED in 43 44 45; do
    OUT="$BASE/avh_sup/outputs_CORAL_seed${SEED}"
    echo ""
    echo "--- CORAL seed=${SEED} ---"
    python train_da_baselines.py \
        --method       coral \
        --config_path  configs/CORAL.yaml \
        --seed         "$SEED" \
        --output_dir   "$OUT"

    CKPT=$(find_ckpt "$OUT")
    if [[ -z "$CKPT" ]]; then
        echo "ERROR: No CORAL checkpoint for seed $SEED" >&2
        continue
    fi
    eval_favc "$CKPT" "coral" "$OUT/results_favc_7030"
    echo "  CORAL seed=$SEED done."
done

# ════════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════════
echo ""
echo "============================================================"
echo "  All training+eval complete: $(date)"
echo "============================================================"
echo ""
echo "=== DANN results ==="
for SEED in 43 44 45; do
    RFILE="$BASE/avh_sup/outputs_DANN_seed${SEED}/results_favc_7030/eval_results.txt"
    if [[ -f "$RFILE" ]]; then
        echo "--- DANN seed=$SEED ---"
        grep -E "AUC|AP|FV-RA|RV-FA|FV-FA|Overall" "$RFILE" | head -15
    fi
done
echo ""
echo "=== CORAL results ==="
for SEED in 43 44 45; do
    RFILE="$BASE/avh_sup/outputs_CORAL_seed${SEED}/results_favc_7030/eval_results.txt"
    if [[ -f "$RFILE" ]]; then
        echo "--- CORAL seed=$SEED ---"
        grep -E "AUC|AP|FV-RA|RV-FA|FV-FA|Overall" "$RFILE" | head -15
    fi
done
