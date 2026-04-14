#!/bin/bash
#SBATCH --job-name=eval_causal_mi
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_all_causal_mi_%j.out
#SBATCH --error=logs/eval_all_causal_mi_%j.err

# ============================================================
#  Causal-MI — Full Evaluation Suite
#
#  1. Cross-domain AUC (direct trained heads on FakeAVCeleb)
#       full / causal / spurious heads evaluated on FAVC test
#       Key: spurious AUC near 0.5 → disentanglement successful
#
#  2. Label probe (logistic regression on Z_c / Z_s)
#       In-domain (AV1M val, 5-fold CV)
#       OOD transfer (train AV1M → test FAVC)
#
#  3. Domain probe (AV1M=0 vs FAVC=1, logistic regression)
#       Z_s domain AUC < 0.65 target → spurious branch is
#       domain-insensitive
#
#  Checkpoint used: best val_auc_causal (epoch 08)
#  Training AUC summary: full≈0.997, causal≈0.998, spurious≈0.505
# ============================================================

echo "============================================================"
echo "  Causal-MI Full Eval (cross + label probe + domain probe)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

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
BASE=/data/projects/punim2637/nnliang/AVH-Align

# ── Checkpoint selection (most recent, non-last) ─────────────────────────────
CKPT_DIR="$BASE/avh_sup/outputs_causal_mi/ckpts"
CKPT=$(ls -t "$CKPT_DIR"/model-epoch=*.ckpt 2>/dev/null | grep -v "v1" | head -1)
CKPT="${CKPT:-$CKPT_DIR/model-epoch=08.ckpt}"
echo ">>> Using checkpoint: $CKPT"

# ── Paths ─────────────────────────────────────────────────────────────────────
AV1M_ROOT="$BASE/data/avh_features/val"
AV1M_CSV="$BASE/avh_sup/csv_metadata/av1m/val_labels.csv"
FAVC_ROOT="$BASE/data/favc_features"
FAVC_CSV="$BASE/avh_sup/csv_metadata/favc"
OUT_DIR="$BASE/avh_sup/outputs_causal_mi/results"

mkdir -p "$OUT_DIR" logs

cd "$BASE/avh_sup"

# ============================================================
#  1. Cross-domain AUC — direct trained heads on FAVC (test)
# ============================================================
echo ""
echo "============================================================"
echo "  1. Cross-domain AUC (direct model heads on FakeAVCeleb)"
echo "============================================================"

$PYTHON eval_favc.py \
    --ckpt           "$CKPT" \
    --model          causal_mi \
    --features_path  "$FAVC_ROOT" \
    --csv_root_path  "$FAVC_CSV" \
    --split          test \
    --output_dir     "$OUT_DIR"

echo ""
cat "$OUT_DIR/eval_results.txt" 2>/dev/null || echo "(no results file yet)"

# ============================================================
#  2. Label probe + domain probe
#     (logistic regression on frozen Z_c / Z_s)
# ============================================================
echo ""
echo "============================================================"
echo "  2. Label Probe + Domain Probe (Z_c vs Z_s)"
echo "============================================================"

$PYTHON eval_label_probe.py \
    --ckpt_path  "$CKPT" \
    --config     configs/A2.yaml \
    --av1m_root  "$AV1M_ROOT" \
    --av1m_csv   "$AV1M_CSV" \
    --favc_root  "$FAVC_ROOT" \
    --n_folds    5 \
    --seed       42 \
    --output_dir "$OUT_DIR" \
    --device     cuda

# ============================================================
#  3. Summary
# ============================================================
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"

$PYTHON - <<'PYEOF'
import json, os

base = "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_causal_mi/results"

print("\n── Cross-domain AUC (direct heads on FAVC) ─────────────────")
txt = os.path.join(base, "eval_results.txt")
if os.path.exists(txt):
    with open(txt) as f:
        for line in f:
            if "overall" in line or "HEAD" in line:
                print(" ", line.rstrip())
else:
    print("  (no eval_results.txt)")

print("\n── Label Probe + Domain Probe ───────────────────────────────")
probe = os.path.join(base, "label_probe.json")
if os.path.exists(probe):
    d = json.load(open(probe))
    zc_i  = d["in_domain"]["zc"]["auc_mean"]
    zs_i  = d["in_domain"]["zs"]["auc_mean"]
    zc_o  = d["ood_transfer"]["zc"]["auc"]
    zs_o  = d["ood_transfer"]["zs"]["auc"]
    zc_d  = d["domain_probe"]["zc"]["auc_mean"]
    zs_d  = d["domain_probe"]["zs"]["auc_mean"]
    raw_d = d["domain_probe"]["raw"]["auc_mean"]
    print(f"  Label probe — in-domain   Z_c={zc_i:.4f}  Z_s={zs_i:.4f}")
    print(f"  Label probe — OOD (FAVC)  Z_c={zc_o:.4f}  Z_s={zs_o:.4f}")
    print(f"  Domain probe (AV1M vs FAVC):")
    print(f"    raw={raw_d:.4f}  Z_c={zc_d:.4f} (Δ{zc_d-raw_d:+.4f})  "
          f"Z_s={zs_d:.4f} (Δ{zs_d-raw_d:+.4f})")
    print(f"  Z_s domain-insensitive? {'YES' if zs_d < 0.65 else 'NO'} (target < 0.65)")
else:
    print("  (no label_probe.json)")
PYEOF

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "  Results in: $OUT_DIR"
echo "============================================================"
