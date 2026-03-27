#!/bin/bash
#SBATCH --job-name=perturb_mi
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/perturb_mi_%j.out
#SBATCH --error=logs/perturb_mi_%j.err

# ============================================================
#  Perturbation Robustness — CLUB-MI model
# ============================================================

echo "============================================================"
echo "  Perturbation Robustness: CLUB-MI"
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

mkdir -p logs avh_sup/outputs_causal_mi/results

cd avh_sup

MI_CKPT=$(ls outputs_causal_mi/ckpts/*.ckpt 2>/dev/null | grep -v ".bak" | sort | tail -1)
if [[ -z "$MI_CKPT" ]]; then
    echo "ERROR: no checkpoint found in outputs_causal_mi/ckpts/"
    exit 1
fi
echo "Checkpoint: $MI_CKPT"

python eval_perturbation_robustness.py \
    --ckpt_path      "$MI_CKPT" \
    --model          mi \
    --config         configs/A2.yaml \
    --favc_root      ../data/favc_features \
    --csv_root_path  csv_metadata/favc \
    --split          test \
    --output_dir     outputs_causal_mi/results \
    --device         cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"

echo ""
echo "=== FULL PERTURBATION COMPARISON ==="
python3 - <<'PYEOF'
import json, os

models = {
    "baseline": "AVH-Align (paper)",
    "A0":       "A0 (orth only)",
    "A1":       "A1 (GRL)",
    "A2":       "A2 (GRL+SG)",
    "causal_mi":"CLUB-MI",
}
ptypes = ["audio_noise", "visual_noise", "audio_masking", "visual_masking", "temporal_shuffle"]

for key, label in models.items():
    path = f"outputs_{key}/results/perturbation_robustness.json"
    if not os.path.exists(path):
        print(f"\n  {label}: no results")
        continue
    d  = json.load(open(path))
    pt = d["perturb_types"]
    clean_c = pt["audio_noise"]["0.0"]["causal_auc"]
    clean_s = pt["audio_noise"]["0.0"]["spurious_auc"]
    print(f"\n  {label}  (clean: Z_c={clean_c:.4f}, Z_s={clean_s:.4f})")
    print(f"    {'Perturbation':<22}  {'Z_c (worst)':>12}  {'Δ Z_c':>8}")
    for ptype in ptypes:
        if ptype not in pt:
            continue
        levels = sorted(pt[ptype].keys(), key=float)
        non_zero = [l for l in levels if float(l) > 0]
        if not non_zero:
            continue
        lv  = non_zero[-1]
        zc  = pt[ptype][lv]["causal_auc"]
        print(f"    {ptype:<22}  {zc:>12.4f}  {zc-clean_c:>+8.4f}")
PYEOF
