#!/bin/bash
#SBATCH --job-name=probe_mi
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe_mi_%j.out
#SBATCH --error=logs/probe_mi_%j.err

# ============================================================
#  Label Predictability Probe — causal_mi (CLUB) vs A1 / A2
#
#  Runs eval_label_probe.py on the CLUB-based MI model and
#  re-runs A1/A2 for direct comparison.
#
#  Key question: does CLUB actually reduce Z_s label probe AUC
#  compared to GRL-only (A1) and GRL+SG (A2)?
# ============================================================

echo "============================================================"
echo "  Label Probe: causal_mi vs A1 / A2"
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

AV1M_VAL_ROOT="data/avh_features/val"
AV1M_CSV="avh_sup/csv_metadata/av1m/val_labels.csv"
FAVC_ROOT="data/favc_features"

mkdir -p logs \
    avh_sup/outputs_causal_mi/results \
    avh_sup/outputs_A1/results \
    avh_sup/outputs_A2/results

cd avh_sup

# Run probe for each model — eval_label_probe.py now auto-detects AVH_Causal_MI
# vs AVH_Causal_Ablation from the checkpoint hyper_parameters.
for MODEL in causal_mi A1 A2; do
    CKPT=$(ls outputs_${MODEL}/ckpts/*.ckpt 2>/dev/null | grep -v ".bak" | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "WARNING: no checkpoint for ${MODEL}, skipping."
        continue
    fi

    # causal_mi uses its own config stored in the checkpoint; A1/A2 use yaml files
    if [[ "$MODEL" == "causal_mi" ]]; then
        CONFIG_ARG="configs/A2.yaml"   # only used for apply_l2 flag; real config loaded from ckpt
    else
        CONFIG_ARG="configs/${MODEL}.yaml"
    fi

    echo ""
    echo "================================================================"
    echo "  Label probe: ${MODEL} | $(basename $CKPT)"
    echo "================================================================"

    python eval_label_probe.py \
        --ckpt_path   "$CKPT" \
        --config      "$CONFIG_ARG" \
        --av1m_root   "../${AV1M_VAL_ROOT}" \
        --av1m_csv    "../${AV1M_CSV}" \
        --favc_root   "../${FAVC_ROOT}" \
        --n_folds     5 \
        --seed        42 \
        --output_dir  outputs_${MODEL}/results \
        --device      cuda
done

echo ""
echo "============================================================"
echo "  Probe complete: $(date)"
echo "============================================================"

echo ""
echo "=== COMPARISON SUMMARY ==="
python3 - <<'PYEOF'
import json, os

models = ["A1", "A2", "causal_mi"]
labels = {"A1": "A1 (GRL only)", "A2": "A2 (GRL+SG)", "causal_mi": "CLUB-MI"}

print(f"\n  {'Model':<20}  {'Z_c in-dom':>11}  {'Z_s in-dom':>11}  {'Z_c OOD':>9}  {'Z_s OOD':>9}")
print(f"  {'-'*20}  {'-'*11}  {'-'*11}  {'-'*9}  {'-'*9}")
for m in models:
    path = f"outputs_{m}/results/label_probe.json"
    if not os.path.exists(path):
        print(f"  {labels[m]:<20}  (no results yet)")
        continue
    d = json.load(open(path))
    zc_i = d["in_domain"]["zc"]["auc_mean"]
    zs_i = d["in_domain"]["zs"]["auc_mean"]
    zc_o = d["ood_transfer"]["zc"]["auc"]
    zs_o = d["ood_transfer"]["zs"]["auc"]
    print(f"  {labels[m]:<20}  {zc_i:>11.4f}  {zs_i:>11.4f}  {zc_o:>9.4f}  {zs_o:>9.4f}")
print()
print("  Goal: Z_s in-dom AUC → 0.5 (chance) while Z_c stays high")
PYEOF
