#!/bin/bash
#SBATCH --job-name=label_probe
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/label_probe_%j.out
#SBATCH --error=logs/label_probe_%j.err

# ============================================================
#  Label Predictability Probe — A0 / A1 / A2
#
#  For each model, trains a logistic regression probe on frozen
#  Z_c and Z_s representations (no model head required — correct
#  for A0 which only trains full_head).
#
#  Outputs three sections per model:
#    1. In-domain (AV1M val, 5-fold CV): Z_c AUC vs Z_s AUC
#    2. OOD transfer (train AV1M → eval FAVC): Z_c vs Z_s
#    3. Domain probe (AV1M=0 vs FAVC=1): checks domain separability
#
#  A0 baseline:  classification + orthogonality loss ONLY
#  A1:           + adversarial GRL on Z_s
#  A2:           + adversarial GRL + stop-gradient (Z_c-only cls)
# ============================================================

echo "============================================================"
echo "  Label Predictability Probe — A0 / A1 / A2"
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
    avh_sup/outputs_A0/results \
    avh_sup/outputs_A1/results \
    avh_sup/outputs_A2/results

cd avh_sup

for MODEL in A0 A1 A2; do
    CKPT=$(ls outputs_${MODEL}/ckpts/*.ckpt 2>/dev/null | grep -v ".bak" | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "WARNING: no checkpoint for ${MODEL}, skipping."
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  Label probe: ${MODEL} | $(basename $CKPT)"
    echo "================================================================"

    python eval_label_probe.py \
        --ckpt_path   "$CKPT" \
        --config      configs/${MODEL}.yaml \
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
echo "  All label probes done: $(date)"
echo "============================================================"
