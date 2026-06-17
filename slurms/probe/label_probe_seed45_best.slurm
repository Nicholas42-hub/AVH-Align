#!/bin/bash
#SBATCH --job-name=lprobe_s45_best
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/label_probe_seed45_best_%j.out
#SBATCH --error=logs/label_probe_seed45_best_%j.err

# ============================================================
#  Label + Domain probe by FAVC subset
#  Ckpt   : seed45 best FV-RA (epoch=76, FV-RA AUC=0.8510)
#  Output : outputs_A6_av1m_pseudodomain_v2_50ep_seed45/results_label_probe_subset/
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
SEED=45
OUT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
CKPT="$OUT_DIR/ckpts/favc_fvra/best-favc-fvra-epoch=76-auc=0.8510.ckpt"
CFG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml"
RESULTS="$OUT_DIR/results_label_probe_subset"

echo "============================================================"
echo "  Label probe by subset | seed=${SEED} | $(date)"
echo "  Ckpt   : $CKPT"
echo "  Config : $CFG"
echo "  Output : $RESULTS"
echo "============================================================"

mkdir -p "$RESULTS"

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

cd avh_sup

$PYTHON eval_label_probe_by_subset.py \
    --ckpt_path "$CKPT" \
    --config    "$CFG" \
    --av1m_root "$BASE/data/avh_features/val" \
    --av1m_csv  "$BASE/avh_sup/csv_metadata/av1m/val_labels.csv" \
    --favc_root "$BASE/data/favc_features" \
    --output_dir "$RESULTS" \
    --seed 42 \
    --n_folds 5

echo "============================================================"
echo "  Done at $(date)"
echo "  Results: $RESULTS/label_probe_by_subset.json"
echo "============================================================"
