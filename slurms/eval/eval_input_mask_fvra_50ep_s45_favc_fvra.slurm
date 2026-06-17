#!/bin/bash
#SBATCH --job-name=mask_50ep_s45
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/mask_50ep_s45_%j.out
#SBATCH --error=logs/mask_50ep_s45_%j.err

# ============================================================
#  Input masking sweep (Table 5) on FAVC FV-RA, single ckpt:
#    outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/
#      best-favc-fvra-epoch=76-auc=0.8510.ckpt
#  Same script as the multi-seed sweep, single seed run.
#  NOTE: this ckpt is favc_fvra-target-selected (sup A6 alone overall=0.929).
# ============================================================

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=76-auc=0.8510.ckpt"
CFG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed45.yaml"
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
RESULTS_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/results_input_mask_fvra"

cd "$BASE"
mkdir -p logs "$RESULTS_DIR"

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "  Input masking sweep | seed=45 (favc_fvra ckpt) | $(date)"
echo "  Ckpt    : $CKPT"
echo "  Config  : $CFG"
echo "  Features: $FEATURES"
echo "  Results : $RESULTS_DIR"
echo "============================================================"

cd avh_sup
$PYTHON eval_perturbation_robustness.py \
  --ckpt_path "$CKPT" \
  --model fcd_a6 \
  --config "$CFG" \
  --favc_root "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --batch_size 1 \
  --seed 45 \
  --output_dir "$RESULTS_DIR" \
  --fake_category FakeVideo-RealAudio

echo "============================================================"
echo "  Done at $(date)"
echo "============================================================"
