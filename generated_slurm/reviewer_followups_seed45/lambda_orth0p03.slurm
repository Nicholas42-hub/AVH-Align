#!/bin/bash
#SBATCH --job-name=rv_lambda_orth0p03
#SBATCH --partition=gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.err

set -euo pipefail

echo "============================================================"
echo "  Reviewer follow-up: lambda_orth = 0.03"
echo "  Seed: 45"
echo "  Config: /data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/reviewer_followups_seed45/A6_pdv2_s45_lambda_orth0p03.yaml"
echo "  Output: /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03"
echo "  Start: $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/logs" "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/ckpts" "/data/projects/punim2637/nnliang/AVH-Align/logs"

cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
python train_test_fcd_a6_av1m_pseudodomain_v2.py \
  --config_path "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/reviewer_followups_seed45/A6_pdv2_s45_lambda_orth0p03.yaml" \
  --seed 45

BEST_TXT="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/ckpts/favc_fvra/best_favc_fvra.txt"
if [[ -f "$BEST_TXT" ]]; then
  CKPT=$(grep '^checkpoint=' "$BEST_TXT" | cut -d= -f2-)
elif [[ -f "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/ckpts/last.ckpt" ]]; then
  CKPT="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/ckpts/last.ckpt"
else
  CKPT=$(ls "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/ckpts"/*.ckpt 2>/dev/null | sort | tail -1)
fi

if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "ERROR: No checkpoint found for lambda_orth0p03" >&2
  exit 2
fi

echo "  Eval checkpoint: $CKPT"
python eval_favc.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "/data/projects/punim2637/nnliang/AVH-Align/data/favc_features" \
  --csv_root_path "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical" \
  --split test \
  --output_dir "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_pdv2_s45_lambda_orth0p03/results_favc_7030_full_best"

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
