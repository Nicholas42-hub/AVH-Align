#!/bin/bash
#SBATCH --job-name=caren_m3
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/caren_m3_%j.out
#SBATCH --error=logs/caren_m3_%j.err

# ============================================================
#  Caren's Method 3: TENT-style TTA on FAVC test
#  (freeze causal_head, adapt CCD/URD/incon)
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SUP_CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=76-auc=0.8510.ckpt"
SUP_CFG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed45.yaml"
OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/results_caren_m3_best"

cd "$BASE"
mkdir -p logs "$OUT"

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1

cd avh_sup
$PYTHON eval_caren_method3_tta.py \
  --sup_ckpt "$SUP_CKPT" \
  --sup_config "$SUP_CFG" \
  --output_dir "$OUT"
echo ""
echo "Done at $(date)"
