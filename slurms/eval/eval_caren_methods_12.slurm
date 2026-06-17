#!/bin/bash
#SBATCH --job-name=caren_m12
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/caren_m12_%j.out
#SBATCH --error=logs/caren_m12_%j.err

# ============================================================
#  Caren's Methods 1 + 2: late fusion of sup TriRoute (A6) + unsup signals
#  - sup ckpt: paper-clean A6, seed43, model-epoch=09 (policy-compliant)
#  - unsup ckpt: vox2-trained TriRoute, seed43 ep20 (best_auc 0.918)
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SUP_CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=76-auc=0.8510.ckpt"
SUP_CFG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed45.yaml"
UNSUP_CKPT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/triroute_unsup_sync_best_auc.pt"
OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/results_caren_m12_best"

cd "$BASE"
mkdir -p logs "$OUT"

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

cd avh_sup
$PYTHON eval_caren_methods_12.py \
  --sup_ckpt "$SUP_CKPT" \
  --sup_config "$SUP_CFG" \
  --unsup_ckpt "$UNSUP_CKPT" \
  --output_dir "$OUT" \
  --knn_k 10 \
  --betas "0.0,0.5,1.0,2.0" \
  --alphas "0.5,1.0,2.0,5.0"
echo ""
echo "Done at $(date)"
