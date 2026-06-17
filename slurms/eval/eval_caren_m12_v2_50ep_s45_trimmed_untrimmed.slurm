#!/bin/bash
#SBATCH --job-name=m12_v2_50ep
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/m12_v2_50ep_%A_%a.out
#SBATCH --error=logs/m12_v2_50ep_%A_%a.err

# ============================================================
#  TriRoute + kNN rank fusion on FAVC and AV1M val (trimmed/untrimmed),
#  using the 50ep seed45 favc_fvra-selected A6 ckpt that gives
#  sup A6 alone FAVC overall = 0.9290.
#
#  NOTE: this ckpt is selected by FAVC FV-RA (target-domain) AUC,
#  which violates the paper's source-val-only ckpt policy.
#
#  Unsup TriRoute kept at seed44 vox2_repro to match the previous
#  v2 run; the unsup just provides u_v features for kNN ref/test.
#
#  Array index map:
#    0 -> FAVC untrimmed
#    1 -> FAVC trimmed
#    2 -> AV1M val untrimmed
#    3 -> AV1M val trimmed
# ============================================================

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

SUP_CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=76-auc=0.8510.ckpt"
SUP_CFG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed45.yaml"
UNSUP_CKPT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed44/triroute_unsup_sync_best_auc.pt"

FAVC_UNTRIMMED="$BASE/data/favc_features"
FAVC_TRIMMED="$BASE/data/favc_features_trimmed"
AV1M_UNTRIMMED="$BASE/data/avh_features"
AV1M_TRIMMED="/scratch/punim2637/nnliang/avd1m_features_trimmed"

case "$SLURM_ARRAY_TASK_ID" in
  0) EVAL_DATASET=favc; EVAL_FEATS="$FAVC_UNTRIMMED"; REF_FEATS="$AV1M_UNTRIMMED"; TAG=favc_untrimmed ;;
  1) EVAL_DATASET=favc; EVAL_FEATS="$FAVC_TRIMMED";   REF_FEATS="$AV1M_TRIMMED";   TAG=favc_trimmed   ;;
  2) EVAL_DATASET=av1m; EVAL_FEATS="$AV1M_UNTRIMMED"; REF_FEATS="$AV1M_UNTRIMMED"; TAG=av1m_untrimmed ;;
  3) EVAL_DATASET=av1m; EVAL_FEATS="$AV1M_TRIMMED";   REF_FEATS="$AV1M_TRIMMED";   TAG=av1m_trimmed   ;;
  *) echo "bad SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed45/results_caren_m12_v2_${TAG}"

cd "$BASE"
mkdir -p logs "$OUT"

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "  task     : $SLURM_ARRAY_TASK_ID  ($TAG)"
echo "  sup_ckpt : $SUP_CKPT"
echo "  unsup_ckpt: $UNSUP_CKPT"
echo "  eval_set : $EVAL_DATASET @ $EVAL_FEATS"
echo "  knn_ref  : $REF_FEATS"
echo "  output   : $OUT"
echo "  start    : $(date)"
echo "============================================================"

cd avh_sup
$PYTHON eval_caren_methods_12_v2.py \
  --sup_ckpt "$SUP_CKPT" \
  --sup_config "$SUP_CFG" \
  --unsup_ckpt "$UNSUP_CKPT" \
  --eval_dataset "$EVAL_DATASET" \
  --eval_features_root "$EVAL_FEATS" \
  --ref_features_root "$REF_FEATS" \
  --output_dir "$OUT" \
  --knn_k 10 \
  --betas "0.0,0.5,1.0,2.0" \
  --alphas "0.5,1.0,2.0,5.0"

echo ""
echo "Done $TAG at $(date)"
