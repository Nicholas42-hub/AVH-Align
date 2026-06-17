#!/bin/bash
#SBATCH --job-name=anom3way_s43
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/anom3way_s43_%j.out
#SBATCH --error=logs/anom3way_s43_%j.err

# ============================================================
#  Caren's three unsup anomaly methods on FAVC
#  ckpt: seed43 unsup TriRoute ep20 (overall_auc=0.918)
#  ref : AV1M val real-only (label==0, 519 clips)
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
CKPT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/triroute_unsup_sync_best_auc.pt"
OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed43/results_anomaly_3way"

echo "============================================================"
echo "  3-way anomaly | seed=43 | $(date)"
echo "  Ckpt: $CKPT"
echo "  Out : $OUT"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$OUT"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

cd avh_sup

$PYTHON eval_anomaly_three_ways.py \
  --ckpt "$CKPT" \
  --av1m_root "$BASE/data/avh_features/val" \
  --av1m_csv  "$BASE/avh_sup/csv_metadata/av1m/val_labels.csv" \
  --favc_root "$BASE/data/favc_features" \
  --favc_split_csv "$BASE/avh_sup/csv_metadata/favc_7030_canonical/test_split.csv" \
  --output_dir "$OUT" \
  --knn_k 5 \
  --shrinkage 0.05 \
  --alphas "0.5,1.0,2.0"

echo ""
echo "============================================================"
echo "  Done at $(date)"
echo "  Results: $OUT/anomaly_three_ways.json"
echo "============================================================"
