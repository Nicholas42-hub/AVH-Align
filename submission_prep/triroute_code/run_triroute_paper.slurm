#!/bin/bash
#SBATCH --job-name=triroute_paper
#SBATCH --partition=gpu-a100-short         # adjust to your cluster partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-2                        # seeds: 43, 44, 45
#SBATCH --output=logs/triroute_%A_%a.out
#SBATCH --error=logs/triroute_%A_%a.err

set -euo pipefail

# ── Paths (edit these) ────────────────────────────────────────────────────────
BASE=/path/to/triroute_code               # directory containing this script
PYTHON=/path/to/python                    # Python interpreter (with dependencies)

AV1M_FEATURES=/path/to/av1m_features
AV1M_CSV=/path/to/av1m_csv
FAVC_FEATURES=/path/to/favc_features
FAVC_CSV=/path/to/favc_csv
# ─────────────────────────────────────────────────────────────────────────────

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="$BASE/outputs/triroute_seed${SEED}"

echo "============================================================"
echo "  TRIROUTE unsup-sync | seed=$SEED"
echo "  Job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node  : $SLURM_NODELIST"
echo "  Out   : $OUT"
echo "  Start : $(date)"
echo "============================================================"

mkdir -p logs "$OUT"

$PYTHON "$BASE/train_eval_triroute_unsup_sync.py" \
  --output_dir          "$OUT"           \
  --av1m_features       "$AV1M_FEATURES" \
  --av1m_csv_root       "$AV1M_CSV"      \
  --favc_features       "$FAVC_FEATURES" \
  --favc_csv_root       "$FAVC_CSV"      \
  --epochs              100              \
  --frames_per_clip     4                \
  --batch_size          64               \
  --num_workers         4                \
  --tau                 15               \
  --lr                  1.0e-4           \
  --cross_domain_eval_every 10           \
  --seed                "$SEED"

echo ""
echo "============================================================"
echo "  Done seed=$SEED at $(date)"
echo "============================================================"
