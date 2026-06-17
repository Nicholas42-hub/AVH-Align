#!/bin/bash
#SBATCH --job-name=tri_unsup_120k
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=logs/tri_unsup_120k_%A_%a.out
#SBATCH --error=logs/tri_unsup_120k_%A_%a.err

# ============================================================
#  TriRoute unsup sync training on Vox2 120k (3x current)
#  Uses same hyperparameters as the 0.918 ckpt run:
#    epochs=40, lr=1e-4, no LR decay, no grad clip
#  best_auc.pt save (cross-domain FAVC AUC) is built into the script.
#
#  Multi-root features: scratch (existing 40k + part2) + gpfs (part3)
#  Combined train_labels.csv lives at $WORK_BASE/csv/train_labels.csv
# ============================================================

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

WORK_BASE=/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n120k

# Three feature roots, colon-separated. Dataset will try in order.
FEAT_ROOTS="/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40000_seed44_e20/features:/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40k_part2/features:/data/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40k_part3/features"

OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_n120k_seed${SEED}"

echo "============================================================"
echo "  TriRoute unsup sync 120k | seed=$SEED"
echo "  Job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Train : $WORK_BASE/csv/train_labels.csv (120k clips)"
echo "  Roots : (3 sources, colon-separated multi-fs)"
echo "  Out   : $OUT"
echo "  Start : $(date)"
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
export CUDA_VISIBLE_DEVICES=0

$PYTHON avh_sup/train_eval_triroute_unsup_sync.py \
  --output_dir "$OUT" \
  --train_features "$FEAT_ROOTS" \
  --train_csv_root "$WORK_BASE/csv" \
  --epochs 40 \
  --batch_size 64 \
  --num_workers 4 \
  --tau 15 \
  --lr 1.0e-4 \
  --cross_domain_eval_every 1 \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "  Done seed=$SEED at $(date)"
echo "============================================================"
