#!/bin/bash
#SBATCH --job-name=tri_unsup_vox2
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/tri_unsup_vox2_%A_%a.out
#SBATCH --error=logs/tri_unsup_vox2_%A_%a.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
WORK=/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks/vox2_n40000_seed44_e20
SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed${SEED}"

echo "============================================================"
echo "  TriRoute-unsup-sync Vox2 | seed=$SEED"
echo "  Job   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Node  : $SLURM_NODELIST"
echo "  Train : $WORK/features"
echo "  CSV   : $WORK/csv"
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

$PYTHON avh_sup/train_eval_triroute_unsup_sync.py \
  --output_dir "$OUT" \
  --train_features "$WORK/features" \
  --train_csv_root "$WORK/csv" \
  --epochs 20 \
  --batch_size 64 \
  --num_workers 4 \
  --tau 15 \
  --lr 1.0e-4 \
  --cross_domain_eval_every 5 \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "  Done seed=$SEED at $(date)"
echo "============================================================"
