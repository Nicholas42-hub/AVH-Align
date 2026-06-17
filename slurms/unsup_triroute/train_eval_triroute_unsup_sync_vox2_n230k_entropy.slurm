#!/bin/bash
#SBATCH --job-name=tri_unsup_n230k_ent
#SBATCH --partition=gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tri_unsup_n230k_ent_%j.out
#SBATCH --error=logs/tri_unsup_n230k_ent_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SCRATCH_ROOT=/data/scratch/projects/punim2637/VoxCeleb2/a6_unsup_chunks
UNIFIED=$SCRATCH_ROOT/vox2_n230k_unified
SEED=${SEED:-44}
EPOCHS=${EPOCHS:-10}
LAMBDA_ENT=${LAMBDA_ENT:-0.1}
OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_n230k_entropy_seed${SEED}"

TRAIN_FEATURES="$SCRATCH_ROOT/vox2_n40000_seed44_e20/features:$SCRATCH_ROOT/vox2_n40k_part2/features:$SCRATCH_ROOT/vox2_p01_80k/features:$SCRATCH_ROOT/vox2_p02_94k/features:$SCRATCH_ROOT/vox2_p03_108k/features:$SCRATCH_ROOT/vox2_p04_122k/features:$SCRATCH_ROOT/vox2_p05_136k/features:$SCRATCH_ROOT/vox2_p06_150k/features:$SCRATCH_ROOT/vox2_p07_164k/features:$SCRATCH_ROOT/vox2_p08_178k/features:$SCRATCH_ROOT/vox2_p09_192k/features:$SCRATCH_ROOT/vox2_p10_206k/features:$SCRATCH_ROOT/vox2_p11_220k/features:$SCRATCH_ROOT/vox2_p12_235k/features"

echo "============================================================"
echo "  TriRoute-unsup-sync n230k + ENTROPY-MIN | seed=$SEED epochs=$EPOCHS"
echo "  Job   : ${SLURM_JOB_ID}"
echo "  Node  : $SLURM_NODELIST"
echo "  lambda_entropy=$LAMBDA_ENT"
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
  --train_features "$TRAIN_FEATURES" \
  --train_csv_root "$UNIFIED/csv" \
  --epochs "$EPOCHS" \
  --batch_size 64 \
  --num_workers 4 \
  --tau 15 \
  --lr 1.0e-4 \
  --cross_domain_eval_every 1 \
  --lambda_entropy "$LAMBDA_ENT" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Output: $OUT"
echo "============================================================"
