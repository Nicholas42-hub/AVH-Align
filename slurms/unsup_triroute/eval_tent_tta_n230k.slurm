#!/bin/bash
#SBATCH --job-name=tent_tta_n230k
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short,gpu-a100-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tent_tta_n230k_%j.out
#SBATCH --error=logs/tent_tta_n230k_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
CKPT=${CKPT:-$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_n230k_seed44/triroute_unsup_sync_best_auc.pt}
LR=${LR:-1e-4}
STEPS=${STEPS:-1}
OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_n230k_tent_lr${LR}_steps${STEPS}"

echo "============================================================"
echo "  TENT TTA on FAVC (LayerNorm-only adaptation)"
echo "  Job   : ${SLURM_JOB_ID}"
echo "  CKPT  : $CKPT"
echo "  LR    : $LR  STEPS: $STEPS"
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

$PYTHON avh_sup/eval_tent_tta.py \
  --ckpt "$CKPT" \
  --output_dir "$OUT" \
  --lr "$LR" \
  --steps "$STEPS"

echo ""
echo "Done: $(date)"
