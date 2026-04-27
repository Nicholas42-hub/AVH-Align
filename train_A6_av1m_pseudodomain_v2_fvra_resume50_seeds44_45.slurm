#!/bin/bash
#SBATCH --job-name=A6_pdv2_r50
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/A6_pdv2_resume50_%A_%a.out
#SBATCH --error=logs/A6_pdv2_resume50_%A_%a.err

SEEDS=(44 45)
CKPTS=(
  "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed44/ckpts/favc_fvra/best-favc-fvra-epoch=05-auc=0.7420.ckpt"
  "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/ckpts/favc_fvra/best-favc-fvra-epoch=09-auc=0.6516.ckpt"
)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  A6-AV1M-PseudoDomain-v2 FV-RA resume to 50ep | seed=${SEED}"
echo "  checkpoint: ${CKPT}"
echo "  Start: $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
cd "$BASE"

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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

OUTPUT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup

"$PYTHON" train_test_fcd_a6_av1m_pseudodomain_v2.py \
  --config_path "configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml" \
  --seed "$SEED" \
  --ckpt_path "$CKPT"

echo ""
echo "============================================================"
echo "  Done: A6-AV1M-PseudoDomain-v2 resume50 seed=${SEED} at $(date)"
echo "============================================================"
