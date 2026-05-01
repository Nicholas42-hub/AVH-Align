#!/bin/bash
#SBATCH --job-name=pdv2_lastck
#SBATCH --partition=gpu-a100
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/retrain_A6_pdv2_50ep_lastckpt_%A_%a.out
#SBATCH --error=logs/retrain_A6_pdv2_50ep_lastckpt_%A_%a.err

# ============================================================
#  Retrain A6-AV1M-PseudoDomain-v2 (50 ep) — seeds 43/44/45
#  Array job: 3 seeds run in parallel on separate nodes.
#
#  Ckpt protocol: last.ckpt (epoch 49), save_last=True in script.
#  No FAVC data involved — protocol-compliant.
# ============================================================

SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  Retrain A6-pdv2-50ep (last ckpt) | seed=${SEED}"
echo "  Array task : $SLURM_ARRAY_TASK_ID"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  Start      : $(date)"
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

OUT="avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
mkdir -p "${OUT}"/{logs,ckpts,results} logs
rm -f "${OUT}"/ckpts/*.ckpt
rm -rf "${OUT}"/logs/lightning_logs

cd avh_sup

$PYTHON train_test_fcd_a6_av1m_pseudodomain.py \
  --config_path configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml

echo ""
echo "============================================================"
echo "  Done: seed=${SEED} at $(date)"
echo "============================================================"
