#!/bin/bash
#SBATCH --job-name=A6_pdv2_rand_spk
#SBATCH --partition=gpu-a100,gpu-l40s,gpu-a100-short,gpu-a100-preempt,gpu-l40s-preempt
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/A6_pdv2_rand_spk_%j.out
#SBATCH --error=logs/A6_pdv2_rand_spk_%j.err

set -euo pipefail

SEED=${SEED:-44}

echo "============================================================"
echo "  \method random-speaker ablation | seed=${SEED}"
echo "  domain_label_mode=random_speaker (speaker-grouped, random label)"
echo "  Job  : ${SLURM_JOB_ID}"
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

OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_random_speaker_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results} logs

cd avh_sup

"$PYTHON" train_test_fcd_a6_av1m_pseudodomain_v2.py \
  --config_path "configs/A6_av1m_pseudodomain_v2_seed${SEED}_random_speaker.yaml" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
