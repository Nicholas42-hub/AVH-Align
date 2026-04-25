#!/bin/bash
#SBATCH --job-name=A6_pd_seeds
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/A6_av1m_pseudodomain_seeds_%j.out
#SBATCH --error=logs/A6_av1m_pseudodomain_seeds_%j.err

echo "============================================================"
echo "  A6-AV1M-PseudoDomain (speaker-group) — seeds 43/44/45"
echo "  Ablation: domain push-pull on AV1M real-only speaker groups"
echo "  No FAVC clips; domain = hash(speaker_id) % 2"
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

mkdir -p avh_sup/outputs_A6_av1m_pseudodomain_seed43/{logs,ckpts,results} \
         avh_sup/outputs_A6_av1m_pseudodomain_seed44/{logs,ckpts,results} \
         avh_sup/outputs_A6_av1m_pseudodomain_seed45/{logs,ckpts,results} \
         logs

cd avh_sup

echo ""
echo "--- Seed 43 ---"
$PYTHON train_test_fcd_a6_av1m_pseudodomain.py \
  --config_path configs/A6_av1m_pseudodomain_seed43.yaml

echo ""
echo "--- Seed 44 ---"
$PYTHON train_test_fcd_a6_av1m_pseudodomain.py \
  --config_path configs/A6_av1m_pseudodomain_seed44.yaml

echo ""
echo "--- Seed 45 ---"
$PYTHON train_test_fcd_a6_av1m_pseudodomain.py \
  --config_path configs/A6_av1m_pseudodomain_seed45.yaml

echo ""
echo "============================================================"
echo "  All seeds done: $(date)"
echo "============================================================"
