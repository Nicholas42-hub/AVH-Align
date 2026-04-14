#!/bin/bash
#SBATCH --job-name=train_A33_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A33_e2e_%j.out
#SBATCH --error=logs/train_A33_e2e_%j.err

echo "============================================================"
echo "  A33_e2e — A2_e2e + Explicit IPW Backdoor Adjustment"
echo "  use_sg=True (causal_head only)"
echo "  domain_predictor on Z_c.detach() — non-adversarial"
echo "  IPW reweights task loss: w_i = 0.5 / q(S|Z_c), clip [0.1, 10]"
echo "  warmup: 2 epochs with uniform weights"
echo "  AV-HuBERT fully unfrozen, ~65k AV1M + 5k FAVC clips"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

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

BASE=/data/projects/punim2637/nnliang/AVH-Align

mkdir -p "$BASE/logs"
mkdir -p "$BASE/avh_sup/outputs_A33_e2e/ckpts"
mkdir -p "$BASE/avh_sup/outputs_A33_e2e/logs"
mkdir -p "$BASE/avh_sup/outputs_A33_e2e/results"

echo ">>> Starting A33_e2e training (A2_e2e + IPW backdoor adjustment)..."
cd "$BASE/avh_sup"

$PYTHON train_e2e_a33.py --config_path configs/A33_e2e.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "============================================================"
