#!/bin/bash
#SBATCH --job-name=train_A3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A3_%j.out
#SBATCH --error=logs/train_A3_%j.err

# ============================================================
#  Ablation A3 — Adversarial + Stop Gradient + Domain-Aware
#
#  Component 1: GRL adversarial loss (Z_s non-informative for label)  ✓
#  Component 2: Stop gradient / Z_c-only classification               ✓
#  Component 3: Domain adversarial (AV1M vs FAVC) on Z_s             ✓
#
#  Key idea: Mix AV1M + FAVC train during training.
#  GRL_domain forces Z_s to be domain-invariant → proper causal/spurious
#  separation. FAVC samples used for domain signal only (cls_label=-1).
# ============================================================

echo "============================================================"
echo "  AVH-Align Ablation A3 — Adversarial + Stop Gradient + Domain"
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

mkdir -p avh_sup/outputs_A3/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_ablation.py \
    --config_path configs/A3.yaml \
    --mode train

echo ""
echo "============================================================"
echo "  A3 training done: $(date)"
echo "============================================================"
