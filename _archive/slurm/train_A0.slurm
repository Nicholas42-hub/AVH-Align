#!/bin/bash
#SBATCH --job-name=train_A0
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A0_%j.out
#SBATCH --error=logs/train_A0_%j.err

# ============================================================
#  Ablation A0 — Orthogonal Loss Only
#
#  Causal-spurious split with orthogonality constraint only.
#  No adversarial pressure on Z_s — control condition.
#
#  Classification: full_head(Z_c + Z_s) — gradient flows to both
#  Orth loss    : forces Z_c ⊥ Z_s  ✓
#  Adversarial  : none               ✗
#  Data         : AV1M only
# ============================================================

echo "============================================================"
echo "  AVH-Align Ablation A0 — Orthogonal Loss Only"
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

mkdir -p avh_sup/outputs_A0/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_ablation.py \
    --config_path configs/A0.yaml \
    --mode train

echo ""
echo "============================================================"
echo "  A0 training done: $(date)"
echo "============================================================"
