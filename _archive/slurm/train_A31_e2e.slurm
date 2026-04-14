#!/bin/bash
#SBATCH --job-name=train_A31_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_A31_e2e_%j.out
#SBATCH --error=logs/train_A31_e2e_%j.err

# ============================================================
#  A31_e2e — L1-d mean alignment (constant gradient, no stalling)
#
#  A30 failure analysis:
#    d² per-step gradient 25× A29 (safe) → task collapsed ep0.
#    lambda_ladv=1.0 contributed 0.559 (45% of non-task) → 3.8× imbalance.
#
#  A31 fixes:
#    (1) L1-d loss: mean(|mu0-mu1|/std_combined) — constant gradient,
#        self-accelerating near convergence (no stalling like A29/A30).
#        Per-dim gradient at λ=0.05: 0.0027/step (31% of A29's safe 0.0088).
#    (2) lambda_ladv=0.0: removes 0.559 competing gradient.
#        Task fraction: 22% (A30) → 72% (A31).
#    (3) gradient_clip_val=2.0: more budget for task gradient.
# ============================================================

echo "============================================================"
echo "  A31_e2e — L1-d mean alignment (constant gradient)"
echo "  lambda_mean_align=0.05 (L1-d loss ~0.53, contribution ~11% task)"
echo "  lambda_ladv=0.0 (removed competing gradient)"
echo "  gradient_clip_val=2.0"
echo "  Finetune from A9_e2e"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date +"%a %d %b %Y %T %Z")"
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

mkdir -p avh_sup/outputs_A31_e2e/ckpts avh_sup/outputs_A31_e2e/results avh_sup/outputs_A31_e2e/logs logs

cd avh_sup

echo ">>> Starting A31_e2e training (L1-d mean alignment)..."
$PYTHON train_e2e_a31.py --config_path configs/A31_e2e.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
