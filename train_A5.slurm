#!/bin/bash
#SBATCH --job-name=train_A5
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A5_%j.out
#SBATCH --error=logs/train_A5_%j.err

# ============================================================
#  Ablation A5 — FCD-Inspired Feature Causality Decomposition
#
#  Each modality feature is split into THREE components:
#    synergistic  s^m  — cross-modal shared, task-relevant
#                         aligned via Sinkhorn divergence (L_SDA)
#    unique       u^m  — modality-specific, task-relevant
#                         constrained via CE proxy (L_MI)
#    redundant    r^m  — spurious shortcuts / noise  (r = h − u)
#                         kept modality-discriminative (L_Dis)
#
#  Classification: causal_head([s_v, u_v, s_a, u_a])   [1024-dim, same as A2]
#  Redundant probe: redundant_head([r_v, r_a])           [target AUC ≈ 0.5]
#
#  Reference: "Plug-and-play Feature Causality Decomposition for
#              Multimodal Representation Learning", NeurIPS 2025.
# ============================================================

echo "============================================================"
echo "  AVH-Align A5 — FCD Feature Causality Decomposition"
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

mkdir -p avh_sup/outputs_A5/{logs,ckpts,results} logs

cd avh_sup
python train_test_fcd.py \
    --config_path configs/A5.yaml \
    --mode train

echo ""
echo "============================================================"
echo "  A5 training done: $(date)"
echo "============================================================"
