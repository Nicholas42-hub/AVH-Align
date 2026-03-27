#!/bin/bash
#SBATCH --job-name=train_B2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_B2_%j.out
#SBATCH --error=logs/train_B2_%j.err

# ============================================================
#  Modal Balance B2 — Cross-Modal Alignment (方向2)
#
#  Base: A2 (adversarial + stop-gradient on Z_s)
#  New:  L_cross = mean(1 - cosine_similarity(v_c_t, a_c_t))
#
#  Pulls v_c and a_c toward the same semantic space per frame,
#  encoding genuine audio-visual causal dependency.
# ============================================================

echo "============================================================"
echo "  AVH-Align Modal Balance B2 — Cross-Modal Alignment"
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

mkdir -p avh_sup/outputs_B2/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_modal.py \
    --config_path configs/B2.yaml

echo ""
echo "============================================================"
echo "  B2 training done: $(date)"
echo "============================================================"
