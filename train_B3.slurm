#!/bin/bash
#SBATCH --job-name=train_B3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_B3_%j.out
#SBATCH --error=logs/train_B3_%j.err

# ============================================================
#  Modal Balance B3 — Modal Discriminator Adversarial (方向3)
#
#  Base: A2 (adversarial + stop-gradient on Z_s)
#  New:  modal_disc(GRL_modal(v_c || a_c)) → BCE(modality label)
#
#  Discriminator learns to tell v_c from a_c tokens.
#  GRL reverses the gradient into both causal projectors:
#  they are forced to produce modality-agnostic representations.
# ============================================================

echo "============================================================"
echo "  AVH-Align Modal Balance B3 — Modal Discriminator Adversarial"
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

mkdir -p avh_sup/outputs_B3/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_modal.py \
    --config_path configs/B3.yaml

echo ""
echo "============================================================"
echo "  B3 training done: $(date)"
echo "============================================================"
