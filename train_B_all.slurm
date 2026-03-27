#!/bin/bash
#SBATCH --job-name=train_B_all
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_B_all_%j.out
#SBATCH --error=logs/train_B_all_%j.err

# ============================================================
#  Modal Balance B_all — All Three Constraints Combined
#
#  Base: A2 (adversarial + stop-gradient on Z_s)
#  方向1: per-modality classification loss (use_modal_cls)
#  方向2: cross-modal alignment loss (use_cross_modal)
#  方向3: modal discriminator adversarial (use_modal_disc)
#
#  Tests whether all three constraints together maximally address
#  the causal branch audio-dominance problem.
# ============================================================

echo "============================================================"
echo "  AVH-Align Modal Balance B_all — All Three Constraints"
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

mkdir -p avh_sup/outputs_B_all/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_modal.py \
    --config_path configs/B_all.yaml

echo ""
echo "============================================================"
echo "  B_all training done: $(date)"
echo "============================================================"
