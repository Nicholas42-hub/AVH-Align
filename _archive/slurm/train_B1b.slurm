#!/bin/bash
#SBATCH --job-name=train_B1b
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_B1b_%j.out
#SBATCH --error=logs/train_B1b_%j.err

# ============================================================
#  Modal Balance B1b — Per-Modality Cls + Balance Penalty (Branch 1b)
#
#  Base: A2 (adversarial + stop-gradient on Z_s)
#  B1:   modal_head_visual(v_c) → CE(labels)
#        modal_head_audio(a_c)  → CE(labels)
#  B1b:  L_balance = (CE_audio - CE_visual)²
#        Forces audio and visual CE losses to stay equal,
#        preventing audio from dominating the causal branch.
# ============================================================

echo "============================================================"
echo "  AVH-Align Modal Balance B1b — Balance Penalty"
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

mkdir -p avh_sup/outputs_B1b/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_modal.py \
    --config_path configs/B1b.yaml

echo ""
echo "============================================================"
echo "  B1b training done: $(date)"
echo "============================================================"
