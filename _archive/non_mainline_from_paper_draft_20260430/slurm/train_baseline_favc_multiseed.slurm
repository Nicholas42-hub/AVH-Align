#!/bin/bash
#SBATCH --job-name=train_baseline_favc
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_baseline_favc_%j.out
#SBATCH --error=logs/train_baseline_favc_%j.err

# ============================================================
#  Train AVH-Align/sup baseline on FakeAVCeleb (seeds 43-45)
#  Architecture: AVH_Sup (MLP, 2048->512->256->128->1)
#  Loss: cross-entropy on log-sum-exp pooled frame scores
#  Optimizer: Adam lr=1e-3  Epochs: 50  EarlyStop: val_loss patience=20
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  Baseline (AVH-Align/sup) FAVC training — seeds 43 44 45"
echo "  Job ID : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

cd avh_sup
for SEED in 43 44 45; do
    echo ""
    echo "--- Baseline FAVC seed ${SEED} ---"
    echo "$(date)"
    mkdir -p outputs_baseline_favc_seed${SEED}/{logs,ckpts,results}
    $PYTHON train_test_baseline_favc.py \
        --config_path configs/baseline_favc_seed${SEED}.yaml
    echo "Seed ${SEED} done: $(date)"
done

echo ""
echo "All seeds done: $(date)"
