#!/bin/bash
#SBATCH --job-name=A6_favc_realdom
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A6_favc_realdom_%j.out
#SBATCH --error=logs/train_A6_favc_realdom_%j.err

# ============================================================
#  A6 FAVC — real-only domain supervision (fixed version)
#  Fake clips get domain_label=-1 → excluded from domain loss
#  Only real clips supply pseudo speaker-group domain labels
#  Seeds 42 43 44 45 sequential on one MIG slice
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  A6 FAVC real-domain — seeds 42 43 44 45 (sequential)"
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
for SEED in 42 43 44 45; do
    echo ""
    echo "--- A6 FAVC realdom seed ${SEED} ---"
    echo "$(date)"
    mkdir -p outputs_A6_favc_realdom_seed${SEED}/{logs,ckpts,results}
    $PYTHON train_test_fcd_a6_favc_pseudodomain.py \
        --config_path configs/A6_favc_realdom_seed${SEED}.yaml
    echo "Seed ${SEED} done: $(date)"
done

echo ""
echo "All seeds done: $(date)"
