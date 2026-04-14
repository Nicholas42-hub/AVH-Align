#!/bin/bash
#SBATCH --job-name=train_A10_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_A10_e2e_%j.out
#SBATCH --error=logs/train_A10_e2e_%j.err

echo "============================================================"
echo "  A10_e2e — SAD + CORAL + Balanced Sampler, Fully E2E"
echo "  Fixes: CORAL loss, balanced domain batching, all FAVC, augmentation"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p avh_sup/outputs_A10_e2e/ckpts avh_sup/outputs_A10_e2e/logs logs

cd avh_sup

# Generate E2E CSVs if not already done
if [ ! -f csv_metadata/av1m_e2e/train_e2e_full.csv ]; then
    echo ">>> Generating comprehensive AV1M E2E CSV..."
    $PYTHON generate_e2e_csv.py
    echo ">>> CSV generation done."
fi

echo ">>> Starting A10_e2e training (CORAL + BalancedDomainSampler)..."
$PYTHON train_e2e_a10.py --config_path configs/A10_e2e.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A10_e2e/ckpts/
echo "============================================================"
