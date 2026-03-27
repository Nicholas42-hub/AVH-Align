#!/bin/bash
#SBATCH --job-name=train_A1_e2e_dadv
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_A1_e2e_dadv_%j.out
#SBATCH --error=logs/train_A1_e2e_dadv_%j.err

echo "============================================================"
echo "  A1_e2e_dadv — A1 + Domain Adversarial, Full E2E"
echo "  AV-HuBERT fully unfrozen, ~63k AV1M training clips"
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

mkdir -p avh_sup/outputs_A1_e2e_dadv/ckpts avh_sup/outputs_A1_e2e_dadv/logs logs

cd avh_sup

# Step 1: Generate comprehensive CSVs (if not already done)
if [ ! -f csv_metadata/av1m_e2e/train_e2e_full.csv ]; then
    echo ">>> Generating comprehensive AV1M E2E CSV..."
    $PYTHON generate_e2e_csv.py
    echo ">>> CSV generation done."
fi

# Step 2: Train
echo ">>> Starting E2E training for A1 + Domain Adversarial..."
$PYTHON train_e2e_a1_dadv.py --config_path configs/A1_e2e_dadv.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A1_e2e_dadv/ckpts/
echo "============================================================"
