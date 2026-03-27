#!/bin/bash
#SBATCH --job-name=train_A2_e2e_dadv
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_A2_e2e_dadv_%j.out
#SBATCH --error=logs/train_A2_e2e_dadv_%j.err

# ============================================================
#  A2_e2e_dadv — A2 (causal_head only) + Domain Adversarial
#
#  use_sg=True:  classification via causal_head(Z_c) only
#                → Z_s cannot compensate at test time
#                → encoder forced to make Z_c truly domain-invariant
#  use_domain=True: GRL on Z_c (push) + direct on Z_s (pull)
#  Finetune from scratch (fully unfrozen, lr_enc=1e-6)
#
#  Expected improvement over A1_dadv cross-domain AUC 0.763
# ============================================================

echo "============================================================"
echo "  A2_e2e_dadv — A2 + Domain Adversarial, Full E2E"
echo "  use_sg=True (causal_head only) + use_domain=True"
echo "  AV-HuBERT fully unfrozen, ~63k AV1M + 5k FAVC clips"
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

mkdir -p avh_sup/outputs_A2_e2e_dadv/ckpts avh_sup/outputs_A2_e2e_dadv/logs logs

cd avh_sup

# Generate CSV if not already present
if [ ! -f csv_metadata/av1m_e2e/train_e2e_full.csv ]; then
    echo ">>> Generating comprehensive AV1M E2E CSV..."
    $PYTHON generate_e2e_csv.py
    echo ">>> CSV generation done."
fi

echo ">>> Starting A2_e2e_dadv training (causal_head only + domain adversarial)..."
$PYTHON train_e2e_a1_dadv.py --config_path configs/A2_e2e_dadv.yaml

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A2_e2e_dadv/ckpts/
echo "============================================================"
