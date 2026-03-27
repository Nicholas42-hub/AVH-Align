#!/bin/bash
#SBATCH --job-name=domain_pred_A7
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/domain_pred_A7_%j.out
#SBATCH --error=logs/domain_pred_A7_%j.err

echo "============================================================"
echo "  Domain Predictability Probe — A7 (FCD + Domain Adv + Label Adv)"
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

mkdir -p avh_sup/outputs_A7/results logs

cd avh_sup

CKPT=$(ls outputs_A7/ckpts/*.ckpt | sort | tail -1)
echo "Using checkpoint: $CKPT"

$PYTHON eval_domain_predictability.py \
    --ckpt_path     "$CKPT" \
    --config        "configs/A7.yaml" \
    --av1m_root     "../data/avh_features" \
    --favc_root     "../data/favc_features" \
    --output_dir    "outputs_A7/results" \
    --device        cuda

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
