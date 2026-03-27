#!/bin/bash
#SBATCH --job-name=cv_probe_A29
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/cv_probe_A29_%j.out
#SBATCH --error=logs/cv_probe_A29_%j.err

echo "============================================================"
echo "  CV probe eval for A29"
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

mkdir -p logs
cd avh_sup

CKPT="outputs_A29_e2e/ckpts/last.ckpt"
echo ">>> Probing checkpoint: $CKPT"
$PYTHON eval_cv_probe_a29.py \
    --config configs/A29_e2e.yaml \
    --ckpt   "$CKPT" \
    --n_probe 400 \
    --n_folds 5

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
