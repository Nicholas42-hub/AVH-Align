#!/bin/bash
#SBATCH --job-name=modality_origin_A2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/modality_origin_A2_%j.out
#SBATCH --error=logs/modality_origin_A2_%j.err

echo "============================================================"
echo "  Modality-of-Origin Probe — A2 (Binary)"
echo "  FV-RA (video fake) vs RV-FA (audio fake) from Z_c"
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

mkdir -p avh_sup/outputs_A2/results logs

cd avh_sup

CKPT=$(ls outputs_A2/ckpts/*.ckpt | sort | tail -1)
echo "Using checkpoint: $CKPT"

$PYTHON eval_modality_origin_probe.py \
    --ckpt_path  "$CKPT" \
    --config     "configs/A2.yaml" \
    --favc_root  "../data/favc_features" \
    --output_dir "outputs_A2/results" \
    --device     cuda

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
