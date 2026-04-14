#!/bin/bash
#SBATCH --job-name=domain_probe
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/domain_probe_%j.out
#SBATCH --error=logs/domain_probe_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

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

cd "$BASE/avh_sup"

echo "============================================================"
echo "  Domain Probe — A42 vs A43"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Start  : $(date)"
echo "============================================================"

echo ""
echo "=== A42 ==="
$PYTHON run_domain_probe.py \
    --ckpt   "$BASE/avh_sup/outputs_A42/ckpts/model-epoch=05.ckpt" \
    --model  causal_a42

echo ""
echo "=== A43 ==="
$PYTHON run_domain_probe.py \
    --ckpt   "$BASE/avh_sup/outputs_A43/ckpts/model-epoch=05.ckpt" \
    --model  causal_a43

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
