#!/bin/bash
#SBATCH --job-name=train_A41_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A41_e2e_%j.out
#SBATCH --error=logs/train_A41_e2e_%j.err

echo "============================================================"
echo "  A41_e2e — A40 + Z_s label adversarial (use_adv=True)"
echo "  L_task + L_adv_zs(GRL) + L_ladv_ds + L_mean_zc"
echo "  Full causal/spurious separation: task-irrelevant Z_s"
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

BASE=/data/projects/punim2637/nnliang/AVH-Align
mkdir -p "$BASE/avh_sup/outputs_A41_e2e/ckpts" \
         "$BASE/avh_sup/outputs_A41_e2e/logs"  \
         "$BASE/avh_sup/outputs_A41_e2e/results" \
         "$BASE/logs"

cd "$BASE/avh_sup"

echo ">>> Starting A41 training..."
$PYTHON train_e2e_a41.py \
    --config_path "$BASE/avh_sup/configs/A41_e2e.yaml"

echo ""
echo "============================================================"
echo "  A41_e2e finished"
echo "  End : $(date)"
echo "============================================================"
