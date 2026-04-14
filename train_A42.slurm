#!/bin/bash
#SBATCH --job-name=train_A42
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A42_%j.out
#SBATCH --error=logs/train_A42_%j.err

echo "============================================================"
echo "  A42 Training — Z_c/Z_s Frozen + Domain Adversarial"
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

mkdir -p avh_sup/outputs_A42/ckpts avh_sup/outputs_A42/logs avh_sup/outputs_A42/results logs

echo ""
echo "=== Architecture Summary ==="
echo "  Z_c/Z_s projection heads (frozen AV-HuBERT encoder)"
echo "  Losses: L_task + lambda_adv*L_adv + lambda_dadv*L_dadv + lambda_ddis*L_ddis + lambda_orth*L_orth"
echo "  Domain data: AV1M train (domain=0) + FAVC val-real x14 (domain=1)"
echo ""

cd avh_sup
$PYTHON train_test_causal_a42.py --config_path configs/A42.yaml --mode train

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Best ckpt in: outputs_A42/ckpts/"
echo "============================================================"
