#!/bin/bash
#SBATCH --job-name=A6_avld
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/A6_avlips_domain_seeds_%j.out
#SBATCH --error=logs/A6_avlips_domain_seeds_%j.err

echo "============================================================"
echo "  A6-AVLips-Domain — seeds 43/44/45"
echo "  Domain signal: AVLips real clips (zero FAVC exposure)"
echo "  Diagnostic: does domain push-pull help without FAVC data?"
echo "  Start: $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
cd "$BASE"

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

mkdir -p avh_sup/outputs_A6_avlips_domain_seed43/{logs,ckpts,results} \
         avh_sup/outputs_A6_avlips_domain_seed44/{logs,ckpts,results} \
         avh_sup/outputs_A6_avlips_domain_seed45/{logs,ckpts,results} \
         logs

cd avh_sup

echo ""
echo "--- Seed 43 ---"
$PYTHON train_test_fcd_a6_avlips_domain.py \
  --config_path configs/A6_avlips_domain_seed43.yaml

echo ""
echo "--- Seed 44 ---"
$PYTHON train_test_fcd_a6_avlips_domain.py \
  --config_path configs/A6_avlips_domain_seed44.yaml

echo ""
echo "--- Seed 45 ---"
$PYTHON train_test_fcd_a6_avlips_domain.py \
  --config_path configs/A6_avlips_domain_seed45.yaml

echo ""
echo "============================================================"
echo "  All seeds done. $(date)"
echo "============================================================"
