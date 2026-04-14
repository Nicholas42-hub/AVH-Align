#!/bin/bash
#SBATCH --job-name=A6_shared_tr
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/A6_sharedonly_trainreal_%j.out
#SBATCH --error=logs/A6_sharedonly_trainreal_%j.err

echo "============================================================"
echo "  A6-sharedonly-trainreal"
echo "  Domain GRL on [s_v, s_a] only — u_v/u_a protected"
echo "  Domain source: FAVC train-real (favc_npz_matched)"
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

mkdir -p avh_sup/outputs_A6_sharedonly_trainreal/{logs,ckpts,results} logs

cd avh_sup
python train_test_fcd_a6_sharedonly.py \
  --config_path configs/A6_sharedonly_trainreal.yaml \
  --mode train

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
