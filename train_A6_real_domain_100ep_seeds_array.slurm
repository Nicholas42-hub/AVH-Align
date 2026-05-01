#!/bin/bash
#SBATCH --job-name=A6_real100e
#SBATCH --partition=deeplearn
#SBATCH --account=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=43,44,45
#SBATCH --output=logs/A6_real_100ep_s%a_%A.out
#SBATCH --error=logs/A6_real_100ep_s%a_%A.err

SEED=$SLURM_ARRAY_TASK_ID

echo "============================================================"
echo "  A6 real-target-domain (AV1M=0 vs FAVC real=1) 100ep | seed=${SEED}"
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

OUTPUT_DIR="$BASE/avh_sup/outputs_A6_real_domain_100ep_seed${SEED}"
mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results} logs

cd avh_sup

"$PYTHON" train_test_fcd_a6.py \
  --config_path "configs/A6_real_domain_100ep_seed${SEED}.yaml" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "  Done: A6 real-domain 100ep seed=${SEED} at $(date)"
echo "============================================================"
