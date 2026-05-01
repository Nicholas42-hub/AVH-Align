#!/bin/bash
#SBATCH --job-name=prep_testB
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-49
#SBATCH --output=/data/projects/punim2637/nnliang/logs/prep_testB_%A_%a.out
#SBATCH --error=/data/projects/punim2637/nnliang/logs/prep_testB_%A_%a.err

echo "============================================================"
echo "  Preprocess testB — shard ${SLURM_ARRAY_TASK_ID}/50"
echo "  Node  : $SLURM_NODELIST"
echo "  Start : $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVHUB_DIR="$BASE/av_hubert/avhubert"

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$AVHUB_DIR"
export PYTHONUNBUFFERED=1
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

cd "$AVHUB_DIR"

$PYTHON "$BASE/preprocess_testB.py" \
    --shard_idx "$SLURM_ARRAY_TASK_ID" \
    --num_shards 50 \
    --max_workers "$SLURM_CPUS_PER_TASK"

echo "============================================================"
echo "  Done: shard ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "============================================================"
