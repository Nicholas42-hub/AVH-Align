#!/bin/bash
#SBATCH --job-name=manip_probe_A6
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/manip_probe_A6_%j.out
#SBATCH --error=logs/manip_probe_A6_%j.err

echo "============================================================"
echo "  A6 Manipulation-Type Probe"
echo "  Task: FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)"
echo "  Job ID : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
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

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT="$BASE/avh_sup/outputs_A6/ckpts/model-epoch=15.ckpt"

mkdir -p avh_sup/outputs_A6/results logs

cd avh_sup

python eval_manipulation_probe_a6.py \
    --ckpt_path     "$CKPT" \
    --config        configs/A6.yaml \
    --favc_root     ../data/favc_features \
    --max_per_class 1000 \
    --n_folds       5 \
    --seed          42 \
    --output_dir    outputs_A6/results \
    --device        cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
