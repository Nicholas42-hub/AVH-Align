#!/bin/bash
#SBATCH --job-name=headabl_A6
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/head_ablation_A6_%j.out
#SBATCH --error=logs/head_ablation_A6_%j.err

echo "=== Head Ablation A6  Job=$SLURM_JOB_ID  $(date) ==="
cd /data/projects/punim2637/nnliang/AVH-Align
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
cd avh_sup
$PYTHON eval_head_ablation.py \
    --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \
    --config     configs/A6.yaml \
    --av1m_root  ../data/avh_features/val \
    --av1m_csv   csv_metadata/av1m/val_labels.csv \
    --favc_root  ../data/favc_features \
    --output_dir outputs_A6/results \
    --device     cuda
echo "=== Done $(date) ==="
