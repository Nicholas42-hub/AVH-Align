#!/bin/bash
#SBATCH --job-name=bprobe_v2_A5
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/branch_probe_v2_A5_%j.out
#SBATCH --error=logs/branch_probe_v2_A5_%j.err

echo "=== Branch Probe v2 (masking+noise) A5  Job=$SLURM_JOB_ID  $(date) ==="
cd /data/projects/punim2637/nnliang/AVH-Align
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
cd avh_sup
$PYTHON eval_branch_probe_perturb.py     --ckpt_path  outputs_A5/ckpts/model-epoch=17.ckpt     --config     configs/A5.yaml     --av1m_root  ../data/avh_features/val     --av1m_csv   csv_metadata/av1m/val_labels.csv     --output_dir outputs_A5/results     --device     cuda
echo "=== Done $(date) ==="
