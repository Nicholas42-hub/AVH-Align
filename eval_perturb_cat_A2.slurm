#!/bin/bash
#SBATCH --job-name=perturb_cat_A2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/perturb_cat_A2_%j.out
#SBATCH --error=logs/perturb_cat_A2_%j.err

echo "=== A2 Per-Category Perturbation (full types)  Job=$SLURM_JOB_ID  $(date) ==="
cd /data/projects/punim2637/nnliang/AVH-Align
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
cd avh_sup

CKPT="outputs_A2/ckpts/model-epoch=09.ckpt"

echo ""
echo "=== A2 | Perturbation — RealVideo-FakeAudio (all types) ==="
$PYTHON eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     ../data/favc_features \
    --csv_root_path csv_metadata/favc \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results \
    --device        cuda

echo ""
echo "=== A2 | Perturbation — FakeVideo-RealAudio (all types) ==="
$PYTHON eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     ../data/favc_features \
    --csv_root_path csv_metadata/favc \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results \
    --device        cuda

echo "=== Done $(date) ==="
