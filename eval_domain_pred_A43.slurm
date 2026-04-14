#!/bin/bash
#SBATCH --job-name=dom_pred_A43
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dom_pred_A43_%j.out
#SBATCH --error=logs/dom_pred_A43_%j.err

echo "============================================================"
echo "  Domain Predictability Probe — A43"
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
CKPT=$(ls "$BASE/avh_sup/outputs_A43/ckpts"/model-epoch=*.ckpt | sort | tail -1)

mkdir -p avh_sup/outputs_A43/results logs
echo "Using checkpoint: $CKPT"

cd avh_sup

python eval_domain_predictability.py \
    --ckpt_path       "$CKPT" \
    --config          configs/A43.yaml \
    --av1m_root       ../data/avh_features/val \
    --favc_root       ../data/favc_features \
    --max_per_dataset 2000 \
    --n_folds         5 \
    --seed            42 \
    --output_dir      outputs_A43/results \
    --device          cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
