#!/bin/bash
#SBATCH --job-name=avh_e2e_eval
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_e2e_domain_%j.out
#SBATCH --error=logs/eval_e2e_domain_%j.err

echo "============================================================"
echo "  E2E Domain & Label Probe Evaluation"
echo "  Job : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start: $(date)"
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

CKPT="avh_sup/outputs_A2_e2e/ckpts/model-epoch=14-val_auc_causal=0.9949.ckpt"
CONFIG="avh_sup/configs/A2_e2e.yaml"
AV1M_RAW="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
FAVC_RAW="/data/scratch/projects/punim2637/nnliang/favc_preprocessed"
AV1M_CSV="avh_sup/csv_metadata/av1m"
OUTPUT_DIR="avh_sup/outputs_A2_e2e/results"

mkdir -p logs "$OUTPUT_DIR"

cd avh_sup

echo "Running domain + label probe on e2e checkpoint..."
python eval_e2e_domain.py \
    --ckpt      "../$CKPT" \
    --config    "../$CONFIG" \
    --av1m_raw  "$AV1M_RAW" \
    --favc_raw  "$FAVC_RAW" \
    --av1m_csv  "../$AV1M_CSV" \
    --output_dir "../$OUTPUT_DIR" \
    --holdout_only \
    --max_clips 500 \
    --batch_size 8 \
    --num_workers 2

echo ""
echo "============================================================"
echo "  Eval done: $(date)"
echo "============================================================"
