#!/bin/bash
#SBATCH --job-name=eval_manip
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_manip_%j.out
#SBATCH --error=logs/eval_manip_%j.err

echo "============================================================"
echo "  A2_e2e_manip Evaluation"
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

mkdir -p avh_sup/outputs_A2_e2e_manip/results logs

cd avh_sup

CKPT="outputs_A2_e2e_manip/ckpts/model-epoch=14-val_auc_causal=0.9961.ckpt"

python eval_e2e_manip.py \
    --ckpt        "$CKPT" \
    --config      configs/A2_e2e_manip.yaml \
    --av1m_raw    /data/scratch/projects/punim2637/nnliang/avd1m_preprocessed \
    --favc_raw    /data/scratch/projects/punim2637/nnliang/favc_preprocessed \
    --av1m_csv    csv_metadata/av1m \
    --output_dir  outputs_A2_e2e_manip/results \
    --max_frames  150 \
    --batch_size  8 \
    --max_clips   500 \
    --holdout_only

echo ""
echo "============================================================"
echo "  Eval done: $(date)"
echo "============================================================"
