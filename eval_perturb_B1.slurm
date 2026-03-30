#!/bin/bash
#SBATCH --job-name=perturb_B1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/perturb_B1_%j.out
#SBATCH --error=logs/perturb_B1_%j.err

# ============================================================
#  Perturbation Robustness — B1 (Branch 1: per-modal classification)
#
#  Experiment 3: AUC under Gaussian noise on audio/visual features
#  σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}
#
#  Hypothesis:
#    B1 with modal balance should show more symmetric degradation
#    between audio_noise and visual_noise than A1/A2 (which were
#    audio-dominant, so visual_noise caused less degradation).
# ============================================================

echo "============================================================"
echo "  Perturbation Robustness — B1 (Branch 1)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
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

FAVC_ROOT="../data/favc_features"
CSV_ROOT="csv_metadata/favc"
CKPT="outputs_B1/ckpts/model-epoch=13.ckpt"
OUTPUT_DIR="outputs_B1/results/perturbation"

mkdir -p "avh_sup/$OUTPUT_DIR" logs

cd avh_sup

echo "--- Running perturbation robustness for B1 ---"
echo "  Checkpoint: $CKPT"
python eval_perturbation_robustness.py \
    --ckpt_path      "$CKPT" \
    --model          modal \
    --config         configs/B1.yaml \
    --favc_root      "$FAVC_ROOT" \
    --csv_root_path  "$CSV_ROOT" \
    --split          test \
    --batch_size     32 \
    --seed           42 \
    --output_dir     "$OUTPUT_DIR" \
    --device         cuda

echo ""
echo "============================================================"
echo "  B1 perturbation done: $(date)"
echo "  Results: avh_sup/$OUTPUT_DIR"
echo "============================================================"
