#!/bin/bash
#SBATCH --job-name=perturb_base
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/perturb_baseline_%j.out
#SBATCH --error=logs/perturb_baseline_%j.err

# ============================================================
#  Perturbation Robustness — AVH-Align Baseline (paper model)
#
#  Experiment 3: AUC under Gaussian noise on audio/visual features
#  σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}
#
#  This is the original AVH-Align supervised model (no causal split),
#  used as the paper benchmark baseline.
# ============================================================

echo "============================================================"
echo "  Perturbation Robustness — AVH-Align Baseline (paper model)"
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
CKPT="outputs_baseline/ckpts/model-epoch=05.ckpt"
OUTPUT_DIR="outputs_baseline/results/perturbation"

mkdir -p "avh_sup/$OUTPUT_DIR" logs

cd avh_sup

echo "--- Running perturbation robustness for AVH-Align baseline ---"
echo "  Checkpoint: $CKPT"
python eval_perturbation_robustness.py \
    --ckpt_path      "$CKPT" \
    --model          baseline \
    --config         configs/baseline.yaml \
    --favc_root      "$FAVC_ROOT" \
    --csv_root_path  "$CSV_ROOT" \
    --split          test \
    --batch_size     32 \
    --seed           42 \
    --output_dir     "$OUTPUT_DIR" \
    --device         cuda

echo ""
echo "============================================================"
echo "  Baseline perturbation done: $(date)"
echo "  Results: avh_sup/$OUTPUT_DIR"
echo "============================================================"
