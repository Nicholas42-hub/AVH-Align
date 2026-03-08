#!/bin/bash
#SBATCH --job-name=eval_A1_favc
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_A1_favc_%j.out
#SBATCH --error=logs/eval_A1_favc_%j.err

# ============================================================
#  Cross-domain evaluation: Ablation A1 → FakeAVCeleb
#
#  A1: Adversarial loss only (full_head on Z_c + Z_s)
#  Checkpoint: outputs_ablation_a1/ckpts/model-epoch=03.ckpt
# ============================================================

echo "============================================================"
echo "  AVH-Align Ablation A1 → FakeAVCeleb Evaluation"
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

CKPT="avh_sup/outputs_A1/ckpts/model-epoch=03.ckpt"
FEATURES="data/favc_features"
CSV_ROOT="avh_sup/csv_metadata/favc"
OUTPUT="avh_sup/outputs_A1/results_favc"

mkdir -p "$OUTPUT" logs

echo "Checkpoint : $CKPT"
echo "Features   : $FEATURES"
echo "Output     : $OUTPUT"
echo ""

cd avh_sup
python eval_favc.py \
    --ckpt          "../$CKPT" \
    --model         ablation \
    --features_path "../$FEATURES" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$OUTPUT"

echo ""
echo "============================================================"
echo "  Evaluation done: $(date)"
echo "  Results: /data/projects/punim2637/nnliang/AVH-Align/$OUTPUT"
echo "============================================================"
