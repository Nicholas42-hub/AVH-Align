#!/bin/bash
#SBATCH --job-name=avhsup_favc_av1m
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_avhsup_favc_av1m_%j.out
#SBATCH --error=logs/eval_avhsup_favc_av1m_%j.err

# ============================================================
#  Evaluate AVH-Align/sup FAVC-trained checkpoint on AV1M val.
#  Checkpoint: checkpoints/avh_sup/AVH_Sup_FAVC.ckpt (from paper)
#  Expected result: ~69.0% AUC (Smeu et al., CVPR 2025, Table 2)
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT="$BASE/checkpoints/avh_sup/AVH_Sup_FAVC.ckpt"
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
OUTPUT_DIR="$BASE/avh_sup/outputs_avhsup_favc/results_av1m_val"

echo "============================================================"
echo "  AVH-Align/sup (FAVC-trained) → AV1M val"
echo "  Ckpt   : $CKPT"
echo "  Out    : $OUTPUT_DIR"
echo "  Start  : $(date)"
echo "============================================================"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUTPUT_DIR" logs

cd avh_sup

$PYTHON eval_av1m.py \
    --ckpt "$CKPT" \
    --model baseline \
    --features_path "$BASE/data/avh_features" \
    --csv_root_path "$BASE/avh_sup/csv_metadata/av1m" \
    --split val \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Done at $(date)"
echo "============================================================"
