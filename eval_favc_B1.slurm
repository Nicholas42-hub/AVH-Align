#!/bin/bash
#SBATCH --job-name=eval_favc_B1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_favc_B1_%j.out
#SBATCH --error=logs/eval_favc_B1_%j.err

# ============================================================
#  Cross-dataset evaluation: B1 (Branch 1) -> FakeAVCeleb
#  Checkpoint: outputs_B1/ckpts/model-epoch=13.ckpt  (best val_auc_causal)
#  Reports per-category AUC for:
#    full / causal / spurious heads
#    × RealVideo-FakeAudio / FakeVideo-RealAudio / FakeVideo-FakeAudio
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Cross-Eval — B1 (Branch 1: per-modal classification)"
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

CKPT="avh_sup/outputs_B1/ckpts/model-epoch=13.ckpt"
FAVC_FEATURES="data/favc_features"
CSV_ROOT="avh_sup/csv_metadata/favc"
OUTPUT_DIR="avh_sup/outputs_B1/results/favc_eval"

mkdir -p "$OUTPUT_DIR" logs

cd avh_sup
python eval_favc.py \
    --ckpt        "../$CKPT" \
    --model       modal \
    --features_path "../$FAVC_FEATURES" \
    --csv_root_path "$CSV_ROOT" \
    --split       test \
    --output_dir  "../$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  B1 FakeAVCeleb eval done: $(date)"
echo "  Results: $OUTPUT_DIR/eval_results.txt"
echo "============================================================"
