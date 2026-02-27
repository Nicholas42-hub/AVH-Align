#!/bin/bash
#SBATCH --job-name=favc_features
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/favc_features_%j.out
#SBATCH --error=logs/favc_features_%j.err

# ============================================================
#  Step 2: Extract AV-HuBERT features for FakeAVCeleb
#
#  Reads preprocessed ROI videos + wavs from data/favc_preprocessed/
#  Saves .npz files to data/favc_features/
#  (one NPZ per clip: keys visual [T×1024] audio [T×1024] multimodal [T×1024])
#
#  Usage:
#    sbatch favc_extract_features.slurm
#  Prerequisite:
#    favc_preprocess.slurm must have completed first.
#    av1m_metadata/favc_metadata.csv must exist.
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Feature Extraction"
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
echo "Python: $(which python) — $(python --version)"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

METADATA=av1m_metadata/favc_metadata.csv
DATA_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/favc_preprocessed
SAVE_PATH=/data/projects/punim2637/nnliang/AVH-Align/data/favc_features
CKPT=self_large_vox_433h.pt

echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Checkpoint : $CKPT"
echo "Clips in metadata: $(tail -n +2 $METADATA | wc -l)"
echo ""

# Sanity checks
if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: AV-HuBERT checkpoint not found: av_hubert/avhubert/$CKPT"
    exit 1
fi
if [ ! -f "$METADATA" ]; then
    echo "ERROR: Metadata not found: $METADATA"
    echo "       Run favc_preprocess.slurm first."
    exit 1
fi

mkdir -p "$SAVE_PATH" logs

# Run from avhubert directory (required for relative imports of hubert modules)
cd av_hubert/avhubert

python deepfake_feature_extraction.py \
    --dataset   FakeAVCeleb \
    --category  all \
    --metadata  /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH"

echo ""
echo "============================================================"
echo "  Feature extraction done: $(date)"
DONE=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
echo "  Feature files ready: $DONE"
echo "  Category breakdown:"
for cat in RealVideo-RealAudio RealVideo-FakeAudio FakeVideo-RealAudio FakeVideo-FakeAudio; do
    N=$(find "$SAVE_PATH/$cat" -name "*.npz" 2>/dev/null | wc -l)
    echo "    $cat : $N"
done
echo "============================================================"
