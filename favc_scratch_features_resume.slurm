#!/bin/bash
#SBATCH --job-name=favc_feat_cat
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/favc_feat_resume_%x_%j.out
#SBATCH --error=logs/favc_feat_resume_%x_%j.err

# ============================================================
#  FakeAVCeleb Feature Extraction — Resume by category
#
#  Usage: sbatch --job-name=<category> favc_scratch_features_resume.slurm <category>
#  OR:    CATEGORY=<category> sbatch favc_scratch_features_resume.slurm
#
#  Valid categories:
#    RealVideo-RealAudio
#    RealVideo-FakeAudio
#    FakeVideo-RealAudio
#    FakeVideo-FakeAudio
#
#  Already-extracted NPZ files are skipped automatically.
# ============================================================

# Accept category from first positional arg or env var
CATEGORY=${1:-${CATEGORY:-all}}

echo "============================================================"
echo "  FakeAVCeleb Feature Extraction (Resume)"
echo "  Category : $CATEGORY"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURM_NODELIST"
echo "  Start    : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
PROJECT_ROOT=/data/projects/punim2637/nnliang/AVH-Align

DATA_PATH=$SCRATCH_ROOT/favc_preprocessed
SAVE_PATH=$SCRATCH_ROOT/favc_features_temp
FINAL_PATH=$PROJECT_ROOT/data/favc_features

METADATA=av1m_metadata/favc_metadata.csv
CKPT=self_large_vox_433h.pt

echo ""
echo "Configuration:"
echo "  Category : $CATEGORY"
echo "  Metadata : $METADATA"
echo "  Data (SCRATCH): $DATA_PATH"
echo "  Save (SCRATCH): $SAVE_PATH"
echo "  Final (PROJECTS): $FINAL_PATH"
echo ""

# Count how many are already done for this category
if [ "$CATEGORY" = "all" ]; then
    EXISTING=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
    PREP=$(find "$DATA_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
else
    EXISTING=$(find "$SAVE_PATH/$CATEGORY" -name "*.npz" 2>/dev/null | wc -l)
    PREP=$(find "$DATA_PATH/$CATEGORY" -name "*_roi.mp4" 2>/dev/null | wc -l)
fi
echo "Already extracted (SCRATCH): $EXISTING"
echo "Preprocessed videos         : $PREP"
echo ""

if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: Checkpoint not found: av_hubert/avhubert/$CKPT"
    exit 1
fi

mkdir -p "$SAVE_PATH" "$FINAL_PATH" logs

cd av_hubert/avhubert

echo "============================================================"
echo "Step 1/2: Feature extraction (skipping already done)..."
echo "============================================================"
echo ""

python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --category "$CATEGORY"

cd /data/projects/punim2637/nnliang/AVH-Align

DONE=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
echo ""
echo "Total NPZ in SCRATCH after extraction: $DONE"
echo ""

echo "============================================================"
echo "Step 2/2: Syncing NPZ files to PROJECTS (permanent storage)..."
echo "============================================================"
echo ""

rsync -av --progress --include="*/" --include="*.npz" --exclude="*" \
    "$SAVE_PATH/" "$FINAL_PATH/" | tail -20

FINAL_COUNT=$(find "$FINAL_PATH" -name "*.npz" 2>/dev/null | wc -l)

echo ""
echo "============================================================"
echo "Completed: $(date)"
echo ""
echo "Summary:"
echo "  Category processed : $CATEGORY"
echo "  NPZ in PROJECTS    : $FINAL_COUNT / 21544"
echo "============================================================"
echo ""
echo "✅ NPZ files are now in PROJECTS (permanent)"
echo "⚠️  Temporary files remain in SCRATCH (will be auto-cleaned)"
echo "============================================================"
