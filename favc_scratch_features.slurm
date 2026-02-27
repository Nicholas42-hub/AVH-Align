#!/bin/bash
#SBATCH --job-name=favc_scratch_feat
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/favc_scratch_features_%j.out
#SBATCH --error=logs/favc_scratch_features_%j.err

# ============================================================
#  Extract Features using SCRATCH, then copy NPZ to PROJECTS
#  
#  Input:   /data/scratch/.../favc_preprocessed (temporary)
#  Output:  /data/scratch/.../favc_features_temp (temporary)
#  Final:   /data/projects/.../data/favc_features (permanent NPZ only)
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Feature Extraction (Scratch → Projects)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

# Use conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Set environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Use SCRATCH for processing, PROJECTS for final NPZ storage
SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
PROJECT_ROOT=/data/projects/punim2637/nnliang/AVH-Align

DATA_PATH=$SCRATCH_ROOT/favc_preprocessed
SAVE_PATH=$SCRATCH_ROOT/favc_features_temp
FINAL_PATH=$PROJECT_ROOT/data/favc_features

METADATA=av1m_metadata/favc_metadata.csv
CKPT=self_large_vox_433h.pt

echo ""
echo "Configuration:"
echo "  Metadata: $METADATA"
echo "  Data path (SCRATCH): $DATA_PATH"
echo "  Temp save (SCRATCH): $SAVE_PATH"
echo "  Final save (PROJECTS): $FINAL_PATH"
echo "  Checkpoint: $CKPT"
echo ""

# Count preprocessed videos
PREP_COUNT=$(find "$DATA_PATH" -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "Preprocessed videos: $PREP_COUNT"

if [ ! -f "av_hubert/avhubert/$CKPT" ]; then
    echo "ERROR: Checkpoint not found: av_hubert/avhubert/$CKPT"
    exit 1
fi

mkdir -p "$SAVE_PATH" "$FINAL_PATH" logs

cd av_hubert/avhubert

echo ""
echo "============================================================"
echo "Step 1/2: Feature extraction to SCRATCH..."
echo "============================================================"
echo ""

python deepfake_feature_extraction.py \
    --dataset FakeAVCeleb \
    --metadata /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --ckpt_path "$CKPT" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH"

DONE=$(find "$SAVE_PATH" -name "*.npz" 2>/dev/null | wc -l)
echo ""
echo "Feature files created: $DONE"
echo ""

echo "============================================================"
echo "Step 2/2: Copying NPZ files to PROJECTS (permanent storage)..."
echo "============================================================"
echo ""

# Copy NPZ files from scratch to projects
rsync -av --progress --include="*/" --include="*.npz" --exclude="*" \
    "$SAVE_PATH/" "$FINAL_PATH/" | tail -20

FINAL_COUNT=$(find "$FINAL_PATH" -name "*.npz" 2>/dev/null | wc -l)

echo ""
echo "============================================================"
echo "Feature extraction completed: $(date)"
echo ""
echo "Summary:"
echo "  Features extracted: $DONE"
echo "  NPZ files in PROJECTS: $FINAL_COUNT"
echo ""
echo "Storage usage:"
echo "  SCRATCH:"
check_scratch_usage
echo ""
echo "  PROJECTS:"
du -sh "$FINAL_PATH"
echo ""
echo "============================================================"
echo ""
echo "✅ NPZ files are now in PROJECTS (permanent)"
echo "⚠️  Temporary files remain in SCRATCH (will be auto-cleaned)"
echo ""
echo "Next step: Run evaluation with:"
echo "  sbatch eval_favc_full.slurm"
echo "============================================================"
