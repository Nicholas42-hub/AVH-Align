#!/bin/bash
# Scratch Space Management for FakeAVCeleb Processing
# ============================================================

SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
PROJECT_ROOT=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  FakeAVCeleb Scratch Space Status"
echo "  $(date)"
echo "============================================================"
echo ""

# Overall scratch usage
echo "📊 Total Scratch Usage:"
check_scratch_usage
echo ""

# Dataset location
echo "📁 Dataset Location:"
DATASET_SIZE=$(du -sh "$SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2" 2>/dev/null | cut -f1)
DATASET_COUNT=$(find "$SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2" -name "*.mp4" 2>/dev/null | wc -l)
echo "  SCRATCH: $SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2"
echo "  Size: $DATASET_SIZE"
echo "  Videos: $DATASET_COUNT"
echo ""

# Preprocessed files (temporary in scratch)
echo "🔄 Preprocessed Files (Temporary):"
if [ -d "$SCRATCH_ROOT/favc_preprocessed" ]; then
    PREP_SIZE=$(du -sh "$SCRATCH_ROOT/favc_preprocessed" 2>/dev/null | cut -f1)
    PREP_COUNT=$(find "$SCRATCH_ROOT/favc_preprocessed" -name "*.mp4" 2>/dev/null | wc -l)
    echo "  Location: $SCRATCH_ROOT/favc_preprocessed"
    echo "  Size: $PREP_SIZE"
    echo "  Files: $PREP_COUNT / 21,544"
else
    echo "  Not started yet"
fi
echo ""

# Feature files (temporary in scratch)
echo "⚡ Feature Files (Temporary in Scratch):"
if [ -d "$SCRATCH_ROOT/favc_features_temp" ]; then
    FEAT_TEMP_SIZE=$(du -sh "$SCRATCH_ROOT/favc_features_temp" 2>/dev/null | cut -f1)
    FEAT_TEMP_COUNT=$(find "$SCRATCH_ROOT/favc_features_temp" -name "*.npz" 2>/dev/null | wc -l)
    echo "  Location: $SCRATCH_ROOT/favc_features_temp"
    echo "  Size: $FEAT_TEMP_SIZE"
    echo "  Files: $FEAT_TEMP_COUNT"
else
    echo "  Not started yet"
fi
echo ""

# Final NPZ files (permanent in projects)
echo "💾 NPZ Files (Permanent in Projects):"
if [ -d "$PROJECT_ROOT/data/favc_features" ]; then
    FEAT_FINAL_SIZE=$(du -sh "$PROJECT_ROOT/data/favc_features" 2>/dev/null | cut -f1)
    FEAT_FINAL_COUNT=$(find "$PROJECT_ROOT/data/favc_features" -name "*.npz" 2>/dev/null | wc -l)
    echo "  Location: $PROJECT_ROOT/data/favc_features"
    echo "  Size: $FEAT_FINAL_SIZE"
    echo "  Files: $FEAT_FINAL_COUNT / 21,544"
else
    echo "  Not started yet"
fi
echo ""

# Current jobs
echo "🔄 Current Jobs:"
squeue -u nnliang --format="  %.10i %.12P %.20j %.8T %.10M %R"
echo ""

echo "============================================================"
echo ""
echo "💡 Commands:"
echo "  Start preprocessing:  sbatch favc_scratch_preprocess.slurm"
echo "  Start features:       sbatch favc_scratch_features.slurm"
echo "  Check progress:       ./check_favc_scratch.sh"
echo "  View logs:            tail -f logs/favc_scratch_*_<jobid>.out"
echo ""
