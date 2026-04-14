#!/bin/bash
# Scratch Space Management for AVDeepfake1M Processing
# ============================================================

SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
PROJECT_ROOT=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  AVDeepfake1M Scratch Space Status"
echo "  $(date)"
echo "============================================================"
echo ""

# Overall scratch usage
echo "📊 Total Scratch Usage:"
check_scratch_usage
echo ""

# Dataset location
echo "📁 AVDeepfake1M Dataset:"
DATASET_SIZE=$(du -sh "$SCRATCH_ROOT/Datasets/AVDeepfake1M" 2>/dev/null | cut -f1)
VAL_COUNT=$(find "$SCRATCH_ROOT/Datasets/AVDeepfake1M/val" -name "*.mp4" 2>/dev/null | wc -l)
echo "  SCRATCH: $SCRATCH_ROOT/Datasets/AVDeepfake1M"
echo "  Size: $DATASET_SIZE"
echo "  Val videos: $VAL_COUNT"
echo ""

# Preprocessed files (temporary in scratch)
echo "🔄 Preprocessed Files (Temporary in Scratch):"
if [ -d "$SCRATCH_ROOT/avd1m_preprocessed" ]; then
    PREP_SIZE=$(du -sh "$SCRATCH_ROOT/avd1m_preprocessed" 2>/dev/null | cut -f1)
    PREP_COUNT=$(find "$SCRATCH_ROOT/avd1m_preprocessed" -name "*_roi.mp4" 2>/dev/null | wc -l)
    PREP_PCT=$(awk "BEGIN {printf \"%.1f%%\", $PREP_COUNT/57340*100}")
    echo "  Location: $SCRATCH_ROOT/avd1m_preprocessed"
    echo "  Size: $PREP_SIZE"
    echo "  Files: $PREP_COUNT / 57,340 ($PREP_PCT)"
else
    echo "  Not started yet"
fi
echo ""

# Feature files (temporary in scratch)
echo "⚡ Feature Files (Temporary in Scratch):"
if [ -d "$SCRATCH_ROOT/avd1m_features_temp" ]; then
    FEAT_TEMP_SIZE=$(du -sh "$SCRATCH_ROOT/avd1m_features_temp" 2>/dev/null | cut -f1)
    FEAT_TEMP_COUNT=$(find "$SCRATCH_ROOT/avd1m_features_temp" -name "*.npz" 2>/dev/null | wc -l)
    echo "  Location: $SCRATCH_ROOT/avd1m_features_temp"
    echo "  Size: $FEAT_TEMP_SIZE"
    echo "  Files: $FEAT_TEMP_COUNT"
else
    echo "  Not started yet"
fi
echo ""

# Final NPZ files (permanent in projects)
echo "💾 NPZ Files (Permanent in Projects):"
if [ -d "$PROJECT_ROOT/data/avh_features" ]; then
    FEAT_FINAL_SIZE=$(du -sh "$PROJECT_ROOT/data/avh_features" 2>/dev/null | cut -f1)
    FEAT_FINAL_COUNT=$(find "$PROJECT_ROOT/data/avh_features" -name "*.npz" 2>/dev/null | wc -l)
    FEAT_PCT=$(awk "BEGIN {printf \"%.1f%%\", $FEAT_FINAL_COUNT/57340*100}")
    echo "  Location: $PROJECT_ROOT/data/avh_features"
    echo "  Size: $FEAT_FINAL_SIZE"
    echo "  Files: $FEAT_FINAL_COUNT / 57,340 ($FEAT_PCT)"
else
    echo "  Directory not found"
fi
echo ""

# FakeAVCeleb status
echo "📦 FakeAVCeleb Status (for reference):"
FAVC_PREP=$(find "$SCRATCH_ROOT/favc_preprocessed" -name "*.mp4" 2>/dev/null | wc -l)
FAVC_FEAT=$(find "$PROJECT_ROOT/data/favc_features" -name "*.npz" 2>/dev/null | wc -l)
echo "  Preprocessed: $FAVC_PREP / 21,544"
echo "  Features (PROJECTS): $FAVC_FEAT / 21,544"
echo ""

# Current jobs
echo "🔄 Current Jobs:"
squeue -u nnliang --format="  %.10i %.12P %.20j %.8T %.10M %R"
echo ""

echo "============================================================"
echo ""
echo "💡 Commands:"
echo "  AVDeepfake1M:"
echo "    Start preprocessing:  sbatch avd1m_scratch_preprocess.slurm"
echo "    Start features:       sbatch avd1m_scratch_features.slurm"
echo ""
echo "  FakeAVCeleb:"
echo "    Check status:         ./check_favc_scratch.sh"
echo ""
echo "  Monitor:"
echo "    Check all:            ./check_avd1m_scratch.sh"
echo "    View logs:            tail -f logs/avd1m_scratch_*_<jobid>.out"
echo ""
