#!/bin/bash
# Overall Scratch Space Management for All Datasets
# ============================================================

SCRATCH_ROOT=/data/scratch/projects/punim2637/nnliang
PROJECT_ROOT=/data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  OVERALL SCRATCH SPACE STATUS"
echo "  $(date)"
echo "============================================================"
echo ""

# Overall scratch usage
echo "📊 Total Scratch Usage:"
check_scratch_usage
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. AVDeepfake1M (57,340 videos)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# AVD1M Dataset
AVD_SIZE=$(du -sh "$SCRATCH_ROOT/Datasets/AVDeepfake1M" 2>/dev/null | cut -f1 || echo "N/A")
AVD_COUNT=$(find "$SCRATCH_ROOT/Datasets/AVDeepfake1M/val" -name "*.mp4" 2>/dev/null | wc -l)
echo "📁 Dataset (SCRATCH): $AVD_SIZE ($AVD_COUNT videos)"

# AVD1M Preprocessed
if [ -d "$SCRATCH_ROOT/avd1m_preprocessed" ]; then
    AVD_PREP=$(find "$SCRATCH_ROOT/avd1m_preprocessed" -name "*_roi.mp4" 2>/dev/null | wc -l)
    AVD_PREP_PCT=$(awk "BEGIN {printf \"%.1f%%\", $AVD_PREP/57340*100}")
    echo "🔄 Preprocessed (SCRATCH): $AVD_PREP / 57,340 ($AVD_PREP_PCT)"
else
    echo "🔄 Preprocessed (SCRATCH): Not started"
fi

# AVD1M Features
AVD_FEAT=$(find "$PROJECT_ROOT/data/avh_features" -name "*.npz" 2>/dev/null | wc -l)
AVD_FEAT_SIZE=$(du -sh "$PROJECT_ROOT/data/avh_features" 2>/dev/null | cut -f1 || echo "N/A")
AVD_FEAT_PCT=$(awk "BEGIN {printf \"%.1f%%\", $AVD_FEAT/57340*100}")
echo "💾 Features (PROJECTS): $AVD_FEAT / 57,340 ($AVD_FEAT_PCT) [$AVD_FEAT_SIZE]"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2. FakeAVCeleb (21,544 videos)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# FakeAVCeleb Dataset
FAVC_SIZE=$(du -sh "$SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2" 2>/dev/null | cut -f1 || echo "N/A")
FAVC_COUNT=$(find "$SCRATCH_ROOT/Datasets/FakeAVCeleb_v1.2" -name "*.mp4" 2>/dev/null | wc -l)
echo "📁 Dataset (SCRATCH): $FAVC_SIZE ($FAVC_COUNT videos)"

# FakeAVCeleb Preprocessed
if [ -d "$SCRATCH_ROOT/favc_preprocessed" ]; then
    FAVC_PREP=$(find "$SCRATCH_ROOT/favc_preprocessed" -name "*.mp4" 2>/dev/null | wc -l)
    FAVC_PREP_PCT=$(awk "BEGIN {printf \"%.1f%%\", $FAVC_PREP/21544*100}")
    echo "🔄 Preprocessed (SCRATCH): $FAVC_PREP / 21,544 ($FAVC_PREP_PCT)"
else
    echo "🔄 Preprocessed (SCRATCH): Not started"
fi

# FakeAVCeleb Features
FAVC_FEAT=$(find "$PROJECT_ROOT/data/favc_features" -name "*.npz" 2>/dev/null | wc -l)
FAVC_FEAT_SIZE=$(du -sh "$PROJECT_ROOT/data/favc_features" 2>/dev/null | cut -f1 || echo "N/A")
FAVC_FEAT_PCT=$(awk "BEGIN {printf \"%.1f%%\", $FAVC_FEAT/21544*100}")
echo "💾 Features (PROJECTS): $FAVC_FEAT / 21,544 ($FAVC_FEAT_PCT) [$FAVC_FEAT_SIZE]"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Current Jobs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
squeue -u nnliang --format="  %.10i %.12P %.20j %.8T %.10M %.6D %R"

echo ""
echo "============================================================"
echo ""
echo "💡 Quick Commands:"
echo ""
echo "  AVDeepfake1M:"
echo "    sbatch avd1m_scratch_preprocess.slurm   # Step 1"
echo "    sbatch avd1m_scratch_features.slurm     # Step 2"
echo ""
echo "  FakeAVCeleb:"
echo "    sbatch favc_scratch_preprocess.slurm    # Step 1"
echo "    sbatch favc_scratch_features.slurm      # Step 2"
echo ""
echo "  Monitor:"
echo "    ./check_all_scratch.sh                  # This script"
echo "    ./check_avd1m_scratch.sh                # AVD1M details"
echo "    ./check_favc_scratch.sh                 # FakeAVCeleb details"
echo ""
