#!/bin/bash
# Quick progress check for FakeAVCeleb full dataset processing

echo "============================================================"
echo "  FakeAVCeleb Full Dataset Processing Progress"
echo "  $(date)"
echo "============================================================"
echo ""

cd /data/projects/punim2637/nnliang/AVH-Align

TOTAL=21544
PREP=$(find data/favc_preprocessed -name "*.mp4" 2>/dev/null | wc -l)
FEAT=$(find data/favc_features -name "*.npz" 2>/dev/null | wc -l)

PREP_PCT=$(awk "BEGIN {printf \"%.1f\", $PREP/$TOTAL*100}")
FEAT_PCT=$(awk "BEGIN {printf \"%.1f\", $FEAT/$TOTAL*100}")

echo "Total videos: $TOTAL"
echo ""
echo "Preprocessing progress:"
echo "  Completed: $PREP / $TOTAL ($PREP_PCT%)"
echo "  Remaining: $((TOTAL - PREP))"
echo ""
echo "Feature extraction progress:"
echo "  Completed: $FEAT / $TOTAL ($FEAT_PCT%)"
echo "  Remaining: $((TOTAL - FEAT))"
echo ""

# Check running jobs
echo "Current job status:"
squeue -u nnliang --format="  %.10i %.12P %.20j %.8T %.10M %.6D %R"

echo ""
echo "============================================================"
echo ""
echo "Log commands:"
echo "  tail -f logs/favc_full_preprocess_*.out"
echo "  tail -f logs/favc_full_features_*.out"
echo ""
