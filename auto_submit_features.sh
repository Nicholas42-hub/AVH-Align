#!/bin/bash
# Auto-submit feature extraction after preprocessing completes
# This script waits for preprocessing to reach a threshold, then submits feature extraction

DATASET=$1  # "avd1m" or "favc"
THRESHOLD=${2:-50000}  # Minimum preprocessed files before submitting (default: 50000)

if [ "$DATASET" == "avd1m" ]; then
    PREPROC_DIR="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
    TARGET=57340
    SLURM_SCRIPT="avd1m_scratch_features.slurm"
    NAME="AVDeepfake1M"
elif [ "$DATASET" == "favc" ]; then
    PREPROC_DIR="/data/scratch/projects/punim2637/nnliang/favc_preprocessed"
    TARGET=21544
    SLURM_SCRIPT="favc_scratch_features.slurm"
    NAME="FakeAVCeleb"
else
    echo "Usage: $0 <avd1m|favc> [threshold]"
    exit 1
fi

cd /data/projects/punim2637/nnliang/AVH-Align

echo "============================================================"
echo "  Auto Feature Extraction Submitter"
echo "  Dataset: $NAME"
echo "  Threshold: $THRESHOLD / $TARGET"
echo "  $(date)"
echo "============================================================"

while true; do
    if [ "$DATASET" == "avd1m" ]; then
        COUNT=$(timeout 10 find "$PREPROC_DIR" -name "*_roi.mp4" 2>/dev/null | wc -l)
    else
        COUNT=$(timeout 10 find "$PREPROC_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    fi
    
    PCT=$(awk "BEGIN {printf \"%.1f%%\", $COUNT/$TARGET*100}")
    echo "[$(date +%H:%M:%S)] Progress: $COUNT / $TARGET ($PCT)"
    
    if [ "$COUNT" -ge "$THRESHOLD" ]; then
        echo ""
        echo "Threshold reached! Submitting feature extraction..."
        sbatch "$SLURM_SCRIPT"
        echo ""
        echo "Feature extraction job submitted for $NAME"
        echo "Monitor with: tail -f logs/${DATASET}_scratch_features_*.out"
        exit 0
    fi
    
    sleep 300  # Check every 5 minutes
done
