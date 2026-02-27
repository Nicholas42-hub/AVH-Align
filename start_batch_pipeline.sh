#!/bin/bash
#
# Quick Start Guide: Batch Download & Feature Extraction Pipeline
#

cd /data/projects/punim2637/nnliang/AVH-Align

echo "==========================================================="
echo "  AVH-Align Batch Pipeline - Quick Start"
echo "==========================================================="
echo ""
echo "This pipeline will:"
echo "  1. Download training videos in batches (20 parts at a time)"
echo "  2. Extract videos from archives"
echo "  3. Preprocess videos (extract lips & audio)"
echo "  4. Extract AV-HuBERT features"
echo "  5. Delete source files to free space"
echo "  6. Repeat for all 254 archive parts"
echo ""
echo "Space required: ~200-300 GB (not 1TB!)"
echo "Time estimate: 5-7 days"
echo "Output: data/avh_features/train/*.npz"
echo ""
echo "==========================================================="
echo ""

# Check if already running
if squeue -u $USER | grep -q "batch_avd1m"; then
    echo "⚠ Batch pipeline is already running!"
    echo ""
    squeue -u $USER | grep "batch_avd1m"
    echo ""
    echo "To monitor: tail -f logs/batch_avd1m_*.out"
    exit 0
fi

# Check disk space
AVAIL=$(df /data/gpfs | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL / 1024 / 1024))

echo "Disk space available: ${AVAIL_GB} GB"

if [ $AVAIL_GB -lt 200 ]; then
    echo ""
    echo "⚠ WARNING: Less than 200 GB available!"
    echo "   Recommended: At least 250 GB for safe operation"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Submitting batch pipeline job..."
echo ""

JOB_ID=$(sbatch --parsable batch_pipeline.slurm)

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/batch_avd1m_${JOB_ID}.out"
    echo ""
    echo "Check progress:"
    echo "  cat batch_progress.json"
    echo "  find data/avh_features/train -name '*.npz' | wc -l"
    echo ""
    echo "The pipeline will automatically resume if interrupted."
    echo "Progress is saved after each batch completes."
else
    echo "✗ Job submission failed"
    exit 1
fi
