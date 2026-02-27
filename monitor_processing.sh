#!/bin/bash
# Monitor preprocessing and feature extraction progress

echo "============================================================"
echo "  Processing Status Monitor"
echo "  $(date)"
echo "============================================================"
echo ""

# Check job queue
echo "=== Current Jobs ==="
squeue -u nnliang --format="%.18i %.10P %.20j %.8T %.10M %.6D %.20R" || echo "No jobs running"
echo ""

# Scratch preprocessing status
echo "=== Preprocessing Progress (Scratch) ==="
AVD1M_PREPROC=$(find /data/scratch/projects/punim2637/nnliang/avd1m_preprocessed -name "*_roi.mp4" 2>/dev/null | wc -l)
FAVC_PREPROC=$(find /data/scratch/projects/punim2637/nnliang/favc_preprocessed -name "*.mp4" 2>/dev/null | wc -l)
echo "  AVDeepfake1M: $AVD1M_PREPROC / 57340 ($(awk "BEGIN {printf \"%.1f%%\", $AVD1M_PREPROC/57340*100}"))"
echo "  FakeAVCeleb:  $FAVC_PREPROC / 21544 ($(awk "BEGIN {printf \"%.1f%%\", $FAVC_PREPROC/21544*100}"))"
echo ""

# Feature extraction status (in projects)
echo "=== Features Extracted (Projects) ==="
AVD1M_FEAT=$(find /data/projects/punim2637/nnliang/AVH-Align/data/avh_features -name "*.npz" 2>/dev/null | wc -l)
FAVC_FEAT=$(find /data/projects/punim2637/nnliang/AVH-Align/data/favc_features -name "*.npz" 2>/dev/null | wc -l)
echo "  AVDeepfake1M: $AVD1M_FEAT / 57340 ($(awk "BEGIN {printf \"%.1f%%\", $AVD1M_FEAT/57340*100}"))"
echo "  FakeAVCeleb:  $FAVC_FEAT / 21544 ($(awk "BEGIN {printf \"%.1f%%\", $FAVC_FEAT/21544*100}"))"
echo ""

# Storage status
echo "=== Storage Usage ==="
check_scratch_usage 2>/dev/null || echo "Scratch quota tool not available"
echo ""

# Latest logs
echo "=== Latest Log Files ==="
cd /data/projects/punim2637/nnliang/AVH-Align/logs
ls -lht *scratch* 2>/dev/null | head -4 || echo "No scratch logs found"
echo ""

echo "============================================================"
echo "To view live logs:"
echo "  tail -f logs/avd1m_scratch_preprocess_*.out"  
echo "  tail -f logs/favc_scratch_preprocess_*.out"
echo ""
echo "To submit feature extraction (after preprocessing completes):"
echo "  sbatch avd1m_scratch_features.slurm"
echo "  sbatch favc_scratch_features.slurm"
echo "============================================================"
