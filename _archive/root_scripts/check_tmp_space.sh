#!/bin/bash
# Check /tmp space on compute node when job runs
echo "=== Node /tmp Space Check ==="
df -h /tmp
echo ""
echo "Available /tmp space: $(df -h /tmp | tail -1 | awk '{print $4}')"
echo ""
# Also check if there's a local scratch directory
if [ -d "/scratch" ]; then
    echo "=== /scratch available ==="
    df -h /scratch
elif [ -d "/local" ]; then
    echo "=== /local available ==="
    df -h /local
elif [ -d "$TMPDIR" ]; then
    echo "=== \$TMPDIR = $TMPDIR ==="
    df -h "$TMPDIR"
fi
