#!/bin/bash
#SBATCH --job-name=extract_testB
#SBATCH --output=/data/projects/punim2637/nnliang/logs/extract_testB_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-a100-mig
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -e

DEST=/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus/testB

echo "Starting extraction at $(date)"
cd "$DEST"

7za x testB.zip.001 -o"$DEST/videos"

echo "Extraction complete at $(date)"
echo "Contents:"
ls -lh "$DEST/videos/" | head -20
echo "Total files: $(find $DEST/videos -name '*.mp4' | wc -l)"
