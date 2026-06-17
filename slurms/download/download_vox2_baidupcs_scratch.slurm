#!/bin/bash
#SBATCH --job-name=vox2_baidu
#SBATCH --partition=sapphire
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_vox2_baidupcs_%j.out
#SBATCH --error=logs/download_vox2_baidupcs_%j.err

set -euo pipefail

REPO=/data/projects/punim2637/nnliang/AVH-Align
DEST=/data/scratch/projects/punim2637/VoxCeleb2/raw
REMOTE=/apps/bypy/Vox2-mp4

mkdir -p "$DEST" "$REPO/logs"
cd "$REPO"

echo "Start: $(date)"
echo "Remote: $REMOTE"
echo "Destination: $DEST"
echo "Before size: $(du -sh "$DEST" 2>/dev/null || true)"

./BaiduPCS-Go config set -savedir "$DEST" -max_parallel 16 -max_download_load 4
./BaiduPCS-Go download "$REMOTE" --saveto "$DEST" -p 16 -l 4 --retry 10 --nocheck --mtime

echo "Done: $(date)"
echo "After size: $(du -sh "$DEST" 2>/dev/null || true)"
