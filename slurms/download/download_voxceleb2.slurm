#!/bin/bash
#SBATCH --job-name=download_vox2
#SBATCH --partition=cascade,sapphire
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_vox2_%j.out
#SBATCH --error=logs/download_vox2_%j.err

set -euo pipefail

DEST=/data/scratch/projects/punim2637/voxceleb2_raw
mkdir -p "$DEST"

echo "Start: $(date)"
echo "Downloading VoxCeleb2 to $DEST"

module purge
module load Miniconda3/23.10.0-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

/data/projects/punim2637/nnliang/avh_env/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Reverb/voxceleb2',
    repo_type='dataset',
    local_dir='/data/scratch/projects/punim2637/voxceleb2_raw',
    max_workers=4,
)
print('Download complete.')
"

echo "Done: $(date)"
echo "Size: $(du -sh $DEST)"
