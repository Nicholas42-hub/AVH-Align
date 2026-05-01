#!/bin/bash
#SBATCH --job-name=download_testB
#SBATCH --output=/data/projects/punim2637/nnliang/logs/download_testB_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -e

DEST=/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus
mkdir -p "$DEST"

echo "Starting testB download at $(date)"

python3 -c "
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
import os
import shutil

repo_id = 'ControlNet/AV-Deepfake1M-PlusPlus'
dest = '$DEST/testB'
os.makedirs(dest, exist_ok=True)

token = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
api = HfApi(token=token)
files = [f for f in api.list_repo_files(repo_id, repo_type='dataset') if f.startswith('testB/')]
print(f'Downloading {len(files)} parts...')

for i, fname in enumerate(sorted(files)):
    out = os.path.join(dest, os.path.basename(fname))
    if os.path.exists(out):
        print(f'[{i+1}/{len(files)}] Already exists: {out}')
        continue
    print(f'[{i+1}/{len(files)}] Downloading {fname}...', flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type='dataset', local_dir=dest, token=token)
    # If hf_hub_download created a symlink or put it in a subdir, move it
    real_path = os.path.realpath(path)
    if real_path != out:
        shutil.copy2(real_path, out)
    print(f'  -> {out}', flush=True)

print('All parts downloaded.')
"

echo "Download complete at $(date). Starting unzip..."

cd "$DEST/testB"
# Move all parts to flat directory (hf_hub_download may put them in testB/testB/)
if [ -d testB ]; then
    mv testB/*.zip.* . 2>/dev/null || true
    rmdir testB 2>/dev/null || true
fi

7z x testB.zip.001 -o"$DEST/testB/videos"

echo "Unzip complete at $(date)"
echo "Contents:"
ls -lh "$DEST/testB/"
