#!/bin/bash
#SBATCH --job-name=test_small
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/test_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/test_%j.err

export PATH=/data/projects/punim2637/nnliang/avh_env/bin:/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin:$PATH
export PYTHONNOUSERSITE=1
cd /data/projects/punim2637/nnliang/AVH-Align
python -c "
import os, subprocess, csv
from huggingface_hub import hf_hub_download

TOKEN = 'HF_TOKEN_REMOVED'
ROOT = '/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M'
os.makedirs(f'{ROOT}/test_arch', exist_ok=True)
os.makedirs(f'{ROOT}/test_ext', exist_ok=True)

print('Downloading 3 parts...')
for i in [1,2,3]:
    print(f'  Part {i}...')
    hf_hub_download('ControlNet/AV-Deepfake1M', f'train/train.zip.{i:03d}', local_dir=f'{ROOT}/test_arch', token=TOKEN, resume_download=True)

print('Extracting...')
r = subprocess.run(f\"7z x '{ROOT}/test_arch/train/train.zip.001' -o'{ROOT}/test_ext' -y\", shell=True, capture_output=True, text=True)
vids = subprocess.run(f\"find '{ROOT}/test_ext' -name '*.mp4'\", shell=True, capture_output=True, text=True).stdout.strip().split('\n')
print(f'Extracted {len(vids)} videos')

print('Creating metadata...')
with open('/data/projects/punim2637/nnliang/AVH-Align/test_meta.csv', 'w') as f:
    f.write('path,label,num_frames\n')
    for v in vids[:100]:  # Only process first 100 videos
        if v:
            rel = os.path.relpath(v, f'{ROOT}/test_ext')
            f.write(f'{rel},0,100\n')
print(f'Metadata: 100 videos')

print('\\nPreprocessing...')
subprocess.run(['python', 'av_hubert/avhubert/deepfake_preprocess.py', '--dataset', 'AV1M', '--split', 'train', '--metadata', '/data/projects/punim2637/nnliang/AVH-Align/test_meta.csv', '--data_path', f'{ROOT}/test_ext', '--save_path', '/data/projects/punim2637/nnliang/AVH-Align/data/avh_preprocessed/test', '--max_workers', '8'])

pc = subprocess.run(\"find /data/projects/punim2637/nnliang/AVH-Align/data/avh_preprocessed/test -name '*.npz' | wc -l\", shell=True, capture_output=True, text=True).stdout.strip()
print(f'\\nPreprocessed files: {pc}')

if int(pc) > 0:
    print('\\nFeature extraction...')
    subprocess.run(['python', 'av_hubert/avhubert/deepfake_feature_extraction.py', '--dataset', 'AV1M', '--split', 'train', '--metadata', '/data/projects/punim2637/nnliang/AVH-Align/test_meta.csv', '--ckpt_path', 'av_hubert/avhubert/self_large_vox_433h.pt', '--data_path', '/data/projects/punim2637/nnliang/AVH-Align/data/avh_preprocessed/test', '--save_path', '/data/projects/punim2637/nnliang/AVH-Align/data/avh_features/test'])
    
    fc = subprocess.run(\"find /data/projects/punim2637/nnliang/AVH-Align/data/avh_features/test -name '*.npz' | wc -l\", shell=True, capture_output=True, text=True).stdout.strip()
    print(f'\\nFeatures: {fc}')
"
