#!/usr/bin/env python3
"""
Process already extracted videos - no downloading needed.
Uses existing videos at /Datasets/AVDeepfake1M/train_batch_extracted_job1/train/
"""

import os
import sys
import subprocess
import csv
from pathlib import Path

# Paths
BASE_DIR = "/data/projects/punim2637/nnliang"
WORK_DIR = f"{BASE_DIR}/AVH-Align"
EXTRACTED_DIR = f"{BASE_DIR}/Datasets/AVDeepfake1M/train_batch_extracted_job1"
PREPROCESSED_DIR = f"{BASE_DIR}/AVH-Align/data/avd1m_preprocessed_batch/train"
FEATURES_DIR = f"{WORK_DIR}/data/avh_features/train"
METADATA_FILE = f"{WORK_DIR}/extracted_videos_metadata.csv"

print("="*70)
print("PROCESS EXISTING EXTRACTED VIDEOS")
print("="*70)
print(f"Input: {EXTRACTED_DIR}/train")
print(f"Preprocessed output: {PREPROCESSED_DIR}")
print(f"Features output: {FEATURES_DIR}")
print()

# Create output directories
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Step 1: Count and create metadata
print("="*70)
print("Step 1: Creating metadata from existing videos")
print("="*70)
print(f"Searching in: {EXTRACTED_DIR}/train")
print()

videos = subprocess.check_output(
    f"find {EXTRACTED_DIR}/train -name '*.mp4' -type f",
    shell=True,
    text=True
).strip().split('\n')

videos = [v for v in videos if v]
print(f"✓ Found {len(videos)} videos")

# Show sample video paths
print(f"\nSample video paths (first 5):")
for v in videos[:5]:
    print(f"  {v}")

# Create metadata
fake_count = 0
real_count = 0
with open(METADATA_FILE, 'w') as f:
    f.write("path,label,num_frames\n")
    for video in videos:
        # Get path relative to EXTRACTED_DIR/train
        rel_path = os.path.relpath(video, f"{EXTRACTED_DIR}/train")
        # Simple label: fake if 'fake' in filename, else real
        label = 1 if 'fake' in os.path.basename(video) else 0
        if label == 1:
            fake_count += 1
        else:
            real_count += 1
        f.write(f"{rel_path},{label},100\n")

print(f"\n✓ Created metadata: {METADATA_FILE}")
print(f"  Total videos: {len(videos)}")
print(f"  Real videos: {real_count}")
print(f"  Fake videos: {fake_count}")
print(f"\nMetadata sample (first 5 entries):")
subprocess.run(f"head -6 {METADATA_FILE}", shell=True)
print()

# Step 2: Preprocess
print("="*70)
print("Step 2: Preprocessing videos")
print("="*70)
print(f"Input directory: {EXTRACTED_DIR}")
print(f"Preprocessing script will look for: {EXTRACTED_DIR}/train/")
print(f"Output directory: {os.path.dirname(PREPROCESSED_DIR)}")
print(f"Videos to process: {len(videos)}")
print(f"Workers: 32")
print(f"\nThis will take a while (probably several hours)...")
print()

cmd = [
    'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py',
    '--dataset', 'AV1M',
    '--split', 'train',
    '--metadata', METADATA_FILE,
    '--data_path', EXTRACTED_DIR,
    '--save_path', os.path.dirname(PREPROCESSED_DIR),  # Save to parent of train/
    '--max_workers', '32'
]

print(f"Command: {' '.join(cmd)}")
print()
print("Starting preprocessing... (watch for progress bar)")
print("-"*70)

result = subprocess.run(cmd, cwd=f'{WORK_DIR}/av_hubert/avhubert')

print("-"*70)
print(f"Preprocessing return code: {result.returncode}")

if result.returncode != 0:
    print("✗ Preprocessing failed with non-zero exit code")
    sys.exit(1)

# Check output
print(f"\nChecking preprocessed output directory: {PREPROCESSED_DIR}")
roi_count = int(subprocess.check_output(
    f"find {PREPROCESSED_DIR} -name '*_roi.mp4' | wc -l",
    shell=True,
    text=True
).strip())
wav_count = int(subprocess.check_output(
    f"find {PREPROCESSED_DIR} -name '*.wav' | wc -l",
    shell=True,
    text=True
).strip())

print(f"\n✓ Preprocessing completed")
print(f"  ROI videos created: {roi_count}")
print(f"  Audio files created: {wav_count}")

if roi_count > 0:
    print(f"\nSample preprocessed files:")
    subprocess.run(f"find {PREPROCESSED_DIR} -name '*_roi.mp4' | head -3", shell=True)
    print()

if roi_count == 0:
    print("\n✗ ERROR: No preprocessed videos created!")
    print("Checking directory structure:")
    subprocess.run(f"ls -lhR {PREPROCESSED_DIR} | head -30", shell=True)
    sys.exit(1)

# Step 3: Extract features
print("="*70)
print("Step 3: Extracting AV-HuBERT features")
print("="*70)
print(f"Input (preprocessed): {os.path.dirname(PREPROCESSED_DIR)}")
print(f"Model checkpoint: {WORK_DIR}/content/self_large_vox_433h.pt")
print(f"Output directory: {FEATURES_DIR}")
print(f"Videos to process: {roi_count}")
print()

cmd = [
    'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_feature_extraction.py',
    '--dataset', 'AV1M',
    '--split', 'train',
    '--metadata', METADATA_FILE,
    '--ckpt_path', f'{WORK_DIR}/content/self_large_vox_433h.pt',
    '--data_path', os.path.dirname(PREPROCESSED_DIR),
    '--save_path', FEATURES_DIR
]

print(f"Command: {' '.join(cmd)}")
print()
print("Starting feature extraction... (this uses GPU)")
print("-"*70)

os.chdir(f'{WORK_DIR}/av_hubert/avhubert')
result = subprocess.run(cmd)

print("-"*70)
print(f"Feature extraction return code: {result.returncode}")

if result.returncode != 0:
    print("✗ Feature extraction failed with non-zero exit code")
    sys.exit(1)

# Check features
print(f"\nChecking feature output directory: {FEATURES_DIR}")
feature_count = int(subprocess.check_output(
    f"find {FEATURES_DIR} -name '*.npz' | wc -l",
    shell=True,
    text=True
).strip())

print(f"\n✓ Feature extraction completed")
print(f"  Features created: {feature_count} .npz files")

if feature_count > 0:
    print(f"\nSample feature files:")
    subprocess.run(f"find {FEATURES_DIR} -name '*.npz' | head -5", shell=True)
    
    # Show file sizes
    print(f"\nSample feature file sizes:")
    subprocess.run(f"find {FEATURES_DIR} -name '*.npz' | head -3 | xargs ls -lh", shell=True)
    print()

if feature_count == 0:
    print("\n✗ ERROR: No features created!")
    print("Checking features directory:")
    subprocess.run(f"ls -lhR {FEATURES_DIR} | head -30", shell=True)
    sys.exit(1)

print("\n" + "="*70)
print("✓ SUCCESS! FULL PIPELINE COMPLETED")
print("="*70)
print(f"Input videos: {len(videos)}")
print(f"  Real: {real_count}")
print(f"  Fake: {fake_count}")
print(f"Preprocessed videos: {roi_count} ROI videos, {wav_count} audio files")
print(f"Features extracted: {feature_count} .npz files")
print(f"\nOutput locations:")
print(f"  Preprocessed: {PREPROCESSED_DIR}")
print(f"  Features: {FEATURES_DIR}")
print(f"  Metadata: {METADATA_FILE}")
print("="*70)
