#!/usr/bin/env python3
"""
Preprocessing only - Step 1 & 2: Create metadata and preprocess videos
No feature extraction (that's in extract_features_only.py)
"""

import os
import sys
import subprocess
import csv
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description='Preprocess AV-Deepfake1M videos')
parser.add_argument('--input-dir', type=str, help='Input directory with extracted videos (e.g., /path/to/train)')
parser.add_argument('--output-dir', type=str, help='Output directory for preprocessed videos')
parser.add_argument('--workers', type=int, default=32, help='Number of parallel workers')
args = parser.parse_args()

# Paths
BASE_DIR = "/data/projects/punim2637/nnliang"
WORK_DIR = f"{BASE_DIR}/AVH-Align"

# Use command line args if provided, otherwise use defaults
if args.input_dir:
    EXTRACTED_DIR = args.input_dir
else:
    EXTRACTED_DIR = f"{BASE_DIR}/Datasets/AVDeepfake1M/train_batch_extracted_job1/train"

if args.output_dir:
    PREPROCESSED_DIR = args.output_dir
else:
    PREPROCESSED_DIR = f"{BASE_DIR}/AVH-Align/data/avd1m_preprocessed_batch/train"

METADATA_FILE = f"{WORK_DIR}/batch_metadata_{os.getpid()}.csv"

print("="*70)
print("PREPROCESSING ONLY - PART 1 OF 2")
print("="*70)
print(f"Input: {EXTRACTED_DIR}")
print(f"Output: {PREPROCESSED_DIR}")
print(f"Workers: {args.workers}")
print()

# Create output directories
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# Step 1: Count and create metadata
print("="*70)
print("Step 1: Creating metadata from existing videos")
print("="*70)
print(f"Searching in: {EXTRACTED_DIR}")
print()

videos = subprocess.check_output(
    f"find {EXTRACTED_DIR} -name '*.mp4' -type f",
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
        # Get path relative to EXTRACTED_DIR
        rel_path = os.path.relpath(video, EXTRACTED_DIR)
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
print(f"Input directory: {os.path.dirname(EXTRACTED_DIR)}")
print(f"Output directory: {os.path.dirname(PREPROCESSED_DIR)}")
print(f"Videos to process: {len(videos)}")
print(f"Workers: {args.workers}")
print(f"\nThis will take several hours...")
print()

cmd = [
    'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py',
    '--dataset', 'AV1M',
    '--split', 'train',
    '--metadata', METADATA_FILE,
    '--data_path', os.path.dirname(EXTRACTED_DIR),
    '--save_path', os.path.dirname(PREPROCESSED_DIR),
    '--max_workers', str(args.workers)
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
    
    # Show directory structure
    print(f"\nDirectory structure sample:")
    subprocess.run(f"find {PREPROCESSED_DIR} -type d | head -10", shell=True)
    print()

if roi_count == 0:
    print("\n✗ ERROR: No preprocessed videos created!")
    print("Checking directory structure:")
    subprocess.run(f"ls -lhR {PREPROCESSED_DIR} | head -30", shell=True)
    sys.exit(1)

print("\n" + "="*70)
print("✓ PREPROCESSING COMPLETED - PART 1 OF 2")
print("="*70)
print(f"Input videos: {len(videos)} ({real_count} real, {fake_count} fake)")
print(f"Preprocessed output: {roi_count} ROI videos, {wav_count} audio files")
print(f"Output location: {PREPROCESSED_DIR}")
print(f"Metadata: {METADATA_FILE}")
print()
print("Next step: Run extract_features_only.py to extract AV-HuBERT features")
print("="*70)
