#!/usr/bin/env python3
"""
Feature extraction only - Step 3: Extract AV-HuBERT features
Requires preprocessed videos from preprocess_only.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
BASE_DIR = "/data/projects/punim2637/nnliang"
WORK_DIR = f"{BASE_DIR}/AVH-Align"
PREPROCESSED_DIR = f"{BASE_DIR}/AVH-Align/data/avd1m_preprocessed_batch/train"
FEATURES_DIR = f"{WORK_DIR}/data/avh_features/train"
METADATA_FILE = f"{WORK_DIR}/extracted_videos_metadata.csv"

print("="*70)
print("FEATURE EXTRACTION ONLY - PART 2 OF 2")
print("="*70)
print(f"Input (preprocessed): {os.path.dirname(PREPROCESSED_DIR)}")
print(f"Output: {FEATURES_DIR}")
print()

# Verify metadata exists
if not os.path.exists(METADATA_FILE):
    print(f"✗ ERROR: Metadata file not found: {METADATA_FILE}")
    print("  Run preprocess_only.py first!")
    sys.exit(1)

# Verify preprocessed videos exist
if not os.path.exists(PREPROCESSED_DIR):
    print(f"✗ ERROR: Preprocessed directory not found: {PREPROCESSED_DIR}")
    print("  Run preprocess_only.py first!")
    sys.exit(1)

# Count preprocessed files
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

print(f"Preprocessed files found:")
print(f"  ROI videos: {roi_count}")
print(f"  Audio files: {wav_count}")

if roi_count == 0:
    print(f"\n✗ ERROR: No preprocessed ROI videos found in {PREPROCESSED_DIR}")
    print("  Run preprocess_only.py first!")
    sys.exit(1)

print()

# Create output directory
os.makedirs(FEATURES_DIR, exist_ok=True)

# Extract features
print("="*70)
print("Extracting AV-HuBERT features")
print("="*70)
print(f"Model checkpoint: {WORK_DIR}/av_hubert/avhubert/self_large_vox_433h.pt")
print(f"Output directory: {FEATURES_DIR}")
print(f"Videos to process: {roi_count}")
print()

cmd = [
    'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_feature_extraction.py',
    '--dataset', 'AV1M',
    '--split', 'train',
    '--metadata', METADATA_FILE,
    '--ckpt_path', 'self_large_vox_433h.pt',  # Relative path from av_hubert/avhubert/
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
print("✓ FEATURE EXTRACTION COMPLETED - PART 2 OF 2")
print("="*70)
print(f"Input (preprocessed): {roi_count} ROI videos, {wav_count} audio files") 
print(f"Features extracted: {feature_count} .npz files")
print(f"Output location: {FEATURES_DIR}")
print()
print("All processing complete! Ready for training.")
print("="*70)
