#!/usr/bin/env python3
"""
Test the full preprocessing and feature extraction pipeline locally.
Downloads just 1 part (~1GB, ~2700 videos), processes them, and verifies output.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Paths
WORK_DIR = "/data/projects/punim2637/nnliang/AVH-Align"
CAUSAL_DIR = "/data/projects/punim2637/nnliang/CausalAVD"
TEST_DIR = tempfile.mkdtemp(prefix="avd1m_test_")
DOWNLOAD_DIR = os.path.join(TEST_DIR, "downloads")
EXTRACTED_DIR = os.path.join(TEST_DIR, "extracted")
PREPROCESSED_DIR = os.path.join(TEST_DIR, "preprocessed")
FEATURES_DIR = os.path.join(TEST_DIR, "features")
METADATA_FILE = os.path.join(TEST_DIR, "test_metadata.csv")

# Test with just part 1 (first file)
TEST_PART = "part-00001-99baffd6-3cb2-480c-817b-c02e8e75b4f1-c000"

print("="*70)
print("LOCAL PIPELINE TEST")
print("="*70)
print(f"Test directory: {TEST_DIR}")
print(f"Testing with: {TEST_PART}")
print()

# Create directories
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

def run_cmd(cmd, description):
    """Run command and print output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"✗ FAILED with exit code {result.returncode}")
        return False
    print(f"✓ SUCCESS")
    return True

try:
    # Step 1: Download one part
    print("\n" + "="*70)
    print("STEP 1: Download Test Part")
    print("="*70)
    
    download_cmd = [
        'python', f'{CAUSAL_DIR}/download_avd1m_huggingface.py',
        '--output_dir', DOWNLOAD_DIR,
        '--repo_id', 'ControlNet/AV-Deepfake1M',
        '--token', os.environ.get('HF_TOKEN', ''),  # Set HF_TOKEN env var before running
        '--parts', TEST_PART
    ]
    
    if not run_cmd(download_cmd, "Downloading test part"):
        sys.exit(1)
    
    # Check downloaded files
    archives = list(Path(DOWNLOAD_DIR).glob("*.7z.*"))
    print(f"\nDownloaded {len(archives)} files")
    for arch in archives[:3]:
        print(f"  {arch.name}")
    
    # Step 2: Extract
    print("\n" + "="*70)
    print("STEP 2: Extract Videos")
    print("="*70)
    
    # Find all parts of the archive
    base_archive = f"{DOWNLOAD_DIR}/{TEST_PART}.7z"
    
    extract_cmd = [
        '7z', 'x', f'{base_archive}.001',
        f'-o{EXTRACTED_DIR}',
        '-y'
    ]
    
    if not run_cmd(extract_cmd, "Extracting videos"):
        sys.exit(1)
    
    # Count extracted videos
    videos = list(Path(EXTRACTED_DIR).rglob("*.mp4"))
    print(f"\n✓ Extracted {len(videos)} videos")
    if videos:
        print(f"Sample videos:")
        for v in videos[:5]:
            print(f"  {v.relative_to(EXTRACTED_DIR)}")
    
    # Step 3: Create metadata
    print("\n" + "="*70)
    print("STEP 3: Create Metadata")
    print("="*70)
    
    with open(METADATA_FILE, 'w') as f:
        f.write("path,label,num_frames\n")
        for video in videos:
            # Get relative path from extracted/train/
            rel_path = video.relative_to(EXTRACTED_DIR / "train")
            f.write(f"{rel_path},0,100\n")
    
    print(f"✓ Created metadata: {len(videos)} entries")
    print(f"Metadata file: {METADATA_FILE}")
    
    # Step 4: Preprocess (sample - just first 10 videos for speed)
    print("\n" + "="*70)
    print("STEP 4: Preprocess Videos (first 10 for testing)")
    print("="*70)
    
    # Create small metadata for testing
    test_metadata = os.path.join(TEST_DIR, "small_test_metadata.csv")
    with open(METADATA_FILE, 'r') as fin:
        lines = fin.readlines()
        with open(test_metadata, 'w') as fout:
            fout.write(lines[0])  # header
            fout.writelines(lines[1:11])  # first 10 videos
    
    preprocess_cmd = [
        'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py',
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', test_metadata,
        '--data_path', f'{EXTRACTED_DIR}/train',
        '--save_path', PREPROCESSED_DIR,
        '--max_workers', '4'
    ]
    
    if not run_cmd(preprocess_cmd, "Preprocessing videos"):
        sys.exit(1)
    
    # Count preprocessed files
    roi_videos = list(Path(PREPROCESSED_DIR).rglob("*_roi.mp4"))
    wav_files = list(Path(PREPROCESSED_DIR).rglob("*.wav"))
    
    print(f"\n✓ Preprocessing created:")
    print(f"  {len(roi_videos)} ROI videos (*_roi.mp4)")
    print(f"  {len(wav_files)} audio files (*.wav)")
    
    if len(roi_videos) == 0:
        print("\n✗ ERROR: No ROI videos created!")
        print("Checking preprocessed directory structure:")
        result = subprocess.run(['ls', '-lah', PREPROCESSED_DIR], capture_output=True, text=True)
        print(result.stdout)
        sys.exit(1)
    
    if roi_videos:
        print(f"\nSample outputs:")
        for roi in roi_videos[:3]:
            size = roi.stat().st_size / 1024  # KB
            print(f"  {roi.name} ({size:.1f} KB)")
    
    # Step 5: Extract features
    print("\n" + "="*70)
    print("STEP 5: Extract Features")
    print("="*70)
    
    feature_cmd = [
        'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_feature_extraction.py',
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', test_metadata,
        '--ckpt_path', f'{WORK_DIR}/content/self_large_vox_433h.pt',
        '--data_path', PREPROCESSED_DIR,
        '--save_path', FEATURES_DIR
    ]
    
    # Change to avhubert directory for feature extraction
    os.chdir(f'{WORK_DIR}/av_hubert/avhubert')
    
    if not run_cmd(feature_cmd, "Extracting features"):
        sys.exit(1)
    
    # Count features
    features = list(Path(FEATURES_DIR).rglob("*.npz"))
    
    print(f"\n✓ Feature extraction created:")
    print(f"  {len(features)} feature files (*.npz)")
    
    if len(features) == 0:
        print("\n✗ ERROR: No features created!")
        print("Checking features directory:")
        result = subprocess.run(['ls', '-lahR', FEATURES_DIR], capture_output=True, text=True)
        print(result.stdout)
        sys.exit(1)
    
    if features:
        print(f"\nSample features:")
        for feat in features[:3]:
            size = feat.stat().st_size / 1024  # KB
            print(f"  {feat.name} ({size:.1f} KB)")
    
    # Success!
    print("\n" + "="*70)
    print("✓ SUCCESS! FULL PIPELINE WORKS!")
    print("="*70)
    print(f"\nProcessed 10 test videos:")
    print(f"  Downloaded: {len(archives)} archive parts")
    print(f"  Extracted: {len(videos)} videos")
    print(f"  Preprocessed: {len(roi_videos)} ROI videos, {len(wav_files)} audio files")
    print(f"  Features: {len(features)} .npz files")
    print(f"\nTest directory preserved at: {TEST_DIR}")
    print("You can inspect the files manually.")

except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
