#!/usr/bin/env python3
"""
Simplified test: Create a dummy video, preprocess it, extract features.
This tests if the preprocessing and feature extraction work correctly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path

# Paths
WORK_DIR = "/data/projects/punim2637/nnliang/AVH-Align"
TEST_DIR = tempfile.mkdtemp(prefix="preprocess_test_")
RAW_DIR = os.path.join(TEST_DIR, "raw/train")
PREPROCESSED_DIR = os.path.join(TEST_DIR, "preprocessed")
FEATURES_DIR = os.path.join(TEST_DIR, "features")
METADATA_FILE = os.path.join(TEST_DIR, "test_metadata.csv")

print("="*70)
print("PREPROCESSING & FEATURE EXTRACTION TEST")
print("="*70)
print(f"Test directory: {TEST_DIR}")
print()

# Create directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

def create_test_video(output_path, duration_sec=3, fps=25):
    """Create a simple test video with a face-like pattern."""
    print(f"Creating test video: {output_path}")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    for i in range(num_frames):
        # Create frame with gradient + circle (simulating a face)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [100 + i % 100, 150, 200]  # Blue-ish background
        
        # Draw a circle (simulating face)
        center = (width // 2, height // 2)
        radius = 100
        cv2.circle(frame, center, radius, (255, 200, 150), -1)
        
        # Draw eyes
        cv2.circle(frame, (center[0] - 30, center[1] - 20), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + 30, center[1] - 20), 10, (50, 50, 50), -1)
        
        # Draw mouth region
        cv2.ellipse(frame, (center[0], center[1] + 30), (40, 20), 0, 0, 180, (200, 100, 100), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Created video: {num_frames} frames")
    return output_path

try:
    # Step 1: Create test videos
    print("\n" + "="*70)
    print("STEP 1: Create Test Videos")
    print("="*70)
    
    test_videos = []
    for i in range(3):
        video_path = os.path.join(RAW_DIR, f"test_video_{i:03d}.mp4")
        create_test_video(video_path)
        test_videos.append(video_path)
    
    print(f"\n✓ Created {len(test_videos)} test videos")
    
    # Step 2: Create metadata
    print("\n" + "="*70)
    print("STEP 2: Create Metadata")
    print("="*70)
    
    with open(METADATA_FILE, 'w') as f:
        f.write("path,label,num_frames\n")
        for video in test_videos:
            filename = os.path.basename(video)
            f.write(f"{filename},0,75\n")
    
    print(f"✓ Created metadata: {len(test_videos)} entries")
    
    # Step 3: Preprocess
    print("\n" + "="*70)
    print("STEP 3: Preprocess Videos")
    print("="*70)
    
    preprocess_cmd = [
        'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py',
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', METADATA_FILE,
        '--data_path', f'{TEST_DIR}/raw',  # Script adds /train automatically
        '--save_path', PREPROCESSED_DIR,
        '--max_workers', '2'
    ]
    
    print(f"Command: {' '.join(preprocess_cmd)}")
    result = subprocess.run(preprocess_cmd, cwd=f'{WORK_DIR}/av_hubert/avhubert')
    
    if result.returncode != 0:
        print(f"✗ Preprocessing failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Check preprocessed files
    roi_videos = list(Path(PREPROCESSED_DIR).rglob("*_roi.mp4"))
    wav_files = list(Path(PREPROCESSED_DIR).rglob("*.wav"))
    
    print(f"\n✓ Preprocessing completed:")
    print(f"  {len(roi_videos)} ROI videos (*_roi.mp4)")
    print(f"  {len(wav_files)} audio files (*.wav)")
    
    if len(roi_videos) == 0:
        print("\n✗ ERROR: No ROI videos created!")
        print("Checking preprocessed directory:")
        for item in Path(PREPROCESSED_DIR).rglob("*"):
            print(f"  {item}")
        sys.exit(1)
    
    # Show created files
    print("\nCreated files:")
    for roi in roi_videos:
        size = roi.stat().st_size / 1024
        print(f"  {roi.name} ({size:.1f} KB)")
    for wav in wav_files:
        size = wav.stat().st_size / 1024
        print(f"  {wav.name} ({size:.1f} KB)")
    
    # Step 4: Extract features
    print("\n" + "="*70)
    print("STEP 4: Extract Features")
    print("="*70)
    
    feature_cmd = [
        'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_feature_extraction.py',
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', METADATA_FILE,
        '--ckpt_path', f'{WORK_DIR}/content/self_large_vox_433h.pt',
        '--data_path', PREPROCESSED_DIR,
        '--save_path', FEATURES_DIR
    ]
    
    print(f"Command: {' '.join(feature_cmd)}")
    os.chdir(f'{WORK_DIR}/av_hubert/avhubert')
    result = subprocess.run(feature_cmd)
    
    if result.returncode != 0:
        print(f"✗ Feature extraction failed with exit code {result.returncode}")
        sys.exit(1)
    
    # Check features
    features = list(Path(FEATURES_DIR).rglob("*.npz"))
    
    print(f"\n✓ Feature extraction completed:")
    print(f"  {len(features)} feature files (*.npz)")
    
    if len(features) == 0:
        print("\n✗ ERROR: No features created!")
        print("Checking features directory:")
        for item in Path(FEATURES_DIR).rglob("*"):
            print(f"  {item}")
        sys.exit(1)
    
    # Show created features
    print("\nCreated features:")
    for feat in features:
        size = feat.stat().st_size / 1024
        print(f"  {feat.name} ({size:.1f} KB)")
    
    # Success!
    print("\n" + "="*70)
    print("✓ SUCCESS! PREPROCESSING & FEATURE EXTRACTION WORK!")
    print("="*70)
    print(f"\nProcessed {len(test_videos)} test videos:")
    print(f"  Preprocessed: {len(roi_videos)} ROI videos, {len(wav_files)} audio files")
    print(f"  Features: {len(features)} .npz files")
    print(f"\nTest directory: {TEST_DIR}")
    
    # Cleanup
    print("\nCleaning up test directory...")
    shutil.rmtree(TEST_DIR)
    print("✓ Cleanup complete")

except KeyboardInterrupt:
    print("\n\n✗ Interrupted by user")
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    sys.exit(1)
except Exception as e:
    print(f"\n\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    if os.path.exists(TEST_DIR):
        print(f"\nTest directory preserved at: {TEST_DIR}")
    sys.exit(1)
