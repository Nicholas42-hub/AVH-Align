#!/usr/bin/env python3
"""Create proper test video using FFmpeg and test preprocessing"""
import os
import subprocess
import tempfile
from pathlib import Path

TEST_DIR = tempfile.mkdtemp(prefix="ffmpeg_test_")
WORK_DIR = "/data/projects/punim2637/nnliang/AVH-Align"
RAW_DIR = os.path.join(TEST_DIR, "raw/train")
PREPROCESSED_DIR = os.path.join(TEST_DIR, "preprocessed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

print(f"Test directory: {TEST_DIR}\n")

# Create test video with FFmpeg (generates a color video with moving pattern)
video_path = os.path.join(RAW_DIR, "test_video.mp4")
print("Creating test video with FFmpeg...")
cmd = [
    "ffmpeg",
    "-f", "lavfi",
    "-i", "testsrc=duration=3:size=640x480:rate=25",  # Test pattern
    "-f", "lavfi",
    "-i", "sine=frequency=1000:duration=3",  # Test audio
    "-pix_fmt", "yuv420p",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-y",
    video_path
]

result = subprocess.run(cmd, capture_output=True)
if result.returncode != 0:
    print(f"✗ FFmpeg failed: {result.stderr.decode()}")
    exit(1)

size = Path(video_path).stat().st_size / 1024
print(f"✓ Created video: {video_path} ({size:.1f} KB)\n")

# Create metadata
metadata = os.path.join(TEST_DIR, "metadata.csv")
with open(metadata, 'w') as f:
    f.write("path,label,num_frames\n")
    f.write("test_video.mp4,0,75\n")

print("Running preprocessing...")
cmd = [
    'python', f'{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py',
    '--dataset', 'AV1M',
    '--split', 'train',
    '--metadata', metadata,
    '--data_path', f'{TEST_DIR}/raw',
    '--save_path', PREPROCESSED_DIR,
    '--max_workers', '1'
]

print(f"Command: {' '.join(cmd)}\n")
result = subprocess.run(cmd, cwd=f'{WORK_DIR}/av_hubert/avhubert')

# Check results
roi_videos = list(Path(PREPROCESSED_DIR).rglob("*_roi.mp4"))
wav_files = list(Path(PREPROCESSED_DIR).rglob("*.wav"))

print(f"\nResults:")
print(f"  ROI videos: {len(roi_videos)}")
print(f"  Audio files: {len(wav_files)}")

if roi_videos:
    for roi in roi_videos:
        size = roi.stat().st_size / 1024
        print(f"  ✓ {roi.name} ({size:.1f} KB)")
    print(f"\n✓ SUCCESS! Preprocessing works!")
    print(f"Test dir: {TEST_DIR}")
else:
    print(f"\n✗ FAILED: No output files")
    print(f"Check: ls -lahR {PREPROCESSED_DIR}")
