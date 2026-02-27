#!/usr/bin/env python3
"""
Download, extract, preprocess, and cleanup pipeline for AV-Deepfake1M training data.
Downloads files in batches to manage disk space efficiently.
"""

import os
import subprocess
import time
import zipfile
from pathlib import Path
import shutil
import sys

# Configuration
DATASET_DIR = Path("/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M/train")
EXTRACTED_DIR = Path("/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M/train_batch_extracted")
PREPROCESSED_DIR = Path("/data/projects/punim2637/nnliang/AVH-Align/data/avd1m_preprocessed_batch/train")
REPO_ID = "ControlNet/AV-Deepfake1M"
SUBSET = "train"
HF_TOKEN = "HF_TOKEN_REMOVED"

# Batch configuration
START_FILE = 1  # Always start from 1 for multi-part zip extraction
END_FILE = 6    # Download through file 006 (6 files total)
BATCH_SIZE = 6

# Files that were already processed (from previous batch)
ALREADY_PROCESSED_ARCHIVES = [1, 2, 3]  # These will be re-downloaded temporarily for extraction

def download_file(file_number):
    """Download a single archive file using huggingface_hub"""
    filename = f"train.zip.{file_number:03d}"
    filepath = DATASET_DIR / filename
    
    if filepath.exists():
        print(f"✓ {filename} already exists, skipping download")
        return True
    
    print(f"⬇️  Downloading {filename}...")
    try:
        from huggingface_hub import login, hf_hub_download
        
        # Login with token
        login(token=HF_TOKEN, add_to_git_credential=False)
        
        # Download file
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{SUBSET}/{filename}",
            repo_type="dataset",
            local_dir=DATASET_DIR.parent,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
            resume_download=True
        )
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def extract_archives():
    """Extract all archive files to extracted directory"""
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📦 Extracting archives to {EXTRACTED_DIR}...")
    
    # Find all archive parts
    archive_parts = sorted(DATASET_DIR.glob("train.zip.*"))
    if not archive_parts:
        print("✗ No archive files found to extract")
        return False
    
    print(f"Found {len(archive_parts)} archive parts")
    
    # Combine and extract (multi-part zip)
    try:
        # Use 7z for multi-part archives
        cmd = f"cd {DATASET_DIR} && 7z x train.zip.001 -o{EXTRACTED_DIR} -y"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Extraction complete")
            return True
        else:
            print(f"✗ Extraction failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return False

def count_videos():
    """Count videos in extracted directory"""
    extracted_train = EXTRACTED_DIR / "train"
    if not extracted_train.exists():
        return 0
    
    video_count = sum(1 for _ in extracted_train.rglob("*.mp4"))
    return video_count

def preprocess_videos():
    """Run preprocessing on extracted videos"""
    print(f"\n🔧 Preprocessing videos...")
    
    extracted_train = EXTRACTED_DIR / "train"
    if not extracted_train.exists():
        print(f"✗ Extracted directory not found: {extracted_train}")
        return False
    
    video_count = count_videos()
    print(f"Found {video_count} videos to preprocess")
    
    # Run preprocessing script
    preprocess_script = Path("/data/projects/punim2637/nnliang/AVH-Align/preprocess_only.py")
    
    try:
        # Update the script to use current batch directory
        cmd = [
            "python3",
            str(preprocess_script),
            "--input-dir", str(extracted_train),
            "--output-dir", str(PREPROCESSED_DIR),
            "--workers", "32"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✓ Preprocessing complete")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"⚠️  Preprocessing completed with issues")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return True  # Continue even with some failures
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def cleanup_files():
    """Clean up archive and extracted files to free space"""
    print(f"\n🧹 Cleaning up intermediate files...")
    
    # Remove archive files
    archive_count = 0
    for archive in DATASET_DIR.glob("train.zip.*"):
        try:
            size_mb = archive.stat().st_size / (1024 * 1024)
            archive.unlink()
            archive_count += 1
            print(f"  ✓ Removed {archive.name} ({size_mb:.0f} MB)")
        except Exception as e:
            print(f"  ✗ Failed to remove {archive.name}: {e}")
    
    # Remove extracted directory
    if EXTRACTED_DIR.exists():
        try:
            size_gb = sum(f.stat().st_size for f in EXTRACTED_DIR.rglob('*') if f.is_file()) / (1024**3)
            shutil.rmtree(EXTRACTED_DIR)
            print(f"  ✓ Removed extracted directory ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"  ✗ Failed to remove extracted directory: {e}")
    
    print(f"✓ Cleanup complete (removed {archive_count} archives)")

def main():
    """Main pipeline: download -> extract -> preprocess -> cleanup"""
    print("=" * 60)
    print("AV-Deepfake1M Batch Download & Process Pipeline")
    print("=" * 60)
    print(f"Downloading files {START_FILE:03d} to {END_FILE:03d}")
    print(f"Note: Files {ALREADY_PROCESSED_ARCHIVES} will be re-downloaded")
    print(f"temporarily for multi-part zip extraction, then deleted")
    print()
    
    # Step 1: Download files
    print("STEP 1: Downloading files")
    print("-" * 60)
    downloaded = []
    for file_num in range(START_FILE, END_FILE + 1):
        if download_file(file_num):
            downloaded.append(file_num)
        else:
            print(f"⚠️  Skipping file {file_num:03d}")
    
    if not downloaded:
        print("✗ No files downloaded, exiting")
        return 1
    
    print(f"\n✓ Successfully downloaded {len(downloaded)} files")
    
    # Step 2: Extract
    print("\n" + "=" * 60)
    print("STEP 2: Extracting archives")
    print("-" * 60)
    if not extract_archives():
        print("✗ Extraction failed, exiting")
        return 1
    
    video_count = count_videos()
    print(f"✓ Extracted {video_count} videos")
    
    # Step 3: Preprocess
    print("\n" + "=" * 60)
    print("STEP 3: Preprocessing videos")
    print("-" * 60)
    if not preprocess_videos():
        print("✗ Preprocessing failed, exiting")
        return 1
    
    # Count successful preprocessing
    roi_count = sum(1 for _ in PREPROCESSED_DIR.rglob("*_roi.mp4"))
    print(f"✓ Created {roi_count} preprocessed videos")
    
    # Step 4: Cleanup
    print("\n" + "=" * 60)
    print("STEP 4: Cleaning up intermediate files")
    print("-" * 60)
    cleanup_files()
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Downloaded: {len(downloaded)} archive files")
    print(f"Extracted: {video_count} videos")
    print(f"Preprocessed: {roi_count} ROI videos")
    print(f"Success rate: {roi_count/video_count*100:.1f}%")
    print("\nIntermediate files cleaned up to free disk space")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
