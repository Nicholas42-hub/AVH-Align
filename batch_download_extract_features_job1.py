#!/usr/bin/env python3
"""
Batch Download & Feature Extraction Pipeline - JOB 1
Processes batches 0-6 (archive parts 1-140)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

# Configuration
TOKEN = "HF_TOKEN_REMOVED"
REPO_ID = "ControlNet/AV-Deepfake1M"
BASE_DIR = "/data/projects/punim2637/nnliang"
OUTPUT_DIR = f"{BASE_DIR}/Datasets/AVDeepfake1M"
CACHE_DIR = f"{BASE_DIR}/.cache/huggingface"
WORK_DIR = f"{BASE_DIR}/AVH-Align"
BATCH_SIZE = 3  # Download 3 archive parts at a time (minimized to avoid quota)

# Job-specific configuration
JOB_ID = "job1"
START_BATCH = 0  # Start from batch 0  
END_BATCH = 47   # Process through batch 46 (exclusive end) - covers parts 1-140 with batch_size=3

# Use node-local /tmp storage to avoid file count quota on project storage
TMP_DIR = "/tmp"
ARCHIVES_DIR = f"{OUTPUT_DIR}/train_archives_{JOB_ID}"  # Keep archives on project storage
EXTRACTED_DIR = f"{TMP_DIR}/avd1m_extracted_{JOB_ID}"  # Extract to node-local tmp
PREPROCESSED_DIR = f"{TMP_DIR}/avd1m_preprocessed_{JOB_ID}"  # Preprocess on node-local tmp
FEATURES_DIR = f"{WORK_DIR}/data/avh_features/train"  # Final features on project storage
METADATA_FILE = f"{WORK_DIR}/av1m_metadata/train_metadata.csv"
PROGRESS_FILE = f"{WORK_DIR}/batch_progress_{JOB_ID}.json"
BATCH_METADATA = f"{TMP_DIR}/batch_metadata_{JOB_ID}.csv"  # Metadata on tmp


def load_progress():
    """Load processing progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'last_completed_batch': START_BATCH, 'processed_parts': [], 'total_features': 0}


def save_progress(progress):
    """Save processing progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_archive_parts():
    """Get list of all training archive parts from HuggingFace."""
    from huggingface_hub import login, list_repo_files
    
    print("Logging in to Hugging Face...")
    login(token=TOKEN, add_to_git_credential=True)
    
    print("Fetching archive parts list...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset", token=TOKEN)
    train_zips = sorted([f for f in all_files if f.startswith("train/train.zip.")])
    
    print(f"✓ Found {len(train_zips)} archive parts")
    return train_zips


def download_batch(archive_parts, start_idx, batch_size):
    """Download a batch of archive parts."""
    from huggingface_hub import hf_hub_download
    
    end_idx = min(start_idx + batch_size, len(archive_parts))
    batch = archive_parts[start_idx:end_idx]
    
    print(f"\n{'='*70}")
    print(f"[{JOB_ID.upper()}] Downloading Batch: Parts {start_idx+1}-{end_idx} of {len(archive_parts)}")
    print(f"{'='*70}\n")
    print(f"Download directory: {ARCHIVES_DIR}")
    print(f"Output directory (HF local_dir): {OUTPUT_DIR}")
    print(f"")
    
    downloaded = []
    for idx, zip_file in enumerate(batch, start_idx+1):
        try:
            print(f"[{idx}/{len(archive_parts)}] Downloading {os.path.basename(zip_file)}...")
            local_file = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=zip_file,
                local_dir=OUTPUT_DIR,
                cache_dir=CACHE_DIR,
                token=TOKEN,
                resume_download=True
            )
            downloaded.append(local_file)
            size_mb = os.path.getsize(local_file) / (1024**2)
            print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
            print(f"    Saved to: {local_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    total_gb = sum(os.path.getsize(p) for p in downloaded) / (1024**3)
    print(f"\n✓ Batch downloaded: {len(downloaded)} parts ({total_gb:.2f} GB)")
    print(f"\nDownloaded files:")
    for f in downloaded:
        print(f"  {f}")
    return downloaded


def extract_batch(first_part_path):
    """Extract videos from archive batch."""
    print(f"\n{'='*70}")
    print(f"[{JOB_ID.upper()}] Extracting Videos from Archives")
    print(f"{'='*70}\n")
    
    # Create extraction directory
    Path(EXTRACTED_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting: {first_part_path}")
    print(f"Output: {EXTRACTED_DIR}")
    
    try:
        # Use 7z to extract (it will automatically use all .zip.XXX parts)
        result = subprocess.run(
            ['7z', 'x', first_part_path, f'-o{EXTRACTED_DIR}', '-y'],
            cwd=os.path.dirname(first_part_path),
            capture_output=False,  # Show extraction progress in real-time
            timeout=3600  # 1 hour timeout
        )
        
        # Count extracted videos (even if 7z reported errors, many files may be extracted)
        video_count = int(subprocess.check_output(
            f"find {EXTRACTED_DIR} -name '*.mp4' | wc -l",
            shell=True,
            text=True
        ).strip())
        
        if video_count > 0:
            print(f"✓ Extracted {video_count} videos")
            if result.returncode != 0:
                print(f"  (Note: 7z reported some errors but extraction succeeded)")
            
            # Show extracted directory structure
            print(f"\nExtracted directory structure:")
            subprocess.run(f"ls -lh {EXTRACTED_DIR} | head -10", shell=True)
            if os.path.exists(f"{EXTRACTED_DIR}/train"):
                print(f"\nContents of {EXTRACTED_DIR}/train/:")
                subprocess.run(f"ls {EXTRACTED_DIR}/train | head -20", shell=True)
                print(f"\nSample video files:")
                subprocess.run(f"find {EXTRACTED_DIR}/train -name '*.mp4' -type f | head -5", shell=True)
            
            return video_count
        else:
            print(f"✗ No videos extracted")
            return 0
    except Exception as e:
        print(f"✗ Error during extraction: {e}")
        return 0


def preprocess_batch():
    """Preprocess videos (extract lips, audio)."""
    print(f"\n{'='*70}")
    print(f"[{JOB_ID.upper()}] Preprocessing Videos")
    print(f"{'='*70}\n")
    
    # Create temporary metadata for this batch on /tmp
    batch_metadata = BATCH_METADATA
    
    # Find all videos in extracted directory and create metadata
    videos = subprocess.check_output(
        f"find {EXTRACTED_DIR} -name '*.mp4' -type f",
        shell=True,
        text=True
    ).strip().split('\n')
    
    videos = [v for v in videos if v]  # Filter empty strings
    
    print(f"Found {len(videos)} videos for metadata")
    if len(videos) > 0:
        print(f"\nSample video paths (first 3):")
        for v in videos[:3]:
            print(f"  {v}")
    
    # Create metadata CSV for this batch
    with open(batch_metadata, 'w') as f:
        f.write("path,label,num_frames\n")
        for video in videos:
            rel_path = os.path.relpath(video, EXTRACTED_DIR)
            # Remove 'train/' prefix if present (preprocessing script adds it)
            if rel_path.startswith('train/'):
                rel_path = rel_path[6:]  # Remove 'train/' prefix
            f.write(f"{rel_path},0,100\n")  # Label 0 for real, frames estimated
    
    print(f"Created batch metadata: {len(videos)} videos")
    print(f"Metadata file: {batch_metadata}")
    print(f"\nMetadata sample (first 5 entries):")
    subprocess.run(f"head -6 {batch_metadata}", shell=True)
    print(f"\nVerifying extracted videos exist:")
    subprocess.run(f"find {EXTRACTED_DIR} -name '*.mp4' -type f | head -3", shell=True)
    print(f"\nMetadata sample (first 5 entries):")
    subprocess.run(f"head -6 {batch_metadata}", shell=True)
    print(f"\nActual video paths (sample):")
    subprocess.run(f"find {EXTRACTED_DIR} -name '*.mp4' -type f | head -3", shell=True)
    
    # Run preprocessing
    preprocess_script = f"{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py"
    
    cmd = [
        'python', preprocess_script,
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', batch_metadata,
        '--data_path', EXTRACTED_DIR,  # Script adds /train automatically
        '--save_path', PREPROCESSED_DIR,
        '--max_workers', '16'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"\nPreprocessing input path: {EXTRACTED_DIR}")
    print(f"Preprocessing will look for videos at: {EXTRACTED_DIR}/train/")
    print(f"Preprocessing output path: {PREPROCESSED_DIR}")
    
    result = subprocess.run(cmd, cwd=f"{WORK_DIR}/av_hubert/avhubert")
    
    if result.returncode == 0:
        # Verify files were actually created
        roi_count = int(subprocess.check_output(
            f"find {PREPROCESSED_DIR} -name '*_roi.mp4' -type f | wc -l",
            shell=True,
            text=True
        ).strip())
        wav_count = int(subprocess.check_output(
            f"find {PREPROCESSED_DIR} -name '*.wav' -type f | wc -l",
            shell=True,
            text=True
        ).strip())
        
        print(f"\n✓ Preprocessing completed")
        print(f"  Created {roi_count} ROI videos and {wav_count} audio files")
        
        # Show preprocessing directory structure
        print(f"\nPreprocessed directory structure:")
        subprocess.run(f"ls -lh {PREPROCESSED_DIR} | head -10", shell=True)
        if os.path.exists(f"{PREPROCESSED_DIR}/train"):
            print(f"\nSample files in preprocessed/train:")
            subprocess.run(f"ls {PREPROCESSED_DIR}/train | head -10", shell=True)
        
        if roi_count == 0:
            print("✗ WARNING: No output files created despite successful return code!")
            # List sample of what's in preprocessed dir for debugging
            sample = subprocess.check_output(
                f"ls -lah {PREPROCESSED_DIR} | head -20",
                shell=True,
                text=True
            )
            print(f"Sample of preprocessed directory:\n{sample}")
            return False
        
        return True
    else:
        print("✗ Preprocessing failed")
        return False


def extract_features_batch():
    """Extract AV-HuBERT features from preprocessed videos."""
    print(f"\n{'='*70}")
    print(f"[{JOB_ID.upper()}] Extracting Features")
    print(f"{'='*70}\n")
    
    # Use the same temporary metadata
    batch_metadata = BATCH_METADATA
    
    feature_script = f"{WORK_DIR}/av_hubert/avhubert/deepfake_feature_extraction.py"
    checkpoint = f"{WORK_DIR}/av_hubert/avhubert/self_large_vox_433h.pt"
    
    cmd = [
        'python', feature_script,
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', batch_metadata,
        '--ckpt_path', checkpoint,
        '--data_path', PREPROCESSED_DIR,
        '--save_path', FEATURES_DIR
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"\nFeature extraction input path: {PREPROCESSED_DIR}")
    print(f"Feature extraction output path: {FEATURES_DIR}")
    print(f"\nChecking preprocessed files before feature extraction:")
    roi_before = int(subprocess.check_output(
        f"find {PREPROCESSED_DIR} -name '*_roi.mp4' -type f | wc -l",
        shell=True,
        text=True
    ).strip())
    print(f"  ROI videos available: {roi_before}")
    
    result = subprocess.run(cmd, cwd=f"{WORK_DIR}/av_hubert/avhubert")
    
    if result.returncode == 0:
        # Count new features
        feature_count = int(subprocess.check_output(
            f"find {FEATURES_DIR} -name '*.npz' | wc -l",
            shell=True,
            text=True
        ).strip())
        print(f"\n✓ Feature extraction completed ({feature_count} total features)")
        
        if feature_count > 0:
            print(f"\nSample feature files:")
            subprocess.run(f"find {FEATURES_DIR} -name '*.npz' -type f | head -5", shell=True)
        else:
            print(f"\n✗ WARNING: No features created!")
            print(f"Checking features directory:")
            subprocess.run(f"ls -lhR {FEATURES_DIR} | head -20", shell=True)
        
        return feature_count
    else:
        print("✗ Feature extraction failed")
        return 0


def cleanup_batch():
    """Delete downloaded archives and extracted videos to free space."""
    print(f"\n{'='*70}")
    print(f"[{JOB_ID.upper()}] Cleaning Up Batch Files")
    print(f"{'='*70}\n")
    
    # Calculate sizes before deletion
    archives_size = float(subprocess.check_output(
        f"du -sb {ARCHIVES_DIR} 2>/dev/null | cut -f1 || echo 0",
        shell=True,
        text=True
    ).strip()) / (1024**3)
    
    extracted_size = float(subprocess.check_output(
        f"du -sb {EXTRACTED_DIR} 2>/dev/null | cut -f1 || echo 0",
        shell=True,
        text=True
    ).strip()) / (1024**3)
    
    preprocessed_size = float(subprocess.check_output(
        f"du -sb {PREPROCESSED_DIR} 2>/dev/null | cut -f1 || echo 0",
        shell=True,
        text=True
    ).strip()) / (1024**3)
    
    total_freed = archives_size + extracted_size + preprocessed_size
    
    print(f"Deleting archives ({archives_size:.2f} GB)...")
    shutil.rmtree(ARCHIVES_DIR, ignore_errors=True)
    
    print(f"Deleting extracted videos ({extracted_size:.2f} GB)...")
    shutil.rmtree(EXTRACTED_DIR, ignore_errors=True)
    
    print(f"Deleting preprocessed videos ({preprocessed_size:.2f} GB)...")
    shutil.rmtree(PREPROCESSED_DIR, ignore_errors=True)
    
    # Recreate directories for next batch
    Path(ARCHIVES_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Freed {total_freed:.2f} GB of disk space")


def main():
    """Main batch processing pipeline."""
    
    try:
        from huggingface_hub import login
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call(['pip', 'install', '-q', 'huggingface_hub'])
    
    print("="*70)
    print(f"AV-Deepfake1M Pipeline - {JOB_ID.upper()}")
    print(f"Processing batches {START_BATCH} to {END_BATCH-1}")
    print("="*70)
    print(f"Batch size: {BATCH_SIZE} archive parts")
    print(f"Features output: {FEATURES_DIR}")
    print()
    
    # Load progress
    progress = load_progress()
    start_batch = max(progress['last_completed_batch'], START_BATCH)
    
    if start_batch > START_BATCH:
        print(f"Resuming from batch {start_batch + 1}")
        print(f"Already processed: {progress['total_features']} features")
        print()
    
    # Get all archive parts
    archive_parts = get_archive_parts()
    
    # Process batches within our range
    for batch_num in range(start_batch, END_BATCH):
        start_idx = batch_num * BATCH_SIZE
        
        if start_idx >= len(archive_parts):
            break
        
        print(f"\n{'#'*70}")
        print(f"# [{JOB_ID.upper()}] BATCH {batch_num + 1}")
        print(f"{'#'*70}")
        
        # Step 1: Download batch
        downloaded = download_batch(archive_parts, start_idx, BATCH_SIZE)
        if not downloaded:
            print("✗ Download failed, stopping")
            return False
        
        progress['processed_parts'].extend(downloaded)
        
        # Step 2: Extract videos
        video_count = extract_batch(downloaded[0])
        if video_count == 0:
            print("✗ Extraction failed, stopping")
            return False
        
        # Step 3: Preprocess videos
        if not preprocess_batch():
            print("✗ Preprocessing failed, stopping")
            return False
        
        # Step 4: Extract features
        feature_count = extract_features_batch()
        if feature_count == 0:
            print("✗ Feature extraction failed, stopping")
            return False
        
        progress['total_features'] = feature_count
        
        # Step 5: Cleanup
        cleanup_batch()
        
        # Save progress
        progress['last_completed_batch'] = batch_num + 1
        save_progress(progress)
        
        print(f"\n✓ [{JOB_ID.upper()}] Batch {batch_num + 1} complete!")
        print(f"  Total features: {feature_count}")
    
    print("\n" + "="*70)
    print(f"✓ [{JOB_ID.upper()}] ALL BATCHES COMPLETED!")
    print("="*70)
    print(f"Features saved to: {FEATURES_DIR}")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
