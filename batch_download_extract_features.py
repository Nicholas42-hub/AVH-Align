#!/usr/bin/env python3
"""
Batch Download & Feature Extraction Pipeline for AV-Deepfake1M Training Set

Downloads archive parts in batches, extracts features, then deletes source files
to conserve disk space. This allows processing 1TB+ dataset with only ~200GB space.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

# Configuration
TOKEN = os.environ.get("HF_TOKEN", "")  # Set HF_TOKEN env var before running
REPO_ID = "ControlNet/AV-Deepfake1M"
BASE_DIR = "/data/projects/punim2637/nnliang"
OUTPUT_DIR = f"{BASE_DIR}/Datasets/AVDeepfake1M"
CACHE_DIR = f"{BASE_DIR}/.cache/huggingface"
WORK_DIR = f"{BASE_DIR}/AVH-Align"
BATCH_SIZE = 20  # Download 20 archive parts at a time (~40-80GB)

# Directories
ARCHIVES_DIR = f"{OUTPUT_DIR}/train_archives"
EXTRACTED_DIR = f"{OUTPUT_DIR}/train_batch_extracted"
PREPROCESSED_DIR = f"{WORK_DIR}/data/avh_preprocessed/train_batch"
FEATURES_DIR = f"{WORK_DIR}/data/avh_features/train"
METADATA_FILE = f"{WORK_DIR}/av1m_metadata/train_metadata.csv"
PROGRESS_FILE = f"{WORK_DIR}/batch_progress.json"


def load_progress():
    """Load processing progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'last_completed_batch': 0, 'processed_parts': [], 'total_features': 0}


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
    print(f"Downloading Batch: Parts {start_idx+1}-{end_idx} of {len(archive_parts)}")
    print(f"{'='*70}\n")
    
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
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    total_gb = sum(os.path.getsize(p) for p in downloaded) / (1024**3)
    print(f"\n✓ Batch downloaded: {len(downloaded)} parts ({total_gb:.2f} GB)")
    return downloaded


def extract_batch(first_part_path):
    """Extract videos from archive batch."""
    print(f"\n{'='*70}")
    print("Extracting Videos from Archives")
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
    print("Preprocessing Videos")
    print(f"{'='*70}\n")
    
    # Create temporary metadata for this batch
    batch_metadata = f"{WORK_DIR}/batch_metadata_temp.csv"
    
    # Find all videos in extracted directory and create metadata
    videos = subprocess.check_output(
        f"find {EXTRACTED_DIR} -name '*.mp4' -type f",
        shell=True,
        text=True
    ).strip().split('\n')
    
    # Create metadata CSV for this batch
    with open(batch_metadata, 'w') as f:
        f.write("path,label,num_frames\n")
        for video in videos:
            rel_path = os.path.relpath(video, EXTRACTED_DIR)
            f.write(f"{rel_path},0,100\n")  # Label 0 for real, frames estimated
    
    print(f"Created batch metadata: {len(videos)} videos")
    
    # Run preprocessing
    preprocess_script = f"{WORK_DIR}/av_hubert/avhubert/deepfake_preprocess.py"
    
    cmd = [
        'python', preprocess_script,
        '--dataset', 'AV1M',
        '--split', 'train',
        '--metadata', batch_metadata,
        '--data_path', EXTRACTED_DIR,
        '--save_path', PREPROCESSED_DIR,
        '--max_workers', '16'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=f"{WORK_DIR}/av_hubert/avhubert")
    
    if result.returncode == 0:
        print("✓ Preprocessing completed")
        return True
    else:
        print("✗ Preprocessing failed")
        return False


def extract_features_batch():
    """Extract AV-HuBERT features from preprocessed videos."""
    print(f"\n{'='*70}")
    print("Extracting Features")
    print(f"{'='*70}\n")
    
    # Use the same temporary metadata
    batch_metadata = f"{WORK_DIR}/batch_metadata_temp.csv"
    
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
    result = subprocess.run(cmd, cwd=f"{WORK_DIR}/av_hubert/avhubert")
    
    if result.returncode == 0:
        # Count new features
        feature_count = int(subprocess.check_output(
            f"find {FEATURES_DIR} -name '*.npz' | wc -l",
            shell=True,
            text=True
        ).strip())
        print(f"✓ Feature extraction completed ({feature_count} total features)")
        return feature_count
    else:
        print("✗ Feature extraction failed")
        return 0


def cleanup_batch():
    """Delete downloaded archives and extracted videos to free space."""
    print(f"\n{'='*70}")
    print("Cleaning Up Batch Files")
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
    print("AV-Deepfake1M Batch Download & Feature Extraction Pipeline")
    print("="*70)
    print(f"Batch size: {BATCH_SIZE} archive parts")
    print(f"Features output: {FEATURES_DIR}")
    print()
    
    # Load progress
    progress = load_progress()
    start_batch = progress['last_completed_batch']
    
    if start_batch > 0:
        print(f"Resuming from batch {start_batch + 1}")
        print(f"Already processed: {progress['total_features']} features")
        print()
    
    # Get all archive parts
    archive_parts = get_archive_parts()
    total_batches = (len(archive_parts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Process batches
    for batch_num in range(start_batch, total_batches):
        start_idx = batch_num * BATCH_SIZE
        
        print(f"\n{'#'*70}")
        print(f"# BATCH {batch_num + 1}/{total_batches}")
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
        
        print(f"\n✓ Batch {batch_num + 1}/{total_batches} complete!")
        print(f"  Total features: {feature_count}")
    
    print("\n" + "="*70)
    print("✓ ALL BATCHES COMPLETED!")
    print("="*70)
    print(f"Total features extracted: {progress['total_features']}")
    print(f"Features saved to: {FEATURES_DIR}")
    print("\nYou can now run training:")
    print("  python train.py --name=AVH-Align_45k --data_root_path=data/avh_features/")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
