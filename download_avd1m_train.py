#!/usr/bin/env python3
"""
Download AV-Deepfake1M Training Set from Hugging Face
Training videos are stored as multi-volume ZIP archives that need extraction
"""

import os
import subprocess
from pathlib import Path

def download_dataset():
    """Download the full AVDeepfake1M dataset including training ZIP archives."""
    
    try:
        from huggingface_hub import login, hf_hub_download, list_repo_files
        print("✓ huggingface_hub library found")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call(['pip', 'install', '-q', 'huggingface_hub'])
        from huggingface_hub import login, hf_hub_download, list_repo_files
    
    # Configuration
    token = "HF_TOKEN_REMOVED"
    output_dir = "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M"
    cache_dir = "/data/projects/punim2637/nnliang/.cache/huggingface"
    repo_id = "ControlNet/AV-Deepfake1M"
    
    print("="*70)
    print("Downloading AV-Deepfake1M Dataset")
    print("="*70)
    print(f"Repository: {repo_id}")
    print(f"Output: {output_dir}")
    print(f"Cache: {cache_dir}")
    print()
    
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    try:
        login(token=token, add_to_git_credential=True)
        print("✓ Successfully logged in")
    except Exception as e:
        print(f"Warning: Login issue: {e}")
        print("Continuing anyway...")
    
    print()
    
    # Create directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    train_archives_dir = os.path.join(output_dir, "train_archives")
    Path(train_archives_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Find all training ZIP parts
    print("Finding training archive parts...")
    try:
        all_files = list_repo_files(repo_id, repo_type="dataset", token=token)
        train_zips = sorted([f for f in all_files if f.startswith("train/train.zip.")])
        print(f"✓ Found {len(train_zips)} training archive parts")
        for idx, f in enumerate(train_zips[:5], 1):
            print(f"  {idx}. {f}")
        if len(train_zips) > 5:
            print(f"  ... and {len(train_zips) - 5} more")
    except Exception as e:
        print(f"✗ Error listing files: {e}")
        return False
    
    print()
    
    # Step 2: Download all ZIP parts
    print("Downloading training archive parts...")
    print("This will download ~1TB+ of data - may take 24-72 hours")
    print()
    
    downloaded_parts = []
    for idx, zip_file in enumerate(train_zips, 1):
        try:
            print(f"[{idx}/{len(train_zips)}] Downloading {zip_file}...")
            local_file = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=zip_file,
                local_dir=output_dir,
                cache_dir=cache_dir,
                token=token,
                resume_download=True
            )
            downloaded_parts.append(local_file)
            print(f"  ✓ Saved to: {local_file}")
            
            # Show progress
            if idx % 10 == 0:
                size_gb = sum(os.path.getsize(p) for p in downloaded_parts) / (1024**3)
                print(f"  Progress: {idx}/{len(train_zips)} parts ({size_gb:.1f} GB so far)")
        except Exception as e:
            print(f"  ✗ Error downloading {zip_file}: {e}")
            print("  You can resume this download later")
            return False
    
    print("\n" + "="*70)
    print("✓ All archive parts downloaded!")
    print("="*70)
    
    total_size_gb = sum(os.path.getsize(p) for p in downloaded_parts) / (1024**3)
    print(f"Total downloaded: {total_size_gb:.2f} GB")
    print()
    
    # Step 3: Extract archives
    print("Extracting training videos...")
    print("This will take several hours...")
    print()
    
    first_part = os.path.join(output_dir, "train", "train.zip.001")
    if os.path.exists(first_part):
        try:
            # Check if 7z is available
            subprocess.run(['7z', '--help'], capture_output=True, check=True)
            
            print(f"Running: 7z x {first_part}")
            result = subprocess.run(
                ['7z', 'x', first_part, f'-o{output_dir}/train_extracted', '-y'],
                cwd=os.path.dirname(first_part),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("\n✓ Extraction completed successfully!")
                
                # Count videos
                train_dir = os.path.join(output_dir, "train_extracted")
                if os.path.exists(train_dir):
                    result = subprocess.run(
                        ['find', train_dir, '-name', '*.mp4', '-type', 'f'],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    video_count = len([l for l in result.stdout.strip().split('\n') if l])
                    print(f"✓ Extracted {video_count} training videos")
                    
                    # Move to final location
                    final_train_dir = os.path.join(output_dir, "train")
                    if not os.path.exists(os.path.join(final_train_dir, "id00001")):
                        print("Moving videos to final location...")
                        subprocess.run(['mv', train_dir, final_train_dir], check=False)
            else:
                print(f"✗ Extraction failed with code: {result.returncode}")
                print("You'll need to extract manually using: 7z x train.zip.001")
                return False
                
        except FileNotFoundError:
            print("✗ 7z not found. Install with: sudo apt install p7zip-full p7zip-rar")
            print("Then run manually: cd train && 7z x train.zip.001")
            return False
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
            return False
    
    print("\n" + "="*70)
    print("✓ Training dataset ready!")
    print("="*70)
    
    return True


if __name__ == '__main__':
    import sys
    success = download_dataset()
    sys.exit(0 if success else 1)
