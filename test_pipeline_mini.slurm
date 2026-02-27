#!/bin/bash
#SBATCH --job-name=test_mini
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_mini_%j.out
#SBATCH --error=logs/test_mini_%j.err

module purge
export PATH="/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/11.3.0/FFmpeg/4.4.2/bin:/data/projects/punim2637/nnliang/avh_env/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /data/projects/punim2637/nnliang/AVH-Align

echo "============================================"
echo "MINI PIPELINE TEST - 1 PART ONLY"
echo "============================================"
echo "Start time: $(date)"
echo ""

# Create test directories in /tmp
TEST_DIR="/tmp/mini_test_$$"
DOWNLOAD_DIR="$TEST_DIR/downloads"
EXTRACTED_DIR="$TEST_DIR/extracted"
PREPROCESSED_DIR="$TEST_DIR/preprocessed"
FEATURES_DIR="$TEST_DIR/features"

mkdir -p "$DOWNLOAD_DIR" "$EXTRACTED_DIR" "$PREPROCESSED_DIR" "$FEATURES_DIR"

echo "Test directory: $TEST_DIR"
echo ""

# Step 1: Download just 1 part
echo "=== STEP 1: Download 1 part ==="
python - << 'PYEOF'
import os
from huggingface_hub import hf_hub_download

repo_id = "ControlNet/AV-Deepfake1M"
token = "HF_TOKEN_REMOVED"
output_dir = os.environ.get("DOWNLOAD_DIR")
part = "part-00001-99baffd6-3cb2-480c-817b-c02e8e75b4f1-c000"

print(f"Downloading {part}...")
for i in range(1, 4):  # 3 sub-parts
    filename = f"train/{part}.7z.{i:03d}"
    print(f"  Downloading {filename}")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=output_dir,
        token=token
    )
print("✓ Download complete")
PYEOF

if [ $? -ne 0 ]; then
    echo "✗ Download failed"
    exit 1
fi

# Count downloaded files
NUM_FILES=$(find "$DOWNLOAD_DIR" -name "*.7z.*" | wc -l)
echo "Downloaded $NUM_FILES files"
echo ""

# Step 2: Extract
echo "=== STEP 2: Extract ==="
BASE_ARCHIVE="$DOWNLOAD_DIR/train/part-00001-99baffd6-3cb2-480c-817b-c02e8e75b4f1-c000.7z"
7z x "${BASE_ARCHIVE}.001" -o"$EXTRACTED_DIR" -y > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Extraction failed"
    exit 1
fi

NUM_VIDEOS=$(find "$EXTRACTED_DIR" -name "*.mp4" -type f | wc -l)
echo "✓ Extracted $NUM_VIDEOS videos"
echo ""

# Step 3: Create metadata (just first 5 videos for quick test)
echo "=== STEP 3: Create metadata ==="
METADATA="$TEST_DIR/metadata.csv"
echo "path,label,num_frames" > "$METADATA"
find "$EXTRACTED_DIR/train" -name "*.mp4" -type f | head -5 | while read video; do
    filename=$(basename "$video")
    echo "$filename,0,100" >> "$METADATA"
done

NUM_ENTRIES=$(($(wc -l < "$METADATA") - 1))
echo "✓ Created metadata: $NUM_ENTRIES videos"
echo ""

# Step 4: Preprocess
echo "=== STEP 4: Preprocess ==="
cd /data/projects/punim2637/nnliang/AVH-Align/av_hubert/avhubert
python deepfake_preprocess.py \
    --dataset AV1M \
    --split train \
    --metadata "$METADATA" \
    --data_path "$EXTRACTED_DIR" \
    --save_path "$PREPROCESSED_DIR" \
    --max_workers 4

if [ $? -ne 0 ]; then
    echo "✗ Preprocessing failed"
    exit 1
fi

# Check preprocessed output
NUM_ROI=$(find "$PREPROCESSED_DIR" -name "*_roi.mp4" -type f | wc -l)
NUM_WAV=$(find "$PREPROCESSED_DIR" -name "*.wav" -type f | wc -l)
echo ""
echo "✓ Preprocessing created:"
echo "  ROI videos: $NUM_ROI"
echo "  Audio files: $NUM_WAV"

if [ $NUM_ROI -eq 0 ]; then
    echo "✗ ERROR: No ROI videos created!"
    ls -lh "$PREPROCESSED_DIR/train/" | head -20
    exit 1
fi

# Show sample files
echo ""
echo "Sample preprocessed files:"
find "$PREPROCESSED_DIR" -name "*_roi.mp4" -type f | head -3 | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $(basename $f) - $SIZE"
done
echo ""

# Step 5: Extract features
echo "=== STEP 5: Extract features ==="
python /data/projects/punim2637/nnliang/AVH-Align/av_hubert/avhubert/deepfake_feature_extraction.py \
    --dataset AV1M \
    --split train \
    --metadata "$METADATA" \
    --ckpt_path /data/projects/punim2637/nnliang/AVH-Align/content/self_large_vox_433h.pt \
    --data_path "$PREPROCESSED_DIR" \
    --save_path "$FEATURES_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Feature extraction failed"
    exit 1
fi

# Check features
NUM_FEATURES=$(find "$FEATURES_DIR" -name "*.npz" -type f | wc -l)
echo ""
echo "✓ Feature extraction created:"
echo "  Features: $NUM_FEATURES .npz files"

if [ $NUM_FEATURES -eq 0 ]; then
    echo "✗ ERROR: No features created!"
    ls -lhR "$FEATURES_DIR" | head -20
    exit 1
fi

# Show sample features
echo ""
echo "Sample feature files:"
find "$FEATURES_DIR" -name "*.npz" -type f | head -3 | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $(basename $f) - $SIZE"
done

# Cleanup
echo ""
echo "=== Cleanup ==="
rm -rf "$TEST_DIR"
echo "✓ Cleaned up test directory"

echo ""
echo "============================================"
echo "✓ SUCCESS! Full pipeline works!"
echo "============================================"
echo "End time: $(date)"
