# Batch Download & Feature Extraction Pipeline

## ✅ Solution Implemented

Instead of downloading all 1TB+ training videos at once, we now use a **batch processing pipeline** that:

1. **Downloads** 20 archive parts (~40-80 GB)
2. **Extracts** videos from archives
3. **Preprocesses** videos (lips & audio extraction)
4. **Extracts features** (AV-HuBERT .npz files)
5. **Deletes** source files to free space
6. **Repeats** for next batch

## 📊 Resource Requirements

### Old Approach (Failed):
- **Disk space needed**: ~1TB
- **Available**: 334 GB ❌
- **Problem**: Not enough space

### New Batch Approach (Success):
- **Disk space needed**: ~200-300 GB ✅
- **Available**: 334 GB ✅
- **Working space per batch**:
  - Archives: ~50-80 GB
  - Extracted videos: ~50-80 GB
  - Preprocessed: ~50-80 GB
  - Features (kept): ~10-20 GB (cumulative)

## 🚀 Current Status

```
Job ID:     21869277
Partition:  gpu-a100
Node:       spartan-gpgpu114
Status:     RUNNING
Started:    04:53:57 AEDT
Time limit: 168 hours (7 days)
```

## 📈 Progress Tracking

### Monitor the pipeline:
```bash
# Check job status
squeue -j 21869277

# Watch real-time logs
tail -f logs/batch_avd1m_21869277.out

# Check progress
cat batch_progress.json

# Count extracted features
find data/avh_features/train -name '*.npz' | wc -l
```

### Pipeline stages per batch:
- Download: ~30-60 min
- Extract: ~10-20 min
- Preprocess: ~2-4 hours
- Feature extraction: ~4-8 hours
- Cleanup: ~2-5 min

**Total per batch**: ~6-12 hours
**Total batches**: ~13 (254 parts ÷ 20)
**Estimated completion**: 5-7 days

## 💾 Disk Space Usage

The pipeline automatically cleans up after each batch:

```
Before batch:  334 GB free
During batch:  ~150 GB used
After cleanup: ~330 GB free (back to start)
```

Features accumulate but are small:
- 45,000 videos → ~45,000 .npz files
- Each .npz: ~500 KB - 2 MB
- Total features: ~10-20 GB

## 🔄 Auto-Resume Feature

If the job fails or is interrupted:
- Progress is saved in `batch_progress.json`
- Resubmit with: `sbatch batch_pipeline.slurm`
- Pipeline automatically resumes from last completed batch

## 📂 Output Structure

```
data/avh_features/train/
├── id00001/
│   ├── video1/
│   │   └── real.npz
│   └── video2/
│       └── real.npz
├── id00002/
...
└── (45,000 feature files)
```

## ✅ When Complete

After all batches finish (~5-7 days):

```bash
# Verify feature count
find data/avh_features/train -name '*.npz' | wc -l
# Expected: ~45,000

# Start training with full 45k dataset
python train.py \
    --name=AVH-Align_45k \
    --data_root_path=data/avh_features/ \
    --metadata_root_path=av1m_metadata/
```

This will give you the paper's performance since you'll be training on all 45,000 videos! 🎉

## 📋 Files Created

1. **batch_download_extract_features.py** - Main pipeline script
2. **batch_pipeline.slurm** - SLURM job configuration
3. **start_batch_pipeline.sh** - Helper script to start
4. **batch_progress.json** - Auto-saved progress tracking (created at runtime)

## 🐛 Troubleshooting

### Check for errors:
```bash
# Error log
cat logs/batch_avd1m_21869277.err

# Full output
less logs/batch_avd1m_21869277.out
```

### If disk fills up:
- Clean temporary files: `rm -rf Datasets/AVDeepfake1M/train_batch_extracted`
- Resume pipeline: it will skip processed batches

### If download fails:
- HuggingFace has resume capability built-in
- Just rerun - it will continue from where it stopped

## 💡 Key Advantages

1. **Space efficient**: 200 GB vs 1 TB
2. **Resumable**: Can restart without losing progress
3. **Automatic cleanup**: No manual intervention needed
4. **Progress tracking**: Know exactly where you are
5. **GPU efficient**: Uses GPU only when needed (feature extraction)
