# Scratch Space Processing Guide

## 📦 Overview

All datasets and intermediate processing files are now stored in **SCRATCH** space to save project quota. Only final **NPZ feature files** are stored in permanent PROJECTS storage.

### Storage Layout

```
SCRATCH (/data/scratch/projects/punim2637/nnliang/)
├── Datasets/
│   ├── AVDeepfake1M/          # 28GB (57,340 videos)
│   └── FakeAVCeleb_v1.2/      # 6.3GB (21,544 videos)
├── avd1m_preprocessed/         # Temporary ROI videos
├── avd1m_features_temp/        # Temporary features
├── favc_preprocessed/          # Temporary ROI videos
└── favc_features_temp/         # Temporary features

PROJECTS (/data/projects/punim2637/nnliang/AVH-Align/data/)
├── avh_features/               # Permanent NPZ files only
└── favc_features/              # Permanent NPZ files only
```

---

## 🚀 Processing Workflow

### AVDeepfake1M (57,340 videos)

**Step 1: Preprocess**
```bash
sbatch avd1m_scratch_preprocess.slurm
# Reads from: SCRATCH/Datasets/AVDeepfake1M/
# Writes to:  SCRATCH/avd1m_preprocessed/
# Time: ~3-5 days (32 CPUs)
```

**Step 2: Extract Features**
```bash
sbatch avd1m_scratch_features.slurm
# Reads from: SCRATCH/avd1m_preprocessed/
# Writes to:  SCRATCH/avd1m_features_temp/ (temp)
# Copies to:  PROJECTS/data/avh_features/ (permanent)
# Time: ~4-7 days (GPU A100)
```

---

### FakeAVCeleb (21,544 videos)

**Step 1: Preprocess**
```bash
sbatch favc_scratch_preprocess.slurm
# Reads from: SCRATCH/Datasets/FakeAVCeleb_v1.2/
# Writes to:  SCRATCH/favc_preprocessed/
# Time: ~1-2 days (32 CPUs)
```

**Step 2: Extract Features**
```bash
sbatch favc_scratch_features.slurm
# Reads from: SCRATCH/favc_preprocessed/
# Writes to:  SCRATCH/favc_features_temp/ (temp)
# Copies to:  PROJECTS/data/favc_features/ (permanent)
# Time: ~2-3 days (GPU A100)
```

---

## 📊 Monitoring

### Quick Status Check
```bash
./check_all_scratch.sh          # Overview of all datasets
./check_avd1m_scratch.sh        # AVDeepfake1M details
./check_favc_scratch.sh         # FakeAVCeleb details
```

### View Logs
```bash
# Preprocessing logs
tail -f logs/avd1m_scratch_preprocess_<jobid>.out
tail -f logs/favc_scratch_preprocess_<jobid>.out

# Feature extraction logs
tail -f logs/avd1m_scratch_features_<jobid>.out
tail -f logs/favc_scratch_features_<jobid>.out
```

### Check Scratch Usage
```bash
check_scratch_usage             # Overall quota
check_project_quota            # Project quota
```

---

## 🎯 Current Status

### AVDeepfake1M
- **Job ID**: 21999143 (Preprocessing)
- **Status**: Waiting for resources
- **Target**: 57,340 videos → 57,340 NPZ files

### FakeAVCeleb
- **Job ID**: 21993211 (Preprocessing)
- **Status**: Waiting for resources
- **Target**: 21,544 videos → 21,544 NPZ files

---

## ⚠️ Important Notes

### Scratch Space Characteristics
- ✅ 1TB capacity (vs 50GB in projects)
- ⚠️ Files auto-deleted after 30-90 days of no access
- ❌ No backup (data loss cannot be recovered)
- 🎯 Best for: intermediate processing files

### Data Safety
- ✅ Original datasets: Keep in scratch (can re-download if lost)
- ✅ Preprocessed ROI: Temporary (can regenerate)
- ✅ **NPZ features**: Automatically copied to PROJECTS (permanent)
- 🔒 Always keep final NPZ files in projects

### Processing Strategy
1. All processing happens in SCRATCH (fast, large capacity)
2. Final NPZ files automatically copied to PROJECTS (permanent)
3. Temporary files remain in SCRATCH (auto-cleaned)
4. No manual copying needed!

---

## 📝 Space Savings

**Before (All in Projects):**
- AVDeepfake1M: 28GB dataset + 20GB preprocessed + 27GB features = 75GB
- FakeAVCeleb: 6.3GB dataset + 8GB preprocessed + 5GB features = 19GB
- **Total: ~94GB** (would exceed 50GB quota)

**After (Using Scratch):**
- AVDeepfake1M: Only 27GB features in projects
- FakeAVCeleb: Only 5GB features in projects
- **Total in Projects: ~32GB** (well within quota!)
- **Scratch usage: ~62GB** (plenty of space)

---

## 🔧 Troubleshooting

### If preprocessing fails
```bash
# Check logs for errors
tail -100 logs/avd1m_scratch_preprocess_<jobid>.err

# Resume from checkpoint (scripts skip already-processed files)
sbatch avd1m_scratch_preprocess.slurm
```

### If feature extraction fails
```bash
# Check GPU availability
squeue -p gpu-a100

# Resume (already-extracted features are skipped)
sbatch avd1m_scratch_features.slurm
```

### If scratch space fills up
```bash
# Check usage
check_scratch_usage

# Clean old temporary files (after NPZ copied to projects)
rm -rf /data/scratch/projects/punim2637/nnliang/avd1m_features_temp
rm -rf /data/scratch/projects/punim2637/nnliang/favc_features_temp
```

---

## 📞 Contact

For issues or questions about scratch space:
- Check HPC documentation: https://dashboard.hpc.unimelb.edu.au/
- Email: hpc-support@unimelb.edu.au

---

**Last Updated**: 26 February 2026
