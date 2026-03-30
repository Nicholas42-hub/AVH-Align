# Processing Status Summary

## Current Progress

### Job Status
- AVDeepfake1M preprocessing was running under job `22009014`
  - Partition: `cascade`
  - Runtime at the time of this note: about 40 minutes
  - Status at the time of this note: processing 57,340 videos

- FakeAVCeleb feature extraction was running under job `22010184`
  - Partition: `gpu-a100`
  - Runtime at the time of this note: about 2 minutes
  - Status at the time of this note: extracting features from 4,089 preprocessed files

- FakeAVCeleb preprocessing still needed to continue
  - Completed: `4,089 / 21,544 (19.0%)`
  - Remaining: `17,455` videos

## Fixed Issues

### Dataset Name Bug
Fixed a dataset name mismatch in `deepfake_preprocess.py`:
- Issue: the script passed `--dataset AVDeepfake1M`, but the code only accepted `AV1M`
- Fix: changed the condition to `elif args.dataset in ['AV1M', 'AVDeepfake1M']:`
- File: `/data/projects/punim2637/nnliang/AVH-Align/av_hubert/avhubert/deepfake_preprocess.py`
- Result: AVDeepfake1M preprocessing worked correctly after the fix

## Monitoring Commands

### 1. Check the Queue
```bash
squeue -u nnliang
```

### 2. Tail the Logs
```bash
# AVDeepfake1M preprocessing
tail -f logs/avd1m_scratch_preprocess_22009014.out

# FakeAVCeleb feature extraction
tail -f logs/favc_scratch_features_22010184.out
```

### 3. Use the Monitoring Script
```bash
cd /data/projects/punim2637/nnliang/AVH-Align
./monitor_processing.sh
```

### 4. Quick Count Checks
```bash
# AVDeepfake1M preprocessing progress
ls -R /data/scratch/projects/punim2637/nnliang/avd1m_preprocessed/val/*/*/*/ | grep "_roi.mp4" | wc -l

# FakeAVCeleb feature count
find /data/projects/punim2637/nnliang/AVH-Align/data/favc_features -name "*.npz" | wc -l
```

## Next Steps

### After the Current Jobs Finish

1. **Submit AVDeepfake1M feature extraction**
   ```bash
   cd /data/projects/punim2637/nnliang/AVH-Align
   sbatch avd1m_scratch_features.slurm
   ```

2. **Continue FakeAVCeleb preprocessing**
   ```bash
   cd /data/projects/punim2637/nnliang/AVH-Align
   sbatch favc_scratch_preprocess.slurm
   ```

3. **Auto-submit feature extraction** (optional)
   ```bash
   # Auto-submit feature extraction when AVD1M preprocessing reaches 50,000 files
   nohup ./auto_submit_features.sh avd1m 50000 > auto_avd1m.log 2>&1 &
   
   # Auto-submit feature extraction when FakeAVCeleb preprocessing reaches 20,000 files
   nohup ./auto_submit_features.sh favc 20000 > auto_favc.log 2>&1 &
   ```

## Estimated Time

Based on the observed progress in this note:
- **AVDeepfake1M preprocessing**: about 18-24 hours for 57,340 videos
- **AVDeepfake1M feature extraction**: about 24-36 hours depending on GPU availability
- **FakeAVCeleb preprocessing**: about 4-6 hours for the remaining 17,455 videos
- **FakeAVCeleb feature extraction**: about 2-4 hours for 4,089 files

## Storage Info

- **Scratch**: 39GB / 1TB (4%)
- **Projects**: monitor feature file size because `.npz` files are copied there
- **Temporary files**: stored on scratch during preprocessing
- **Final files**: NPZ feature files stored under projects

## File Locations

### Input Data (Scratch)
- AVDeepfake1M: `/data/scratch/projects/punim2637/nnliang/Datasets/AVDeepfake1M/val`
- FakeAVCeleb: `/data/scratch/projects/punim2637/nnliang/Datasets/FakeAVCeleb_v1.2`

### Preprocessing Output (Scratch)
- AVDeepfake1M: `/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed/val`
- FakeAVCeleb: `/data/scratch/projects/punim2637/nnliang/favc_preprocessed`

### Feature Files (Projects)
- AVDeepfake1M: `/data/projects/punim2637/nnliang/AVH-Align/data/avh_features`
- FakeAVCeleb: `/data/projects/punim2637/nnliang/AVH-Align/data/favc_features`

## Notes

1. **GPU priority**: FakeAVCeleb feature extraction runs on GPU and may be preempted by higher-priority jobs
2. **Error handling**: some video files may fail to load because of corruption or unsupported formatting
3. **Scratch cleanup**: scratch files may be deleted after 30-90 days, so copy important outputs in time
4. **Partition guidance**: the cluster recommends `sapphire` instead of `cascade` when possible

## Troubleshooting

If a job fails or stops:
```bash
# Check error logs
tail -100 logs/avd1m_scratch_preprocess_*.err
tail -100 logs/favc_scratch_features_*.err

# Re-submit jobs
sbatch avd1m_scratch_preprocess.slurm
sbatch favc_scratch_features.slurm
```

---
**Last updated**: 2026-02-26 20:30
**Status**: Fix completed, jobs running
