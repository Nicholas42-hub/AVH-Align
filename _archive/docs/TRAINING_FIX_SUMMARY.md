# Training Set Fix - AVH-Align

## Problem Identified

Your training setup was only using **5,000 videos** from the validation set, not the full **45,000 videos** from the training set. This is why your performance was lower than the paper.

### What was wrong:

1. **`avh_preprocess_train.slurm`**: 
   - Used `val_labels.csv` (5k videos)
   - Used `--split test` which reads from `val/` directory

2. **`avh_features_train.slurm`**:
   - Used `val_labels.csv` (5k videos)
   - Extracted features only from validation set

3. **`train.py`** line 88:
   - Validation dataset pointed to `train` directory instead of `val`

## Solution Applied

### New SLURM Scripts Created:

1. **Training set (45k videos)**:
   - [`avh_preprocess_train_45k.slurm`](avh_preprocess_train_45k.slurm) - Preprocesses 45k training videos
   - [`avh_features_train_45k.slurm`](avh_features_train_45k.slurm) - Extracts features from 45k videos
   - Uses: `train_metadata.csv` with `--split train`
   - Reads from: `data_path/train/`
   - Saves to: `data/avh_features/train/`

2. **Validation set (5k videos)**:
   - [`avh_preprocess_val.slurm`](avh_preprocess_val.slurm) - Preprocesses 5k validation videos
   - [`avh_features_val.slurm`](avh_features_val.slurm) - Extracts features from 5k videos  
   - Uses: `val_metadata.csv` with `--split test`
   - Reads from: `data_path/val/`
   - Saves to: `data/avh_features/val/`

### Code Files Updated:

1. **[config.py](config.py#L41)**:
   - Changed default `data_root_path` from `av1m_features/` to `data/avh_features/`

2. **[train.py](train.py#L88)**:
   - Fixed validation dataset to read from `val` directory (was incorrectly using `train`)

## How to Run

Execute the automated pipeline:

```bash
./run_full_training_pipeline.sh
```

This will submit all 4 jobs in the correct order with dependencies.

### Or run manually:

```bash
# 1. Preprocess training set (45k videos)
sbatch avh_preprocess_train_45k.slurm

# 2. Preprocess validation set (5k videos)
sbatch avh_preprocess_val.slurm

# 3. After preprocessing completes, extract features
sbatch avh_features_train_45k.slurm
sbatch avh_features_val.slurm

# 4. Train the model
python train.py --name=AVH-Align_45k --data_root_path=data/avh_features/ --metadata_root_path=av1m_metadata/
```

## Expected Results

After these changes, your model should achieve performance comparable to the paper since you'll be training on the full 45,000 real videos instead of just 5,000.

## Processing Time Estimates

- Preprocessing 45k videos: ~24-48 hours
- Feature extraction 45k videos: ~24-48 hours  
- Training: Depends on epochs and hardware

## Disk Space Requirements

With 45k training videos + 5k validation videos, expect:
- Preprocessed videos: ~500GB - 1TB
- Extracted features: ~100-200GB

Make sure you have sufficient storage space before starting.
