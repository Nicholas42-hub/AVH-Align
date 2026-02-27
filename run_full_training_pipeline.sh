#!/bin/bash
# 
# Workflow to train AVH-Align on FULL 45k training set
# Run this to submit all jobs in the correct order
#

echo "========================================================"
echo "  AVH-Align Training Pipeline - FULL DATASET (45k)"
echo "========================================================"
echo ""
echo "This will process the FULL 45k training set, not just 5k"
echo ""

cd /data/projects/punim2637/nnliang/AVH-Align

# Step 1: Preprocess training set (45k videos)
echo "Step 1: Submitting preprocessing for training set (45k videos)..."
JOB1=$(sbatch --parsable avh_preprocess_train_45k.slurm)
echo "  Job ID: $JOB1"

# Step 2: Preprocess validation set (5k videos) 
echo "Step 2: Submitting preprocessing for validation set (5k videos)..."
JOB2=$(sbatch --parsable avh_preprocess_val.slurm)
echo "  Job ID: $JOB2"

# Step 3: Feature extraction for training set (depends on JOB1)
echo "Step 3: Submitting feature extraction for training set (depends on $JOB1)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 avh_features_train_45k.slurm)
echo "  Job ID: $JOB3"

# Step 4: Feature extraction for validation set (depends on JOB2)
echo "Step 4: Submitting feature extraction for validation set (depends on $JOB2)..."
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 avh_features_val.slurm)
echo "  Job ID: $JOB4"

echo ""
echo "========================================================"
echo "  All jobs submitted!"
echo "========================================================"
echo ""
echo "Job chain:"
echo "  1. Preprocess training (45k):   $JOB1"
echo "  2. Preprocess validation (5k):  $JOB2"
echo "  3. Extract train features:      $JOB3 (after $JOB1)"
echo "  4. Extract val features:        $JOB4 (after $JOB2)"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Check logs in logs/ directory"
echo ""
echo "After all jobs complete, train the model with:"
echo "  python train.py --name=AVH-Align_45k --data_root_path=data/avh_features/ --metadata_root_path=av1m_metadata/"
echo ""
