#!/bin/bash
#SBATCH --job-name=avh_sup_train
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/avh_sup_train_%j.out
#SBATCH --error=logs/avh_sup_train_%j.err

# ============================================================
#  Step 3: Train AVH-Align supervised model on AV-Deepfake1M val split
#
#  Uses val_labels.csv (5k clips) as training data
#  and test_labels.csv (10k clips) for final evaluation.
#  Features must already be extracted to data/avh_features/val/
# ============================================================

echo "============================================================"
echo "  AVH-Align Supervised Training"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
echo "Python: $(which python) — $(python --version)"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

FEATURES_DIR=data/avh_features
CSV_DIR=avh_sup/csv_metadata/av1m

# ---- Setup: use val clips as training data ----
echo "Setting up train/val splits from val/ features..."

# Symlink train -> val so the dataset loader finds features at root_path/train/
if [ ! -e "$FEATURES_DIR/train" ]; then
    ln -s /data/projects/punim2637/nnliang/AVH-Align/$FEATURES_DIR/val \
          /data/projects/punim2637/nnliang/AVH-Align/$FEATURES_DIR/train
    echo "Created symlink: $FEATURES_DIR/train -> $FEATURES_DIR/val"
else
    echo "Symlink/dir $FEATURES_DIR/train already exists."
fi

# Use pre-generated train/val CSVs (generated from all available NPZs, 80/20 split)
echo "Using pre-generated CSVs in $CSV_DIR"

# Sanity check
TRAIN_FEATS=$(tail -n +2 $CSV_DIR/train_labels.csv | wc -l)
VAL_FEATS=$(tail -n +2 $CSV_DIR/val_labels.csv | wc -l)
FEAT_FILES=$(find $FEATURES_DIR/val -name "*.npz" 2>/dev/null | wc -l)
echo ""
echo "Train clips : $TRAIN_FEATS"
echo "Val clips   : $VAL_FEATS"
echo "Feature npz : $FEAT_FILES"
echo ""

if [ "$FEAT_FILES" -eq 0 ]; then
    echo "ERROR: No .npz feature files found in $FEATURES_DIR/val/"
    echo "       Run avh_extract_features.slurm and avh_features_train.slurm first!"
    exit 1
fi

# Install lightning if not present
python -c "import lightning" 2>/dev/null || \
    pip install lightning --quiet

mkdir -p avh_sup/outputs/logs avh_sup/outputs/ckpts logs

CONFIG=avh_sup/configs/train_config_av1m.yaml

echo "Starting supervised training with config: $CONFIG"
echo ""

cd avh_sup

python train_test.py --config_path configs/train_config_av1m.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints → outputs/ckpts/"
echo "  Logs        → outputs/logs/"
echo "============================================================"
