#!/bin/bash
#SBATCH --job-name=avh_causal_train
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/avh_causal_train_%j.out
#SBATCH --error=logs/avh_causal_train_%j.err

# ============================================================
#  AVH-Align Causal Supervised Training
#
#  Trains the causal-disentangled model (mlp_causal.py) which
#  splits the AV-HuBERT representation into:
#    causal_repr   = concat(visual_proj_causal, audio_proj_causal)
#    spurious_repr = concat(visual_proj_spurious, audio_proj_spurious)
#
#  Three separate heads are trained:
#    full_head     (causal + spurious)
#    causal_head   (causal only)
#    spurious_head (spurious only)
#
#  Evaluation reports v_full / v_causal / v_spurious AUC.
#  Hypothesis: v_causal ≈ v_full  and  v_spurious ≈ 0.5
#
#  Must be run after avh_extract_features.slurm completes.
# ============================================================

echo "============================================================"
echo "  AVH-Align Causal Training"
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
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

FEATURES_DIR=data/avh_features
CSV_DIR=avh_sup/csv_metadata/av1m

# ---- Setup: use val clips as training data ----
echo "Setting up train/val splits from val/ features..."

if [ ! -e "$FEATURES_DIR/train" ]; then
    ln -s /data/projects/punim2637/nnliang/AVH-Align/$FEATURES_DIR/val \
          /data/projects/punim2637/nnliang/AVH-Align/$FEATURES_DIR/train
    echo "Created symlink: $FEATURES_DIR/train -> $FEATURES_DIR/val"
else
    echo "Symlink/dir $FEATURES_DIR/train already exists."
fi

# Use pre-generated train/val CSVs (generated from all available NPZs, 80/20 split)
echo "Using pre-generated CSVs in $CSV_DIR"

TRAIN_CLIPS=$(tail -n +2 $CSV_DIR/train_labels.csv | wc -l)
VAL_CLIPS=$(tail -n +2 $CSV_DIR/val_labels.csv | wc -l)
FEAT_FILES=$(find $FEATURES_DIR/val -name "*.npz" 2>/dev/null | wc -l)

echo ""
echo "Train clips : $TRAIN_CLIPS"
echo "Val clips   : $VAL_CLIPS"
echo "Feature npz : $FEAT_FILES"
echo ""

if [ "$FEAT_FILES" -eq 0 ]; then
    echo "ERROR: No .npz feature files found in $FEATURES_DIR/val/"
    echo "       Run avh_extract_features.slurm and avh_features_train.slurm first!"
    exit 1
fi

# Install lightning if not present
python -c "import lightning" 2>/dev/null || pip install lightning --quiet

mkdir -p avh_sup/outputs_causal/logs avh_sup/outputs_causal/ckpts

# ---- Train ----
echo ""
echo "Starting causal training..."
cd avh_sup
python train_test_causal.py \
    --config_path configs/train_config_av1m_causal.yaml

echo ""
echo "============================================================"
echo "  Causal training done: $(date)"
echo "  Checkpoints: /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_causal/ckpts/"
echo "  Logs:        /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_causal/logs/"
echo "============================================================"
