#!/bin/bash
#SBATCH --job-name=extract_feat_avlips
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/extract_feat_avlips_%j.out
#SBATCH --error=logs/extract_feat_avlips_%j.err

echo "============================================================"
echo "  AVLips Feature Extraction — AV-HuBERT"
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

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVHUB_DIR="$BASE/av_hubert/avhubert"
# PYTHONPATH not needed: deepfake_feature_extraction.py adds av_hubert parent itself
AVLIPS_BASE="/data/projects/punim2637/nnliang/Datasets/AVLips/AVLips"
PREP_PATH="$BASE/data/avlips_preprocessed"   # output of preprocess_avlips.slurm
SAVE_PATH="$BASE/data/avlips_features"
CKPT_PATH="$AVHUB_DIR/self_large_vox_433h.pt"

mkdir -p "$SAVE_PATH" "$BASE/logs"

echo ">>> Preprocessed ROIs : $PREP_PATH/{0_real,1_fake}/"
echo ">>> WAV files         : $AVLIPS_BASE/wav/{0_real,1_fake}/"
echo ">>> Output NPZ        : $SAVE_PATH/{0_real,1_fake}/"
echo ">>> AV-HuBERT ckpt   : $CKPT_PATH"
echo ""

cd "$BASE"

$PYTHON "$BASE/deepfake_feature_extraction.py" \
    --dataset   AVLips \
    --ckpt_path "$CKPT_PATH" \
    --data_path "$PREP_PATH" \
    --wav_path  "$AVLIPS_BASE/wav" \
    --save_path "$SAVE_PATH"

echo ""
echo "============================================================"
echo "  Feature extraction done: $(date)"
echo "  Real NPZ : $(find $SAVE_PATH/0_real -name '*.npz' 2>/dev/null | wc -l)"
echo "  Fake NPZ : $(find $SAVE_PATH/1_fake -name '*.npz' 2>/dev/null | wc -l)"
echo "============================================================"
