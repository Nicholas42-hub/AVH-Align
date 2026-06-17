#!/bin/bash
#SBATCH --job-name=preprocess_avlips
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/preprocess_avlips_%j.out
#SBATCH --error=logs/preprocess_avlips_%j.err

echo "============================================================"
echo "  AVLips Preprocessing — Mouth ROI Extraction"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load FFmpeg/4.4.2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVHUB_DIR="$BASE/av_hubert/avhubert"
export PYTHONPATH="$AVHUB_DIR"
AVLIPS_RAW="/data/projects/punim2637/nnliang/Datasets/AVLips/AVLips"
SAVE_PATH="$BASE/data/avlips_preprocessed"

mkdir -p "$SAVE_PATH" "$BASE/logs"

echo ">>> Input  : $AVLIPS_RAW/{0_real,1_fake}/"
echo ">>> Output : $SAVE_PATH/{0_real,1_fake}/*_roi.mp4"
echo ">>> Workers: 8"
echo ""

# Must run from av_hubert/avhubert/ so that `preparation` module is on the path
cd "$AVHUB_DIR"

$PYTHON "$BASE/deepfake_preprocess.py" \
    --dataset   AVLips \
    --data_path "$AVLIPS_RAW" \
    --save_path "$SAVE_PATH" \
    --max_workers 8

echo ""
echo "============================================================"
echo "  Preprocessing done: $(date)"
echo "  Real ROIs : $(find $SAVE_PATH/0_real -name '*_roi.mp4' 2>/dev/null | wc -l)"
echo "  Fake ROIs : $(find $SAVE_PATH/1_fake -name '*_roi.mp4' 2>/dev/null | wc -l)"
echo "============================================================"
