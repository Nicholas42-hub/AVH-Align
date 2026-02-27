#!/bin/bash
#SBATCH --job-name=avh_val_full
#SBATCH --partition=cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/avh_preprocess_val_full_%j.out
#SBATCH --error=logs/avh_preprocess_val_full_%j.err

# ============================================================
#  Preprocess ALL AV-Deepfake1M VALIDATION videos (57,340)
#
#  Uses val_metadata_full.csv (all 57,340 clips)
#  SPLIT=val to read from data_path/val/
#  and save to avh_preprocessed/val/
# ============================================================

echo "============================================================"
echo "  AVH-Align Preprocess - FULL VALIDATION SET (57,340 videos)"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

module purge
module load GCCcore/11.3.0
module load FFmpeg/4.4.2
module load CMake/3.24.3
module load dlib/19.24.2

# Use conda environment
source /data/projects/punim2637/nnliang/avh_env/etc/profile.d/conda.sh
conda activate /data/projects/punim2637/nnliang/avh_env

echo "Python: $(which python) — $(python --version)"
echo "FFmpeg: $(which ffmpeg)"

# Set environment
export PATH=/data/projects/punim2637/nnliang/avh_env/bin:$PATH
export PYTHONPATH=/data/projects/punim2637/nnliang/avh_env/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Update FFmpeg path in preprocessing script
FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py 2>/dev/null || true

# Use full validation metadata (57,340 clips)
METADATA=av1m_metadata/val_metadata_full.csv
DATA_PATH=/data/projects/punim2637/nnliang/Datasets/AVDeepfake1M
SAVE_PATH=data/avh_preprocessed

echo "Metadata   : $METADATA"
echo "Data path  : $DATA_PATH"
echo "Save path  : $SAVE_PATH"
echo "Clips in metadata: $(tail -n +2 $METADATA | wc -l)"
echo ""

mkdir -p "$SAVE_PATH" logs

cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset    AV1M \
    --split      val \
    --metadata   /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path  "$DATA_PATH" \
    --save_path  /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH \
    --max_workers 32

echo ""
echo "============================================================"
echo "  Preprocessing done: $(date)"
DONE=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH/val -name "*_roi.mp4" 2>/dev/null | wc -l)
echo "  Total ROI videos ready in val/: $DONE"
echo "============================================================"
