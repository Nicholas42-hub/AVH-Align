#!/bin/bash
#SBATCH --job-name=favc_preprocess
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/favc_preprocess_%j.out
#SBATCH --error=logs/favc_preprocess_%j.err

# ============================================================
#  Step 1: Preprocess FakeAVCeleb videos for AVH-Align
#
#  - Unzips FakeAVCeleb_v1.2.zip
#  - Generates favc_metadata.csv in type,path,filename format
#  - Runs deepfake_preprocess.py --dataset FakeAVCeleb
#
#  Outputs ROI crops + wav files to data/favc_preprocessed/
#
#  Usage:
#    sbatch favc_preprocess.slurm
#  Prerequisite:
#    Datasets/FakeCeleb/FakeAVCeleb_v1.2.zip must exist
# ============================================================

echo "============================================================"
echo "  FakeAVCeleb Preprocess"
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
echo "Python: $(which python) — $(python --version)"

# Patch ffmpeg path
FFMPEG=$(which ffmpeg)
sed -i "s|/usr/bin/ffmpeg|${FFMPEG}|g" av_hubert/avhubert/deepfake_preprocess.py \
    2>/dev/null || true

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

ZIP_FILE=/data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2.zip
FAVC_ROOT=/data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2
METADATA=av1m_metadata/favc_metadata.csv
SAVE_PATH=data/favc_preprocessed

# ── Step 1: Verify dataset ────────────────────────────────────
if [ ! -d "$FAVC_ROOT" ] || [ "$(find $FAVC_ROOT -name '*.mp4' | wc -l)" -lt 20000 ]; then
    echo "ERROR: FakeAVCeleb dataset not found at $FAVC_ROOT"
    echo "       Expected 21,544 mp4 files. Found: $(find $FAVC_ROOT -name '*.mp4' 2>/dev/null | wc -l)"
    exit 1
fi

TOTAL_VIDS=$(find "$FAVC_ROOT" -name "*.mp4" | wc -l)
echo "Total mp4 files in FakeAVCeleb: $TOTAL_VIDS  (in $FAVC_ROOT)"

# ── Step 2: Generate metadata CSV ─────────────────────────────
echo ""
echo "Generating metadata CSV ..."
python generate_favc_metadata.py \
    --split all \
    --csv_dir avh_sup/csv_metadata/favc \
    --out     "$METADATA"
echo "Metadata rows: $(tail -n +2 $METADATA | wc -l)"

# ── Step 3: Preprocess (face ROI crop + wav extraction) ────────
echo ""
echo "Starting preprocessing ..."
mkdir -p "$SAVE_PATH" logs

cd av_hubert/avhubert

python deepfake_preprocess.py \
    --dataset    FakeAVCeleb \
    --category   all \
    --metadata   /data/projects/punim2637/nnliang/AVH-Align/$METADATA \
    --data_path  "$FAVC_ROOT" \
    --save_path  /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH \
    --max_workers 16

echo ""
echo "============================================================"
echo "  Preprocessing done: $(date)"
ROI=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH -name "*_roi.mp4" 2>/dev/null | wc -l)
WAV=$(find /data/projects/punim2637/nnliang/AVH-Align/$SAVE_PATH -name "*.wav"    2>/dev/null | wc -l)
echo "  ROI videos : $ROI"
echo "  WAV files  : $WAV"
echo "============================================================"
