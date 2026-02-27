#!/bin/bash
# ============================================================
#  One-time setup for AVH-Align reproduction
#
#  Run this ONCE from a login node before submitting SLURM jobs:
#    bash setup_avh_align.sh
# ============================================================

set -e
cd /data/projects/punim2637/nnliang/AVH-Align
echo "Working in: $PWD"

# ── 1. Link CausalAVD's av_hubert (already cloned + fairseq installed) ────────
if [ ! -L "av_hubert" ] && [ ! -d "av_hubert" ]; then
    echo "Symlinking av_hubert from CausalAVD ..."
    ln -s /data/projects/punim2637/nnliang/CausalAVD/av_hubert av_hubert
    echo "  Done."
else
    echo "  av_hubert already present — skipping."
fi

AVHUBERT_DIR=av_hubert/avhubert

# ── 2. Copy AVH-Align's custom scripts into av_hubert/avhubert ────────────────
echo "Copying deepfake_preprocess.py and deepfake_feature_extraction.py ..."
cp deepfake_preprocess.py  "$AVHUBERT_DIR/deepfake_preprocess.py"
cp deepfake_feature_extraction.py "$AVHUBERT_DIR/deepfake_feature_extraction.py"
echo "  Done."

# ── 3. Create content/data/misc directory for landmark models ─────────────────
mkdir -p "$AVHUBERT_DIR/content/data/misc"

# shape predictor (dlib 68-point model)
PREDICTOR="$AVHUBERT_DIR/content/data/misc/shape_predictor_68_face_landmarks.dat"
if [ ! -f "$PREDICTOR" ]; then
    echo "Downloading shape_predictor_68_face_landmarks.dat ..."
    wget -q "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" \
         -O "${PREDICTOR}.bz2"
    bzip2 -d "${PREDICTOR}.bz2"
    echo "  Done."
else
    echo "  shape_predictor already present — skipping."
fi

# 20words mean face
MEAN_FACE="$AVHUBERT_DIR/content/data/misc/20words_mean_face.npy"
if [ ! -f "$MEAN_FACE" ]; then
    echo "Downloading 20words_mean_face.npy ..."
    wget -q \
      "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy" \
      -O "$MEAN_FACE"
    echo "  Done."
else
    echo "  mean_face already present — skipping."
fi

# ── 4. Download AVH-Align's AV-HuBERT checkpoint ─────────────────────────────
CKPT="$AVHUBERT_DIR/self_large_vox_433h.pt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading self_large_vox_433h.pt (AVH-Align's AV-HuBERT checkpoint) ..."
    wget -q \
      "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt" \
      -O "$CKPT"
    echo "  Done. Size: $(du -sh "$CKPT" | cut -f1)"
else
    echo "  self_large_vox_433h.pt already present — skipping."
fi

# ── 5. Create a patched deepfake_preprocess.py with dynamic ffmpeg path ───────
# The original script hardcodes /usr/bin/ffmpeg — patch it for cluster use
PATCH="$AVHUBERT_DIR/deepfake_preprocess.py"
if grep -q "/usr/bin/ffmpeg" "$PATCH"; then
    echo "Patching ffmpeg path in deepfake_preprocess.py ..."
    FFMPEG_PATH=$(which ffmpeg 2>/dev/null || echo "/usr/bin/ffmpeg")
    sed -i "s|/usr/bin/ffmpeg|${FFMPEG_PATH}|g" "$PATCH"
    echo "  Patched to: $FFMPEG_PATH"
fi

# ── 6. Output directories ─────────────────────────────────────────────────────
mkdir -p data/avh_preprocessed/train
mkdir -p data/avh_preprocessed/val
mkdir -p data/avh_features/train
mkdir -p data/avh_features/val
mkdir -p logs

echo ""
echo "============================================================"
echo "  AVH-Align setup complete!"
echo ""
echo "  Next steps (submit in order):"
echo "  1. sbatch avh_preprocess.slurm        # preprocess AV1M val (test set)"
echo "  2. sbatch avh_extract_features.slurm  # extract AV-HuBERT features"
echo "  3. sbatch avh_eval.slurm              # evaluate pretrained model"
echo "============================================================"
