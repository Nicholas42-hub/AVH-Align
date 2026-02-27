#!/bin/bash
#SBATCH --job-name=avh_setup_env
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --chdir=/data/projects/punim2637/nnliang/AVH-Align
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/setup_env_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/setup_env_%j.err

# ============================================================
#  Create a clean conda env for the full AVH-Align pipeline
#  Uses absolute paths throughout — no reliance on conda activate
# ============================================================

ENV_PREFIX=/data/projects/punim2637/nnliang/avh_env
FAIRSEQ_SRC=/data/projects/punim2637/nnliang/CausalAVD/av_hubert/fairseq
PIP=$ENV_PREFIX/bin/pip
PY=$ENV_PREFIX/bin/python

echo "============================================================"
echo "  AVH-Align Environment Setup"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "  Env    : $ENV_PREFIX"
echo "============================================================"

mkdir -p /data/projects/punim2637/nnliang/AVH-Align/logs
export PYTHONNOUSERSITE=1

# ---- Redirect tmp to /data to avoid home quota -------------
export TMPDIR=/data/projects/punim2637/nnliang/.tmp
export PIP_CACHE_DIR=/data/projects/punim2637/nnliang/.pip_cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# ---- Load Miniconda ----------------------------------------
module purge
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"

# Store conda package cache on /data (not home)
export CONDA_PKGS_DIRS=/data/projects/punim2637/nnliang/.conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"

# ---- Create env with Python 3.9 (skip if already exists) --
if [ ! -f "$PY" ]; then
    echo "Creating new conda env (Python 3.9) ..."
    conda create --prefix "$ENV_PREFIX" python=3.9 -y
else
    echo "Env already exists at $ENV_PREFIX, skipping creation (torch+dlib present)"
fi

echo "Env python: $($PY --version)"

# ---- Upgrade pip (use absolute path) -----------------------
$PY -m pip install --upgrade pip

# ---- Install dlib via conda-forge (pre-compiled binary) ----
echo ""
if $PY -c "import dlib" 2>/dev/null; then
    echo "dlib already installed, skipping."
else
    echo "Installing dlib (conda-forge binary) ..."
    conda install --prefix "$ENV_PREFIX" -c conda-forge dlib -y
fi

# ---- Install PyTorch 2.2 + CUDA 12.1 -----------------------
echo ""
if $PY -c "import torch; assert torch.__version__.startswith('2.2')" 2>/dev/null; then
    echo "PyTorch 2.2 already installed, skipping."
else
    echo "Installing PyTorch 2.2.0 (cu121) ..."
    $PIP install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# ---- Install remaining Python packages ---------------------
echo ""
echo "Installing remaining packages ..."
$PIP install \
    numpy \
    "opencv-python==4.5.4.60" \
    tqdm \
    librosa \
    python-speech-features \
    sk-video \
    scikit-learn \
    pandas \
    soundfile \
    "scipy==1.10.0" \
    sentencepiece \
    editdistance \
    "omegaconf>=2.1" \
    "hydra-core>=1.1" \
    iopath \
    submitit

# ---- Install fairseq (--no-build-isolation avoids /tmp quota) ---
echo ""
echo "Installing fairseq from $FAIRSEQ_SRC ..."
$PIP install --no-build-isolation "$FAIRSEQ_SRC"

# ---- Verify key imports ------------------------------------
echo ""
echo "Verifying imports ..."
$PY -c "import torch; print(f'torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
$PY -c "import dlib; print(f'dlib {dlib.__version__}')"
$PY -c "import cv2; print(f'opencv {cv2.__version__}')"
$PY -c "import librosa; print(f'librosa {librosa.__version__}')"
$PY -c "import skvideo; print('scikit-video OK')"
$PY -c "from fairseq import checkpoint_utils; print('fairseq OK')"
$PY -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
$PY -c "import pandas; print(f'pandas {pandas.__version__}')"

echo ""
echo "============================================================"
echo "  Setup complete: $(date)"
echo "  Env ready at: $ENV_PREFIX"
echo "============================================================"
