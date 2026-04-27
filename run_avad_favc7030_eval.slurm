#!/bin/bash
#SBATCH --job-name=avad_favc7030
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/avad_favc7030_%j.out
#SBATCH --error=logs/avad_favc7030_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVAD_DIR="$BASE/external/audio-visual-forensics"
PROTO_DIR="$BASE/submission_prep/avad_matched_baseline"

cd "$BASE"
mkdir -p logs external "$PROTO_DIR"

module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"

# Use a separate environment if available. Otherwise create it once.
if conda env list | awk '{print $1}' | grep -qx "avad_env"; then
    conda activate avad_env
elif [ -d /data/projects/punim2637/nnliang/avad_env ]; then
    conda activate /data/projects/punim2637/nnliang/avad_env
else
    conda create -y -n avad_env python=3.9
    conda activate avad_env
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

if [ ! -d "$AVAD_DIR/.git" ]; then
    git clone https://github.com/cfeng16/audio-visual-forensics.git "$AVAD_DIR"
fi

python -m pip install --upgrade pip
python -m pip install \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html
conda install -y -c conda-forge av=9.0.1 ffmpeg pkg-config
grep -vE '^(torch|torchvision|torchaudio|av|ffmpeg)==' "$AVAD_DIR/requirements.txt" > "$PROTO_DIR/avad_requirements_no_torch.txt"
python -m pip install -r "$PROTO_DIR/avad_requirements_no_torch.txt"
python -m pip install "numba==0.56.4" "llvmlite==0.39.1"
python -m pip install scikit-learn tensorboard

python "$BASE/submission_prep/prepare_avad_protocol.py" prepare \
    --favc_csv "$BASE/avh_sup/csv_metadata/favc_7030_canonical/test_split.csv" \
    --favc_root /data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2 \
    --output_dir "$PROTO_DIR"

cd "$AVAD_DIR"

# AVAD requires sync_model.pth; the official README links it from Google Drive.
# Put it in $AVAD_DIR/sync_model.pth before running if the automatic download is unavailable.
if [ ! -f sync_model.pth ]; then
    echo "Missing $AVAD_DIR/sync_model.pth."
    echo "Download it from the AVAD README link, then rerun this job."
    exit 2
fi

python detect.py \
    --test_video_path "$PROTO_DIR/favc_7030_test_videos.txt" \
    --device cuda:0 \
    --max-len 50 \
    --n_workers 4 \
    --bs 1 \
    --lam 0 \
    --output_dir "$AVAD_DIR/outputs_favc_7030"

cd "$BASE"
python submission_prep/prepare_avad_protocol.py evaluate \
    --labels_csv "$PROTO_DIR/favc_7030_test_labels.csv" \
    --scores_npy "$AVAD_DIR/outputs_favc_7030/testing_scores.npy" \
    --output_json "$PROTO_DIR/favc_7030_results.json"
