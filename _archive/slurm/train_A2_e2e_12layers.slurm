#!/bin/bash
#SBATCH --job-name=avh_e2e_A2_12L
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A2_e2e_12layers_%j.out
#SBATCH --error=logs/train_A2_e2e_12layers_%j.err

# ============================================================
#  End-to-end A2 training — AV-HuBERT top-12 layers unfrozen
#
#  Difference from train_A2_e2e.slurm:
#    n_unfreeze_layers: 12 (was 4)
#    batch_size: 8 (was 16)
#    lr_encoder: 5e-6 (was 1e-5)
#    mem: 80G (was 64G)
#    time: 24h (was 12h)
# ============================================================

echo "============================================================"
echo "  End-to-end A2 Training (AV-HuBERT top-12 layers unfrozen)"
echo "  Job : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start: $(date)"
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

AVH_CKPT="/data/projects/punim2637/nnliang/av_hubert/avhubert/self_large_vox_433h.pt"
AVH_URL="https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt"

# ── Step 1: Download AV-HuBERT pretrained checkpoint if missing ───────────────
if [[ ! -f "$AVH_CKPT" ]]; then
    echo "Downloading AV-HuBERT checkpoint..."
    mkdir -p "$(dirname $AVH_CKPT)"
    wget -q --show-progress -O "$AVH_CKPT" "$AVH_URL"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Failed to download AV-HuBERT checkpoint from $AVH_URL"
        exit 1
    fi
    echo "Downloaded: $AVH_CKPT"
else
    echo "AV-HuBERT checkpoint found: $AVH_CKPT"
fi

# ── Step 2: Verify raw data exists ────────────────────────────────────────────
AV1M_RAW="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
FAVC_RAW="/data/scratch/projects/punim2637/nnliang/favc_preprocessed"

if [[ ! -d "$AV1M_RAW" ]]; then
    echo "ERROR: AV1M preprocessed data not found at $AV1M_RAW"
    exit 1
fi
echo "AV1M raw data: OK ($AV1M_RAW)"

# ── Step 3: Run training ──────────────────────────────────────────────────────
mkdir -p logs \
    avh_sup/outputs_A2_e2e_12layers/ckpts \
    avh_sup/outputs_A2_e2e_12layers/logs \
    avh_sup/outputs_A2_e2e_12layers/results

cd avh_sup

echo ""
echo "Starting end-to-end A2 training (12 unfrozen layers)..."
python train_e2e.py --config_path configs/A2_e2e_12layers.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "============================================================"
