#!/bin/bash
#SBATCH --job-name=avh_e2e_A2_manip
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A2_e2e_manip_%j.out
#SBATCH --error=logs/train_A2_e2e_manip_%j.err

# ============================================================
#  End-to-end A2+ManipDomain training
#
#  Same as A2_e2e but adds manipulation-type adversarial loss on Z_c:
#    • use_adv=true          GRL on Z_s forces label-invariant Z_s
#    • use_sg=true           cls only from Z_c
#    • use_manip_domain=true GRL on Z_c (4-class: real/rvfa/fvra/fvfa)
#
#  Hypothesis: Z_c invariant to manip type → learns universal fake detector
#              Z_s free to be manip-type-specific
#
#  Data: val_manip.csv (2000 clips) 80/20 split → 1600 train / 400 holdout
# ============================================================

echo "============================================================"
echo "  End-to-end A2+ManipDomain Training"
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

# ── Step 2: Verify raw data and manip CSV ─────────────────────────────────────
AV1M_RAW="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
MANIP_CSV="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m/val_manip.csv"

if [[ ! -d "$AV1M_RAW" ]]; then
    echo "ERROR: AV1M preprocessed data not found at $AV1M_RAW"
    exit 1
fi
if [[ ! -f "$MANIP_CSV" ]]; then
    echo "ERROR: val_manip.csv not found at $MANIP_CSV"
    exit 1
fi
echo "AV1M raw data : OK ($AV1M_RAW)"
echo "Manip CSV     : OK ($MANIP_CSV)"

# ── Step 3: Run training ──────────────────────────────────────────────────────
mkdir -p logs \
    avh_sup/outputs_A2_e2e_manip/ckpts \
    avh_sup/outputs_A2_e2e_manip/logs \
    avh_sup/outputs_A2_e2e_manip/results

cd avh_sup

echo ""
echo "Starting A2+ManipDomain e2e training..."
python train_e2e.py --config_path configs/A2_e2e_manip.yaml

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "============================================================"
