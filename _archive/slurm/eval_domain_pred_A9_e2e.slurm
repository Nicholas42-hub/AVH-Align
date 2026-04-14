#!/bin/bash
#SBATCH --job-name=domain_A9_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/domain_A9_e2e_%j.out
#SBATCH --error=logs/domain_A9_e2e_%j.err

# ============================================================
#  Domain Predictability Probe — A9_e2e (SAD + MMD, fully E2E)
#
#  Target outcomes:
#    Z_sync domain AUC ≈ 0.5   (MMD + unfrozen encoder expelled domain)
#    Z_app  domain AUC >> 0.5  (domain concentrated in appearance branch)
#
#  Frozen baselines:
#    Raw AV-HuBERT     AUC ≈ 0.999
#    Frozen A8 (GRL)   AUC ≈ 0.986  Δraw = −0.014
#    Frozen A9 (MMD)   AUC ≈ 0.988  Δraw = −0.011
#    A9_e2e (this)     AUC = ???    target: ≪ 0.9
# ============================================================

echo "============================================================"
echo "  Domain Predictability Probe — A9_e2e (SAD + MMD, E2E)"
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

mkdir -p avh_sup/outputs_A9_e2e/results logs

cd avh_sup

# Auto-select best checkpoint (highest val_auc_causal, exclude 'last')
CKPT=$(ls outputs_A9_e2e/ckpts/*.ckpt 2>/dev/null | grep -v last | sort | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in outputs_A9_e2e/ckpts/"
    exit 1
fi
echo "Using checkpoint: $CKPT"

AV1M_RAW="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
FAVC_RAW="/data/scratch/projects/punim2637/nnliang/favc_preprocessed"

$PYTHON eval_e2e_domain_a9.py \
    --ckpt        "$CKPT" \
    --config      "configs/A9_e2e.yaml" \
    --av1m_raw    "$AV1M_RAW" \
    --favc_raw    "$FAVC_RAW" \
    --av1m_csv    "csv_metadata/av1m" \
    --output_dir  "outputs_A9_e2e/results" \
    --max_clips   500 \
    --batch_size  8 \
    --num_workers 2

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
