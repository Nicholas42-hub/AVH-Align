#!/bin/bash
#SBATCH --job-name=domain_pred_A5
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/domain_pred_A5_%j.out
#SBATCH --error=logs/domain_pred_A5_%j.err

# ============================================================
#  Domain Predictability Probe — A5 (FCD)
#
#  Probes whether Z_causal and Z_redundant encode domain identity
#  (AV1M=0 vs FakeAVCeleb=1).
#
#  Expected results if FCD works:
#    Z_causal   domain AUC ≈ 0.5   (domain-invariant: doesn't know which dataset)
#    Z_redundant domain AUC >> 0.5  (redundant branch absorbed domain shortcuts)
#
#  Note: raw AV-HuBERT features are ~99% domain-separable (baseline).
#  The meaningful signal is how much each branch reduces domain AUC vs raw.
# ============================================================

echo "============================================================"
echo "  Domain Predictability Probe — A5 (FCD)"
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

mkdir -p avh_sup/outputs_A5/results logs

cd avh_sup

CKPT=$(ls outputs_A5/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint found in outputs_A5/ckpts/ — has training finished?"
    exit 1
fi
echo "Using checkpoint: $CKPT"

python eval_domain_predictability.py \
    --ckpt_path         "$CKPT" \
    --config            configs/A5.yaml \
    --av1m_root         ../data/avh_features/val \
    --favc_root         ../data/favc_features \
    --max_per_dataset   2000 \
    --output_dir        outputs_A5/results \
    --device            cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
