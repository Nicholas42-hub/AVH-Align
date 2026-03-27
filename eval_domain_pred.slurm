#!/bin/bash
#SBATCH --job-name=avh_domain_pred
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/avh_domain_pred_%j.out
#SBATCH --error=logs/avh_domain_pred_%j.err

# ============================================================
#  Experiment 2 — Domain Predictability Probe
#
#  Goal: Measure how well Z_c vs Z_s can predict dataset identity
#        AV1M (0) vs FakeAVCeleb (1).
#        Binary logistic regression probe, 5-fold stratified CV.
#
#  Expected result (if GRL disentanglement works):
#    Z_c AUC ≈ 0.5   (causal branch is domain-invariant)
#    Z_s AUC >> 0.5  (spurious branch absorbed domain artefacts)
#
#  Runs all three models: A0, A1, A2
# ============================================================

echo "============================================================"
echo "  Domain Predictability Probe — A0 / A1 / A2"
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

mkdir -p avh_sup/outputs_A0/results avh_sup/outputs_A1/results avh_sup/outputs_A2/results logs

cd avh_sup

AV1M_ROOT="../data/avh_features/val"
FAVC_ROOT="../data/favc_features"

for MODEL in A0 A1 A2; do
    # Find best checkpoint (latest by epoch, adjust if needed)
    CKPT=$(ls outputs_${MODEL}/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "  WARNING: no checkpoint found for $MODEL, skipping."
        continue
    fi
    echo ""
    echo "--- Running domain predictability probe for $MODEL ---"
    echo "  Checkpoint: $CKPT"
    python eval_domain_predictability.py \
        --ckpt_path       "$CKPT" \
        --config          configs/${MODEL}.yaml \
        --av1m_root       "$AV1M_ROOT" \
        --favc_root       "$FAVC_ROOT" \
        --max_per_dataset 2000 \
        --n_folds         5 \
        --seed            42 \
        --output_dir      outputs_${MODEL}/results \
        --device          cuda
done

echo ""
echo "============================================================"
echo "  Domain predictability eval done: $(date)"
echo "============================================================"
