#!/bin/bash
#SBATCH --job-name=avh_domain_pred_A3
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/avh_domain_pred_A3_%j.out
#SBATCH --error=logs/avh_domain_pred_A3_%j.err

# ============================================================
#  Experiment 2 — Domain Predictability Probe for A3
#
#  Goal: Verify that the domain adversarial component (Component 3)
#        successfully removes domain information from Z_s.
#        Binary logistic regression probe: AV1M (0) vs FakeAVCeleb (1)
#        5-fold stratified CV.
#
#  Expected result (if domain adversarial is working):
#    Z_s AUC ≈ 0.5   (domain GRL supresses domain info from Z_s)  ← KEY CHECK
#    Z_c AUC >> 0.5  (causal branch freely encodes domain features)
#
#  Compare with A2 results (both branches ~0.997) to show the
#  contribution of the domain adversarial component (A3 vs A2).
#
#  Model: A3 checkpoint (adversarial + stop-gradient + domain-aware)
# ============================================================

echo "============================================================"
echo "  Domain Predictability Probe — A3 (adv + stop-grad + domain)"
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

mkdir -p avh_sup/outputs_A3/results logs

cd avh_sup
python eval_domain_predictability.py \
    --ckpt_path       outputs_A3/ckpts/model-epoch=10.ckpt \
    --config          configs/A3.yaml \
    --av1m_root       ../data/avh_features/val \
    --favc_root       ../data/favc_features \
    --max_per_dataset 2000 \
    --n_folds         5 \
    --seed            42 \
    --output_dir      outputs_A3/results \
    --device          cuda

echo ""
echo "============================================================"
echo "  A3 domain predictability probe done: $(date)"
echo "============================================================"
