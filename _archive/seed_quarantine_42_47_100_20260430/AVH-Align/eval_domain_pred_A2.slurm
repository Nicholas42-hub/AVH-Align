#!/bin/bash
#SBATCH --job-name=avh_domain_pred_A2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/avh_domain_pred_A2_%j.out
#SBATCH --error=logs/avh_domain_pred_A2_%j.err

# ============================================================
#  Experiment 2 — Audio Manipulation Predictability Probe
#
#  Goal: Measure how well Z_c vs Z_s can distinguish audio
#        manipulation within FakeVideo clips only:
#          FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)
#        Binary logistic regression probe, 5-fold stratified CV.
#
#  Probe design: FakeVideo-* only (same video manipulation).
#  RealVideo-FakeAudio excluded — confounded at input level.
#
#  Expected result (if disentanglement works):
#    Z_s AUC >> 0.5  (spurious branch sensitive to audio type)
#    Z_c AUC ≈ 0.5   (causal/sync branch blind to audio swap)
#
#  Model: A2 checkpoint (adversarial + stop-gradient)
# ============================================================

echo "============================================================"
echo "  Audio Manipulation Predictability Probe — A2 (adv + stop-grad)"
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

mkdir -p avh_sup/outputs_A2/results logs

cd avh_sup
python eval_domain_predictability.py \
    --ckpt_path       outputs_A2/ckpts/model-epoch=09.ckpt \
    --config          configs/A2.yaml \
    --av1m_root       ../data/avh_features/val \
    --favc_root       ../data/favc_features \
    --n_folds    5 \
    --seed       42 \
    --output_dir outputs_A2/results \
    --device     cuda

echo ""
echo "============================================================"
echo "  Audio manipulation predictability eval done: $(date)"
echo "============================================================"
