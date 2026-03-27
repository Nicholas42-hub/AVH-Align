#!/bin/bash
#SBATCH --job-name=avh_perturb_A0
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/avh_perturb_A0_%j.out
#SBATCH --error=logs/avh_perturb_A0_%j.err

# ============================================================
#  Experiment 3 — Perturbation Robustness
#
#  Goal: Measure cosine similarity between clean and Gaussian-
#        noised Z_c vs Z_s representations at multiple σ levels.
#
#  Expected result (if disentanglement works):
#    sim(Z_c, Z_c') > sim(Z_s, Z_s') at all noise levels
#    → causal branch is more stable under input perturbation
#
#  Model: A0 (orthogonality only — no adversarial)
# ============================================================

echo "============================================================"
echo "  Perturbation Robustness — A0 (orth only)"
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

mkdir -p avh_sup/outputs_A0/results logs

cd avh_sup
python eval_perturbation_robustness.py \
    --ckpt_path      outputs_A0/ckpts/model-epoch=09.ckpt \
    --model          ablation \
    --config         configs/A0.yaml \
    --favc_root      ../data/favc_features \
    --csv_root_path  csv_metadata/favc \
    --split          test \
    --batch_size     32 \
    --seed           42 \
    --output_dir     outputs_A0/results \
    --device         cuda

echo ""
echo "============================================================"
echo "  Perturbation robustness eval done: $(date)"
echo "============================================================"
