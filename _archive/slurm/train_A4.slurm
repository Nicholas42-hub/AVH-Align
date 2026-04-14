#!/bin/bash
#SBATCH --job-name=train_A4
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_A4_%j.out
#SBATCH --error=logs/train_A4_%j.err

# ============================================================
#  Ablation A4 — A2 + β-VAE / Geometric Constraints
#
#  Components:
#    Component 1: GRL adversarial loss (Z_s non-informative for label)  ✓
#    Component 2: Stop gradient / Z_c-only classification               ✓
#    Component 3: Domain adversarial                                     ✗  (excluded)
#    Component 4: β-VAE variational projections + reconstruction        ✓
#      β_s=10 >> β_c=1 compresses spurious subspace toward N(0,I)
#      Decoder reconstructs [V̄; Ā] from cat(Z_c, Z_s) preventing
#      information collapse.
#
#  To sweep β_s, edit beta_s in configs/A4.yaml and rename --name.
# ============================================================

echo "============================================================"
echo "  AVH-Align Ablation A4 — A2 + β-VAE (no L_dom)"
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

mkdir -p avh_sup/outputs_A4/{logs,ckpts,results} logs

cd avh_sup
python train_test_causal_ablation.py \
    --config_path configs/A4.yaml \
    --mode train

echo ""
echo "============================================================"
echo "  A4 training done: $(date)"
echo "============================================================"
