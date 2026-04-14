#!/bin/bash
#SBATCH --job-name=train_A37_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_A37_e2e_%j.out
#SBATCH --error=logs/train_A37_e2e_%j.err

echo "============================================================"
echo "  A37_e2e — Label-Aware Domain Adversarial on z_sync"
echo "  Conditional GRL: concat(GRL(z_sync_pool), y_hat_prob)"
echo "  No subspace split — simpler than A36, tests if cond-GRL alone works"
echo "  Curriculum λ_grl: 0.0 → 0.5 (ramps ep 3-9)"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

BASE=/data/projects/punim2637/nnliang/AVH-Align
mkdir -p "$BASE/avh_sup/outputs_A37_e2e/ckpts" \
         "$BASE/avh_sup/outputs_A37_e2e/logs"  \
         "$BASE/avh_sup/outputs_A37_e2e/results" \
         "$BASE/logs"

cd "$BASE/avh_sup"

echo ">>> Starting A37 training..."
srun $PYTHON train_e2e_a37.py \
    --config_path "$BASE/avh_sup/configs/A37_e2e.yaml"

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Best ckpt in: $BASE/avh_sup/outputs_A37_e2e/ckpts/"
echo "============================================================"
