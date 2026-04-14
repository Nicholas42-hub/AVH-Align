#!/bin/bash
#SBATCH --job-name=train_A36_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_A36_e2e_%j.out
#SBATCH --error=logs/train_A36_e2e_%j.err

echo "============================================================"
echo "  A36_e2e — Conditional GRL + Subspace Split + Class-Cond Alignment"
echo "  z_sync → z_c_inv (384d, conditional GRL) + z_c_aux (128d, attractor)"
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
mkdir -p "$BASE/avh_sup/outputs_A36_e2e/ckpts" \
         "$BASE/avh_sup/outputs_A36_e2e/logs"  \
         "$BASE/avh_sup/outputs_A36_e2e/results" \
         "$BASE/logs"

cd "$BASE/avh_sup"

echo ">>> Starting A36 training..."
srun $PYTHON train_e2e_a36.py \
    --config_path "$BASE/avh_sup/configs/A36_e2e.yaml"

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Best ckpt in: $BASE/avh_sup/outputs_A36_e2e/ckpts/"
echo "============================================================"
