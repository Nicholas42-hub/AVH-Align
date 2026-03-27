#!/bin/bash
#SBATCH --job-name=train_A8
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_A8_%j.out
#SBATCH --error=logs/train_A8_%j.err

# ============================================================
#  A8 — SAD: Synchrony-Appearance Decomposition
#
#  Full architectural redesign replacing FCD (A5–A7) with a
#  task-aware decomposition:
#
#    Z_sync (causal):   cat(V, A, V⊙A, V−A) → projection
#      • V⊙A and V−A are algebraic signatures of AV coherence
#      • Disrupted by all deepfake manipulations (V/A mismatch)
#
#    Z_app (spurious):  V-only + A-only appearance encoders
#      • Structurally cannot encode cross-modal coherence
#      • Captures identity, recording conditions, speaker style
#
#  Push-pull losses:
#    L_task  — Z_sync must detect fake/real
#    L_orth  — Z_sync ⊥ Z_app per frame
#    L_dadv  — Z_sync must NOT encode domain (GRL)
#    L_ddis  — Z_app must encode domain
#    L_ladv  — Z_app must NOT predict label (GRL)
#
#  This fixes the fundamental A5–A7 failure: FCD's redundant(V,A)
#  = AV coherence = the actual manipulation cue, so it could never
#  be the "spurious" branch regardless of adversarial pressure.
# ============================================================

echo "============================================================"
echo "  A8 Training — SAD: Synchrony-Appearance Decomposition"
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

mkdir -p avh_sup/outputs_A8/{logs,ckpts,results} logs

cd avh_sup

$PYTHON train_test_sad_a8.py --config_path configs/A8.yaml --mode train

echo ""
echo "============================================================"
echo "  Training done: $(date)"
echo "  Checkpoints:"
ls -lh outputs_A8/ckpts/
echo "============================================================"
