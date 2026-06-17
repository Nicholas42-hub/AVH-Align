#!/bin/bash
#SBATCH --job-name=ts_unsup_s4445
#SBATCH --partition=gpu-a100-mig,gpu-l40s-preempt,gpu-l40s,gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --array=0-1
#SBATCH --output=logs/ts_unsup_trimmed_%A_%a.out
#SBATCH --error=logs/ts_unsup_trimmed_%A_%a.err

# ============================================================
#  Table 8 row: TriRoute unsup (Vox2) — seed 44 and 45 best ckpts
#  Each runs untrimmed + trimmed (FAVC + AV1M) — 4 evals per seed
#  array_0 = seed 44, array_1 = seed 45
# ============================================================

SEEDS=(44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
CKPT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed${SEED}/triroute_unsup_sync_best_auc.pt"

UNTRIMMED_OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed${SEED}/results_table8_untrimmed"
TRIMMED_OUT="$BASE/avh_sup/outputs_triroute_unsup_sync_vox2_repro_seed${SEED}/results_table8_trimmed"

FAVC_TRIM="$BASE/data/favc_features_trimmed"
AV1M_TRIM="/scratch/punim2637/nnliang/avd1m_features_trimmed"

echo "============================================================"
echo "  Table 8 unsup eval | seed=${SEED} | $(date)"
echo "  Ckpt: $CKPT"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$UNTRIMMED_OUT" "$TRIMMED_OUT"

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

cd avh_sup

echo "Pass 1/2 : UNTRIMMED"
$PYTHON train_eval_triroute_unsup_sync.py \
    --eval_only \
    --ckpt "$CKPT" \
    --output_dir "$UNTRIMMED_OUT" \
    --num_workers 4

echo "Pass 2/2 : TRIMMED"
$PYTHON train_eval_triroute_unsup_sync.py \
    --eval_only \
    --ckpt "$CKPT" \
    --output_dir "$TRIMMED_OUT" \
    --favc_features "$FAVC_TRIM" \
    --av1m_features "$AV1M_TRIM" \
    --num_workers 4

echo "Done seed=$SEED at $(date)"
