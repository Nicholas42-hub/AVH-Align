#!/bin/bash
#SBATCH --job-name=eval_ts_A1_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_ts_A1_e2e_%j.out
#SBATCH --error=logs/eval_ts_A1_e2e_%j.err

echo "============================================================"
echo "  eval_auc_testset_A1_e2e — Causal & Spurious AUC (test split)"
echo "  A1_e2e_dadv on FAVC test_split.csv (balanced real/fake)"
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

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT_DIR="$BASE/avh_sup/outputs_A1_e2e_dadv/ckpts"

BEST=$(ls -t "$CKPT_DIR"/model-epoch=*.ckpt 2>/dev/null | grep -v "last" | head -1)
CKPT="${BEST:-$CKPT_DIR/last.ckpt}"
echo ">>> Using checkpoint: $CKPT"

mkdir -p "$BASE/avh_sup/outputs_A1_e2e_dadv/results" "$BASE/logs"

cd "$BASE/avh_sup"

echo ">>> Running Causal & Spurious AUC eval on FAVC test split..."
$PYTHON eval_auc_a1_e2e.py \
    --ckpt      "$CKPT" \
    --config    "$BASE/avh_sup/configs/A1_e2e_dadv.yaml" \
    --av1m_raw  /scratch/punim2637/nnliang/avd1m_preprocessed/val \
    --av1m_csv  "$BASE/avh_sup/csv_metadata/av1m_e2e" \
    --favc_raw  /scratch/punim2637/nnliang/favc_preprocessed \
    --favc_csv  "$BASE/avh_sup/csv_metadata/favc/test_split.csv" \
    --output_dir "$BASE/avh_sup/outputs_A1_e2e_dadv/results" \
    --max_clips 10000 \
    --batch_size 4 \
    --num_workers 2

echo ""
echo "============================================================"
echo "  AUC eval done: $(date)"
echo "  Results: $BASE/avh_sup/outputs_A1_e2e_dadv/results/eval_e2e_auc_testset.json"
echo "============================================================"
cat "$BASE/avh_sup/outputs_A1_e2e_dadv/results/eval_e2e_auc_testset.json" 2>/dev/null
