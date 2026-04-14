#!/bin/bash
#SBATCH --job-name=eval_A42
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_A42_%j.out
#SBATCH --error=logs/eval_A42_%j.err

echo "============================================================"
echo "  A42 Evaluation — FakeAVCeleb"
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

mkdir -p avh_sup/outputs_A42/results logs

# Auto-detect best checkpoint
BEST=$(ls "$BASE/avh_sup/outputs_A42/ckpts"/model-epoch=*.ckpt 2>/dev/null | sort | tail -1)
CKPT="${BEST:-$BASE/avh_sup/outputs_A42/ckpts/last.ckpt}"
echo "Using checkpoint: $CKPT"

FAVC_ROOT="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc"

cd "$BASE/avh_sup"

echo ""
echo "=== A42 | FakeAVCeleb test ==="
$PYTHON eval_favc.py \
    --ckpt          "$CKPT" \
    --model         causal_a42 \
    --features_path "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$BASE/avh_sup/outputs_A42/results" \
    --device        cuda

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
