#!/bin/bash
#SBATCH --job-name=eval_B1b
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_B1b_%j.out
#SBATCH --error=logs/eval_B1b_%j.err

echo "============================================================"
echo "  B1b Evaluation — FakeAVCeleb (N=1114 balanced subset)"
echo "  Consistent with all other frozen model evaluations"
echo "  Job ID : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
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

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT=$(ls "$BASE/avh_sup/outputs_B1b/ckpts"/model-epoch=*.ckpt | sort | tail -1)
FAVC_ROOT="$BASE/data/favc_features"
CSV_ROOT="csv_metadata/favc"

mkdir -p avh_sup/outputs_B1b/results logs
echo "Using checkpoint: $CKPT"

cd avh_sup

echo ""
echo "=== B1b | FakeAVCeleb test (N=1114) ==="
python eval_favc.py \
    --ckpt          "$CKPT" \
    --model         modal \
    --features_path "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$BASE/avh_sup/outputs_B1b/results" \
    --device        cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
