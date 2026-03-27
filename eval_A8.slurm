#!/bin/bash
#SBATCH --job-name=eval_A8
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_A8_%j.out
#SBATCH --error=logs/eval_A8_%j.err

echo "============================================================"
echo "  A8 Evaluation — SAD: Synchrony-Appearance Decomposition"
echo "  FakeAVCeleb (trimmed + non-trimmed)"
echo "  + Per-category perturbation robustness"
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

mkdir -p avh_sup/outputs_A8/results/trimmed avh_sup/outputs_A8/results logs

cd avh_sup

CKPT=$(ls outputs_A8/ckpts/*.ckpt | sort | tail -1)
echo "Using checkpoint: $CKPT"

FAVC_ROOT="../data/favc_features"
FAVC_TRIMMED="../data/favc_features_trimmed"
CSV_ROOT="csv_metadata/favc"

echo ""
echo "=== A8 | FakeAVCeleb non-trimmed ==="
$PYTHON eval_favc.py \
    --ckpt          "$CKPT" \
    --model         sad_a8 \
    --features_path "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    outputs_A8/results \
    --device        cuda

echo ""
echo "=== A8 | FakeAVCeleb trimmed ==="
$PYTHON eval_favc.py \
    --ckpt          "$CKPT" \
    --model         sad_a8 \
    --features_path "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    outputs_A8/results/trimmed \
    --device        cuda

echo ""
echo "=== A8 | Perturbation — full test set ==="
$PYTHON eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         sad_a8 \
    --config        configs/A8.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A8/results \
    --device        cuda

echo ""
echo "=== A8 | Perturbation — RealVideo-FakeAudio ==="
$PYTHON eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         sad_a8 \
    --config        configs/A8.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A8/results \
    --device        cuda

echo ""
echo "=== A8 | Perturbation — FakeVideo-RealAudio ==="
$PYTHON eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         sad_a8 \
    --config        configs/A8.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A8/results \
    --device        cuda

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
