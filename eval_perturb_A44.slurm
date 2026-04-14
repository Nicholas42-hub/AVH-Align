#!/bin/bash
#SBATCH --job-name=perturb_A44
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/perturb_A44_%j.out
#SBATCH --error=logs/perturb_A44_%j.err

echo "============================================================"
echo "  A44 Perturbation Robustness"
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
CKPT=$(ls "$BASE/avh_sup/outputs_A44/ckpts"/model-epoch=*.ckpt | sort | tail -1)
FAVC_ROOT="$BASE/data/favc_features"
CSV_ROOT="csv_metadata/favc"

mkdir -p avh_sup/outputs_A44/results logs
echo "Using checkpoint: $CKPT"

cd avh_sup

echo ""
echo "=== A44 | Perturbation — full test set ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         causal_a44 \
    --config        configs/A44.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A44/results \
    --device        cuda

echo ""
echo "=== A44 | Perturbation — FakeVideo-RealAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         causal_a44 \
    --config        configs/A44.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A44/results \
    --device        cuda

echo ""
echo "=== A44 | Perturbation — RealVideo-FakeAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         causal_a44 \
    --config        configs/A44.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A44/results \
    --device        cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
