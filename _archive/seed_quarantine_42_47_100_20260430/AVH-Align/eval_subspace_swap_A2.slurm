#!/bin/bash
#SBATCH --job-name=swap_A2
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32G --time=2:00:00
#SBATCH --output=logs/eval_subspace_swap_A2_%j.out
#SBATCH --error=logs/eval_subspace_swap_A2_%j.err

echo "============================================================"
echo "  Subspace Swap — A2 (binary causal)"
echo "  Job ID : $SLURM_JOB_ID    Node: $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

BASE=/data/projects/punim2637/nnliang/AVH-Align
cd "$BASE"

module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p "$BASE/avh_sup/outputs_A2/results" "$BASE/logs"
CKPT="$BASE/avh_sup/outputs_A2/ckpts/model-epoch=09.ckpt"
echo ">>> Using checkpoint: $CKPT"

cd "$BASE/avh_sup"
$PYTHON eval_subspace_swap.py \
    --ckpt_path  "$CKPT" \
    --config     "$BASE/avh_sup/configs/A2.yaml" \
    --favc_root  "$BASE/data/favc_features" \
    --n_donors   30 \
    --max_per_cat 200 \
    --output_dir "$BASE/avh_sup/outputs_A2/results" \
    --seed       42 \
    --device     cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
python3 -c "import json; d=json.load(open('$BASE/avh_sup/outputs_A2/results/subspace_swap.json')); \
  [print(f'{sp}: {list(v[\"swap\"].keys()[:4])}') for sp,v in d['subspaces'].items()]" 2>/dev/null
