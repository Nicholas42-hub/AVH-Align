#!/bin/bash
#SBATCH --job-name=perturb_A1_new
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/perturb_A1_new_%j.out
#SBATCH --error=logs/perturb_A1_new_%j.err

echo "============================================================"
echo "  Perturbation Robustness — A1 (retrained, no linear probe)"
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

mkdir -p avh_sup/outputs_A1/results logs

cd avh_sup

FAVC_ROOT="../data/favc_features"
CSV_ROOT="csv_metadata/favc"

CKPT=$(ls outputs_A1/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint found in outputs_A1/ckpts/, aborting."
    exit 1
fi

echo "--- Running perturbation robustness for A1 (retrained) ---"
echo "  Checkpoint: $CKPT"
python eval_perturbation_robustness.py \
    --ckpt_path      "$CKPT" \
    --model          ablation \
    --config         configs/A1.yaml \
    --favc_root      "$FAVC_ROOT" \
    --csv_root_path  "$CSV_ROOT" \
    --split          test \
    --batch_size     32 \
    --seed           42 \
    --output_dir     outputs_A1/results \
    --device         cuda

echo ""
echo "============================================================"
echo "  Perturbation eval done: $(date)"
echo "============================================================"
