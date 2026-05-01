#!/bin/bash
#SBATCH --job-name=label_probe_A5
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/label_probe_A5_%j.out
#SBATCH --error=logs/label_probe_A5_%j.err

# ============================================================
#  Label Predictability Probe — A5 (FCD)
#
#  Probes frozen Z_causal = [s_v, u_v, s_a, u_a]
#        and     Z_redundant = [r_v, r_a]
#  using a logistic regression classifier.
#
#  Expected results if FCD works:
#    Z_causal  in-domain AUC >> 0.5   (task-useful)
#    Z_redundant in-domain AUC ≈ 0.5  (task-agnostic)
#    Z_causal  OOD AUC >> Z_redundant OOD AUC
# ============================================================

echo "============================================================"
echo "  Label Predictability Probe — A5 (FCD)"
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

mkdir -p avh_sup/outputs_A5/results logs

cd avh_sup

CKPT=$(ls outputs_A5/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint found in outputs_A5/ckpts/ — has training finished?"
    exit 1
fi
echo "Using checkpoint: $CKPT"

python eval_label_probe.py \
    --ckpt_path  "$CKPT" \
    --config     configs/A5.yaml \
    --av1m_root  ../data/avh_features/val \
    --av1m_csv   csv_metadata/av1m/val_labels.csv \
    --favc_root  ../data/favc_features \
    --output_dir outputs_A5/results \
    --device     cuda

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
