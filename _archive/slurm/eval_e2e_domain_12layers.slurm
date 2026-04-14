#!/bin/bash
#SBATCH --job-name=avh_e2e_eval_12L
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=logs/eval_e2e_domain_12layers_%j.out
#SBATCH --error=logs/eval_e2e_domain_12layers_%j.err

echo "============================================================"
echo "  E2E Domain & Label Probe Evaluation (12 layers)"
echo "  Job : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start: $(date)"
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

AV1M_RAW="/data/scratch/projects/punim2637/nnliang/avd1m_preprocessed"
FAVC_RAW="/data/scratch/projects/punim2637/nnliang/favc_preprocessed"
AV1M_CSV="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m"
CKPT_DIR="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A2_e2e_12layers/ckpts"
CONFIG="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/A2_e2e_12layers.yaml"
OUTPUT_DIR="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_A2_e2e_12layers/results"

# Auto-select best checkpoint by val_auc_causal in filename
CKPT=$(python3 -c "
import glob, re, sys
ckpts = glob.glob('$CKPT_DIR/model-epoch=*.ckpt')
if not ckpts:
    sys.exit('No checkpoints found in $CKPT_DIR')
best = max(ckpts, key=lambda p: float(m.group(1)) if (m := re.search(r'val_auc_causal=([0-9]+\.[0-9]+)', p)) else 0.0)
print(best)
")

if [[ -z "$CKPT" ]]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

echo "Using checkpoint: $CKPT"

mkdir -p logs "$OUTPUT_DIR"

cd avh_sup

echo "Running domain + label probe on e2e-12L checkpoint..."
python eval_e2e_domain.py \
    --ckpt      "$CKPT" \
    --config    "$CONFIG" \
    --av1m_raw  "$AV1M_RAW" \
    --favc_raw  "$FAVC_RAW" \
    --av1m_csv  "$AV1M_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --holdout_only \
    --max_clips 500 \
    --batch_size 4 \
    --num_workers 2

echo ""
echo "============================================================"
echo "  Eval done: $(date)"
echo "============================================================"
