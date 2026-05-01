#!/bin/bash
#SBATCH --job-name=avhsup_modal
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/avhsup_modal_%A_%a.out
#SBATCH --error=logs/avhsup_modal_%A_%a.err

set -euo pipefail

COMBOS=(
  "both 43"
  "both 44"
  "both 45"
  "audio 43"
  "audio 44"
  "audio 45"
  "video 43"
  "video 44"
  "video 45"
)

read -r INPUT_TYPE SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo "  Plain AVH_Sup modal baseline"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Array task : $SLURM_ARRAY_TASK_ID"
echo "  Node       : $SLURM_NODELIST"
echo "  Input type : $INPUT_TYPE"
echo "  Seed       : $SEED"
echo "  Start      : $(date)"
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

OUT="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_avhsup_av1m_${INPUT_TYPE}_seed${SEED}"
mkdir -p "$OUT"/{logs,ckpts,results_favc_7030} logs

cd avh_sup

python train_avh_sup.py \
  --config_path configs/baseline.yaml \
  --input_type "$INPUT_TYPE" \
  --seed "$SEED" \
  --output_dir "$OUT"

CKPT="$(ls -t "$OUT"/ckpts/*.ckpt | head -1)"
echo "Selected checkpoint: $CKPT"

python eval_favc.py \
  --ckpt "$CKPT" \
  --model baseline \
  --features_path ../data/favc_features \
  --csv_root_path csv_metadata/favc \
  --split test \
  --output_dir "$OUT/results_favc_7030"

echo "============================================================"
echo "  Done: $(date)"
echo "  Results: $OUT/results_favc_7030"
echo "============================================================"
