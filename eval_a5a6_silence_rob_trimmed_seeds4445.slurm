#!/bin/bash
#SBATCH --job-name=a5a6_sil_trim
#SBATCH --partition=gpu-l40s-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/a5a6_silence_rob_trimmed_%A_%a.out
#SBATCH --error=logs/a5a6_silence_rob_trimmed_%A_%a.err

# Array mapping: 0=A5s44, 1=A5s45, 2=A6s44, 3=A6s45

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

MODELS=(A5 A5 A6 A6)
SEEDS=(44 45 44 45)
MODEL_FLAGS=(fcd fcd fcd_a6_lite fcd_a6_lite)

IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$IDX]}
SEED=${SEEDS[$IDX]}
MODEL_FLAG=${MODEL_FLAGS[$IDX]}

OUTPUT_DIR="$BASE/avh_sup/outputs_${MODEL}_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_silence_rob_trimmed"
CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)

if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi

echo "============================================================"
echo "  ${MODEL} seed${SEED} leading-silence robustness (truly-trimmed)"
echo "  Ckpt    : $CKPT"
echo "  Results : $RESULTS_DIR"
echo "  Start   : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$RESULTS_DIR"

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

cd avh_sup
# Use truly-trimmed features (raw waveform trimmed, re-extracted AV-HuBERT)
python eval_leading_silence_robustness.py \
    --ckpt      "$CKPT" \
    --model     "$MODEL_FLAG" \
    --features_path   "$BASE/data/favc_features_trimmed" \
    --csv_root_path   csv_metadata/favc \
    --split     test \
    --trim_frames 0 \
    --output_dir "$RESULTS_DIR"

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
