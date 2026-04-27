#!/bin/bash
#SBATCH --job-name=a2_mask_fvra
#SBATCH --partition=gpu-l40s-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/a2_mask_fvra_full_%A_%a.out
#SBATCH --error=logs/a2_mask_fvra_full_%A_%a.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SEEDS=(43 44 45)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="$BASE/avh_sup/outputs_A2_seed${SEED}"
RESULTS_DIR="$OUTPUT_DIR/results_favc_7030_full_input_mask_fvra"
CKPT=$(ls "$OUTPUT_DIR"/ckpts/*.ckpt 2>/dev/null | sort | tail -1)

if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found under $OUTPUT_DIR/ckpts" >&2
  exit 1
fi

echo "============================================================"
echo "  Two-way baseline (A2) input masking FV-RA full FAVC 70/30"
echo "  Seed    : $SEED"
echo "  Output  : $OUTPUT_DIR"
echo "  Ckpt    : $CKPT"
echo "  CSV     : $CSV_ROOT"
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
"$PYTHON" eval_perturbation_robustness.py \
  --ckpt_path "$CKPT" \
  --model ablation \
  --config configs/A2.yaml \
  --favc_root "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --batch_size 1 \
  --seed "$SEED" \
  --output_dir "$RESULTS_DIR" \
  --fake_category FakeVideo-RealAudio

echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"
