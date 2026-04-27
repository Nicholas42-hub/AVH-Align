#!/bin/bash
#SBATCH --job-name=A6pdv2_ablate
#SBATCH --partition=gpu-a100-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/A6_pdv2_50ep_ablate_%A_%a.out
#SBATCH --error=logs/A6_pdv2_50ep_ablate_%A_%a.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
SEEDS=(44 45)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_50ep_seed${SEED}"
CONFIG="$BASE/avh_sup/configs/A6_av1m_pseudodomain_v2_50ep_seed${SEED}.yaml"
CKPT=$(ls "$OUTPUT_DIR"/ckpts/favc_fvra/best-favc-fvra-*.ckpt 2>/dev/null | sort -V | tail -1)

if [[ -z "$CKPT" ]]; then
  echo "ERROR: no best FV-RA checkpoint found under $OUTPUT_DIR/ckpts/favc_fvra" >&2
  exit 1
fi

echo "============================================================"
echo "  A6 pdv2 50ep ablation studies"
echo "  Seed    : $SEED"
echo "  Output  : $OUTPUT_DIR"
echo "  Config  : $CONFIG"
echo "  Ckpt    : $CKPT"
echo "  Start   : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs \
  "$OUTPUT_DIR/results_head_ablation_pdv2_50ep" \
  "$OUTPUT_DIR/results_favc_7030_full_input_mask_fvra_pdv2_50ep" \
  "$OUTPUT_DIR/results_silence_rob_pdv2_50ep" \
  "$OUTPUT_DIR/results_silence_rob_trimmed_pdv2_50ep"

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

cd "$BASE/avh_sup"

echo ""
echo "==== Exp 4: head/subspace ablation ===="
"$PYTHON" eval_head_ablation.py \
  --ckpt_path "$CKPT" \
  --config "$CONFIG" \
  --av1m_root "$BASE/data/avh_features/val" \
  --av1m_csv "$BASE/avh_sup/csv_metadata/av1m/val_labels.csv" \
  --favc_root "$BASE/data/favc_features" \
  --output_dir "$OUTPUT_DIR/results_head_ablation_pdv2_50ep" \
  --device cuda

echo ""
echo "==== Exp 7: FV-RA input masking / perturbation sweep ===="
"$PYTHON" eval_perturbation_robustness.py \
  --ckpt_path "$CKPT" \
  --model fcd_a6 \
  --config "$CONFIG" \
  --favc_root "$BASE/data/favc_features" \
  --csv_root_path "$BASE/avh_sup/csv_metadata/favc_7030_canonical" \
  --split test \
  --batch_size 1 \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR/results_favc_7030_full_input_mask_fvra_pdv2_50ep" \
  --fake_category FakeVideo-RealAudio \
  --device cuda

echo ""
echo "==== Exp 5a: leading-silence diagnostic, untrimmed features ===="
"$PYTHON" eval_leading_silence_robustness.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$BASE/data/favc_features" \
  --csv_root_path "$BASE/avh_sup/csv_metadata/favc" \
  --split test \
  --trim_frames 1 3 5 10 \
  --output_dir "$OUTPUT_DIR/results_silence_rob_pdv2_50ep" \
  --device cuda

echo ""
echo "==== Exp 5b: true-trimmed feature evaluation ===="
"$PYTHON" eval_leading_silence_robustness.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$BASE/data/favc_features_trimmed" \
  --csv_root_path "$BASE/avh_sup/csv_metadata/favc" \
  --split test \
  --trim_frames 0 \
  --output_dir "$OUTPUT_DIR/results_silence_rob_trimmed_pdv2_50ep" \
  --device cuda

echo "============================================================"
echo "  Done: seed${SEED} ablation studies at $(date)"
echo "============================================================"
