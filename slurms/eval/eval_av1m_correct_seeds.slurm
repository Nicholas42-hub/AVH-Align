#!/bin/bash
#SBATCH --job-name=av1m_correct
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/av1m_correct_mig_%j.out
#SBATCH --error=logs/av1m_correct_mig_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
FEATS="$BASE/data/avh_features"
TRIM_FEATS=/scratch/punim2637/nnliang/avd1m_features_trimmed
CSV_ROOT="$BASE/avh_sup/csv_metadata/av1m"

A2_CKPT="$BASE/avh_sup/outputs_A2_seed45/ckpts/model-epoch=23.ckpt"
A5_CKPT="$BASE/avh_sup/outputs_A5_seed45/ckpts/model-epoch=10.ckpt"
A6_CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/ckpts/model-epoch=09.ckpt"

cd "$BASE"

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

mkdir -p logs
cd avh_sup

echo "=== A2_seed45 untrimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A2_seed45/results_av1m_val_untrimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A2_CKPT" --model ablation \
  --features_path "$FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A2_seed45/results_av1m_val_untrimmed"
echo "A2 untrimmed done: $(date)"

echo "=== A2_seed45 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A2_seed45/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A2_CKPT" --model ablation \
  --features_path "$TRIM_FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A2_seed45/results_av1m_val_trimmed"
echo "A2 trimmed done: $(date)"

echo "=== A5_seed45 untrimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A5_seed45/results_av1m_val_untrimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A5_CKPT" --model fcd \
  --features_path "$FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A5_seed45/results_av1m_val_untrimmed"
echo "A5 untrimmed done: $(date)"

echo "=== A5_seed45 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A5_seed45/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A5_CKPT" --model fcd \
  --features_path "$TRIM_FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A5_seed45/results_av1m_val_trimmed"
echo "A5 trimmed done: $(date)"

echo "=== A6_pseudov2_seed45 untrimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/results_av1m_val_untrimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A6_CKPT" --model fcd_a6 \
  --features_path "$FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/results_av1m_val_untrimmed"
echo "A6 untrimmed done: $(date)"

echo "=== A6_pseudov2_seed45 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$A6_CKPT" --model fcd_a6 \
  --features_path "$TRIM_FEATS" --csv_root_path "$CSV_ROOT" --split val \
  --output_dir "$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_seed45/results_av1m_val_trimmed"
echo "A6 trimmed done: $(date)"

echo "=== ALL DONE ==="
