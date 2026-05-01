#!/bin/bash
#SBATCH --job-name=av1m_trim
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/av1m_trim_interactive_%j.out
#SBATCH --error=logs/av1m_trim_interactive_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
TRIM_FEATS=/scratch/punim2637/nnliang/avd1m_features_trimmed
CSV_ROOT="$BASE/avh_sup/csv_metadata/av1m"

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

mkdir -p logs

cd avh_sup

echo "=== A2 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A2/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$BASE/avh_sup/outputs_A2/ckpts/model-epoch=09.ckpt" \
  --model ablation \
  --features_path "$TRIM_FEATS" \
  --csv_root_path "$CSV_ROOT" \
  --split val \
  --output_dir "$BASE/avh_sup/outputs_A2/results_av1m_val_trimmed"
echo "A2 done: $(date)"

echo "=== A5 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A5/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$BASE/avh_sup/outputs_A5/ckpts/model-epoch=17.ckpt" \
  --model fcd \
  --features_path "$TRIM_FEATS" \
  --csv_root_path "$CSV_ROOT" \
  --split val \
  --output_dir "$BASE/avh_sup/outputs_A5/results_av1m_val_trimmed"
echo "A5 done: $(date)"

echo "=== A6 trimmed ==="
mkdir -p "$BASE/avh_sup/outputs_A6/results_av1m_val_trimmed"
$PYTHON eval_av1m.py \
  --ckpt "$BASE/avh_sup/outputs_A6/ckpts/model-epoch=15.ckpt" \
  --model fcd_a6 \
  --features_path "$TRIM_FEATS" \
  --csv_root_path "$CSV_ROOT" \
  --split val \
  --output_dir "$BASE/avh_sup/outputs_A6/results_av1m_val_trimmed"
echo "A6 done: $(date)"
