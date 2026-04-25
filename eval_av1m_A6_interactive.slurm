#!/bin/bash
#SBATCH --job-name=av1m_A6
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/av1m_A6_interactive_%j.out
#SBATCH --error=logs/av1m_A6_interactive_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

CKPT="$BASE/avh_sup/outputs_A6/ckpts/model-epoch=15.ckpt"
RESULTS_DIR="$BASE/avh_sup/outputs_A6/results_av1m_val_untrimmed"
mkdir -p "$RESULTS_DIR" logs

echo "=== A6 AV1M untrimmed val ==="
cd avh_sup
$PYTHON eval_av1m.py \
  --ckpt "$CKPT" \
  --model fcd_a6 \
  --features_path "$BASE/data/avh_features" \
  --csv_root_path "$BASE/avh_sup/csv_metadata/av1m" \
  --split val \
  --output_dir "$RESULTS_DIR"
echo "Done: $(date)"
