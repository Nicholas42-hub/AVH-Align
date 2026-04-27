#!/bin/bash
#SBATCH --job-name=A6_newseed_train
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/A6_newseed_train_%A_%a.out
#SBATCH --error=logs/A6_newseed_train_%A_%a.err

# ============================================================
#  Retrain fcd_a6 (full architecture, incon_dim=0) with
#  new random seeds 100/101/102/103 to expand the seed pool
#  and get a more reliable estimate of mean FV-RA AUC.
#
#  Array index → SEED:
#    0: 100
#    1: 101
#    2: 102
#    3: 103
# ============================================================

SEEDS=(100 101 102 103)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

BASE=/data/projects/punim2637/nnliang/AVH-Align
OUTPUT_DIR="$BASE/avh_sup/outputs_A6_seed${SEED}"

echo "============================================================"
echo "  Train fcd_a6 | seed=${SEED}"
echo "  Config  : avh_sup/configs/A6_legacy.yaml"
echo "  Output  : $OUTPUT_DIR"
echo "  Start   : $(date)"
echo "============================================================"

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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

mkdir -p logs
mkdir -p "$OUTPUT_DIR"/{logs,ckpts,results}

cd avh_sup

# Override epochs to 20 via a temporary per-run config
TMP_CFG=$(mktemp /tmp/A6_newseed_seed${SEED}_XXXXXX.yaml)
python -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/A6_legacy.yaml'))
cfg['epochs'] = 20
yaml.dump(cfg, open(sys.argv[1], 'w'), default_flow_style=False)
" "$TMP_CFG"

"$PYTHON" train_test_fcd_a6.py \
  --config_path "$TMP_CFG" \
  --mode train \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR"

rm -f "$TMP_CFG"

echo ""
echo "============================================================"
echo "  Done: fcd_a6 seed=${SEED} at $(date)"
echo "============================================================"
