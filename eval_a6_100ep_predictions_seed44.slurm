#!/bin/bash
#SBATCH --job-name=a6_predcsv_s44
#SBATCH --partition=gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/a6_predcsv_s44_%A.out
#SBATCH --error=logs/a6_predcsv_s44_%A.err

# Generates per-clip predictions.csv on FAVC test for A6_100ep seed=44 best ckpt
# (epoch 60, AUC 0.8372). Required for downstream qualitative failure mining.

BASE=/data/projects/punim2637/nnliang/AVH-Align
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_100ep_seed44/ckpts/favc_fvra/best-favc-fvra-epoch=60-auc=0.8372.ckpt"
OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_100ep_seed44/results_favc_7030"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: ckpt missing: $CKPT" >&2
  exit 1
fi

echo "============================================================"
echo "  A6 100ep seed=44 → FAVC test predictions.csv"
echo "  Ckpt    : $CKPT"
echo "  Output  : $OUT"
echo "  Start   : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$OUT"
module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

cd avh_sup
python eval_favc.py \
    --ckpt          "$CKPT" \
    --model         fcd_a6 \
    --features_path "$FEATURES" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    "$OUT"

echo "Done: $(date)"
