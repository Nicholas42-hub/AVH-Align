#!/bin/bash
#SBATCH --job-name=avhsup_eval
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/avhsup_eval_%A_%a.out
#SBATCH --error=logs/avhsup_eval_%A_%a.err

# ============================================================
#  AVH-Align/sup evaluation for use as external baseline.
#  Array 0: matched FAVC (untrimmed, AV1M-trained ckpt)   – csv: favc_7030_canonical, N~1114 matched
#  Array 1: matched FAVC (trimmed,   AV1M-trained ckpt)   – csv: favc_7030_canonical, N~1114 matched
#  Array 2: FAVC 70/30 full (untrimmed, AV1M-trained ckpt) – csv: favc_7030_canonical, N=6667
#  NOTE: For matched tasks (0,1) the features dir only has ~1114 clips so N~1114.
#        For 70/30 task (2) features dir has all 9642 clips → N=6667.
# ============================================================

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT="$BASE/checkpoints/avh_sup/AVH_Sup_AV1M.ckpt"
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

# favc_features      = matched N=1114 (only matched clips present)
# favc_features_full = full 70/30 N=6667 (all FAVC test clips present)
# favc_features_trimmed = matched N=1114, leading-silence trimmed
# Both matched and 70/30 use the same csv_root (favc/test_split.csv has 6668 rows);
# the actual N is determined by which features exist on disk.
MATCHED_FEAT="$BASE/data/favc_features"
TRIMMED_FEAT="$BASE/data/favc_features_trimmed"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"

FEATURES_ARR=("$MATCHED_FEAT" "$TRIMMED_FEAT" "$MATCHED_FEAT")
OUT_ARR=(
  "$BASE/avh_sup/outputs_avhsup_av1m/results_favc_matched"
  "$BASE/avh_sup/outputs_avhsup_av1m/results_favc_matched_trimmed"
  "$BASE/avh_sup/outputs_avhsup_av1m/results_favc_7030"
)
# Array 0 → matched untrimmed, 1 → matched trimmed, 2 → 70/30 full untrimmed

FEATURES=${FEATURES_ARR[$SLURM_ARRAY_TASK_ID]}
OUTPUT_DIR=${OUT_ARR[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  AVH-Align/sup eval | task=${SLURM_ARRAY_TASK_ID}"
echo "  Ckpt    : $CKPT"
echo "  Features: $FEATURES"
echo "  CSV     : $CSV_ROOT"
echo "  Out     : $OUTPUT_DIR"
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

mkdir -p "$OUTPUT_DIR" logs

cd avh_sup

$PYTHON eval_favc.py \
  --ckpt "$CKPT" \
  --model baseline \
  --features_path "$FEATURES" \
  --csv_root_path "$CSV_ROOT" \
  --split test \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Done: AVH-Align/sup task=${SLURM_ARRAY_TASK_ID} at $(date)"
echo "============================================================"
