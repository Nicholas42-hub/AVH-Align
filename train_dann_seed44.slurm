#!/bin/bash
#SBATCH --job-name=dann_s44
#SBATCH --partition=gpu-a100-short
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:50:00
#SBATCH --output=logs/dann_seed44_%j.out
#SBATCH --error=logs/dann_seed44_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
FEATURES="$BASE/data/favc_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/favc_7030_canonical"
SEED=44
METHOD=dann
OUT="$BASE/avh_sup/outputs_DANN_seed${SEED}"

echo "=== ${METHOD} seed=${SEED} on $(hostname) ==="
echo "Start: $(date)"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

cd avh_sup
python train_da_baselines.py     --method       $METHOD     --config_path  configs/DANN.yaml     --seed         $SEED     --output_dir   "$OUT"

CKPT=$(ls "$OUT/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
if [[ -z "$CKPT" ]]; then echo "ERROR: no checkpoint" >&2; exit 1; fi

python eval_favc.py     --ckpt          "$CKPT"     --model         $METHOD     --features_path "$FEATURES"     --csv_root_path "$CSV_ROOT"     --split         test     --output_dir    "$OUT/results_favc_7030"

echo "Done: $(date)"
