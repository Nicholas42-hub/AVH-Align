#!/bin/bash
#SBATCH --job-name=eval_xdomain
#SBATCH --partition=gpu-a100-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_xdomain_%j.out
#SBATCH --error=logs/eval_xdomain_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
FEATURES="$BASE/data/avh_features"
CSV_ROOT="$BASE/avh_sup/csv_metadata/av1m"

echo "============================================================"
echo "  Cross-domain eval: FAVC-trained models → AV1M val set"
echo "  Job ID : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
cd avh_sup

for SEED in 43 44 45; do
    echo ""; echo "=== A2 seed${SEED} ===" ; echo "$(date)"
    OUTPUT_DIR="$BASE/avh_sup/outputs_A2_favc_seed${SEED}"
    RESULTS_DIR="${OUTPUT_DIR}/results_av1m_val_crossdomain"
    CKPT=$(ls "${OUTPUT_DIR}/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then echo "No ckpt for A2 seed${SEED}, skipping"; continue; fi
    mkdir -p "$RESULTS_DIR"
    $PYTHON eval_av1m.py \
        --ckpt         "$CKPT" \
        --model        ablation \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split        val \
        --output_dir   "$RESULTS_DIR"
    echo "A2 seed${SEED} done: $(date)"
done

for SEED in 43 44 45; do
    echo ""; echo "=== A5 seed${SEED} ===" ; echo "$(date)"
    OUTPUT_DIR="$BASE/avh_sup/outputs_A5_favc_seed${SEED}"
    RESULTS_DIR="${OUTPUT_DIR}/results_av1m_val_crossdomain"
    CKPT=$(ls "${OUTPUT_DIR}/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then echo "No ckpt for A5 seed${SEED}, skipping"; continue; fi
    mkdir -p "$RESULTS_DIR"
    $PYTHON eval_av1m.py \
        --ckpt         "$CKPT" \
        --model        fcd \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split        val \
        --output_dir   "$RESULTS_DIR"
    echo "A5 seed${SEED} done: $(date)"
done

for SEED in 43 44 45; do
    echo ""; echo "=== A6 seed${SEED} ===" ; echo "$(date)"
    OUTPUT_DIR="$BASE/avh_sup/outputs_A6_favc_seed${SEED}"
    RESULTS_DIR="${OUTPUT_DIR}/results_av1m_val_crossdomain"
    CKPT=$(ls "${OUTPUT_DIR}/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then echo "No ckpt for A6 seed${SEED}, skipping"; continue; fi
    mkdir -p "$RESULTS_DIR"
    $PYTHON eval_av1m.py \
        --ckpt         "$CKPT" \
        --model        fcd_a6 \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split        val \
        --output_dir   "$RESULTS_DIR"
    echo "A6 seed${SEED} done: $(date)"
done

echo ""; echo "All cross-domain evals done: $(date)"
