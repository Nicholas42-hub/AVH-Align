#!/bin/bash
#SBATCH --job-name=xdom_trim_s42
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xdom_trim_s42_%j.out
#SBATCH --error=logs/xdom_trim_s42_%j.err

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
FEATURES=/scratch/punim2637/nnliang/avd1m_features_trimmed
CSV_ROOT="$BASE/avh_sup/csv_metadata/av1m"
SEED=42

echo "============================================================"
echo "  Cross-domain eval: FAVC-trained -> trimmed AV1M val"
echo "  Seed: 42  Models: A2, A5, A6"
echo "  Job ID : $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd "$BASE"
module purge
module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs
cd avh_sup

run_eval () {
    local tag="$1"
    local model="$2"
    local output_dir="$BASE/avh_sup/outputs_${tag}_favc_seed${SEED}"
    local results_dir="${output_dir}/results_av1m_trimmed_crossdomain"
    local ckpt
    ckpt=$(ls "${output_dir}/ckpts/"*.ckpt 2>/dev/null | sort | tail -1)
    echo ""
    echo "=== ${tag} seed${SEED} ==="
    echo "$(date)"
    if [[ -z "$ckpt" ]]; then
        echo "No ckpt for ${tag} seed${SEED}, skipping"
        return 0
    fi
    mkdir -p "$results_dir"
    "$PYTHON" eval_av1m.py \
        --ckpt          "$ckpt" \
        --model         "$model" \
        --features_path "$FEATURES" \
        --csv_root_path "$CSV_ROOT" \
        --split         val \
        --output_dir    "$results_dir"
    echo "${tag} seed${SEED} done: $(date)"
}

run_eval A2 ablation
run_eval A5 fcd
run_eval A6 fcd_a6

echo ""
echo "Seed42 trimmed cross-domain evals done: $(date)"
