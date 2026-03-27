#!/bin/bash
#SBATCH --job-name=avh_perturb_A1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/avh_perturb_A1_%j.out
#SBATCH --error=logs/avh_perturb_A1_%j.err

# ============================================================
#  Experiment 3 — Perturbation Robustness (detection AUC)
#
#  Goal: Measure how FakeAVCeleb detection AUC of Z_c vs Z_s
#        degrades as Gaussian noise is added to input features.
#
#  Expected result (if disentanglement works):
#    Z_c AUC degrades more slowly under audio noise
#    Z_s AUC moves toward 0.5 (random) as acoustic artefacts
#    are destroyed
#
#  Runs all three models: A0, A1, A2
# ============================================================

echo "============================================================"
echo "  Perturbation Robustness (detection AUC) — A0 / A1 / A2"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================================"

cd /data/projects/punim2637/nnliang/AVH-Align

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

mkdir -p avh_sup/outputs_A0/results avh_sup/outputs_A1/results avh_sup/outputs_A2/results avh_sup/outputs_causal/results avh_sup/outputs_baseline/results logs

cd avh_sup

FAVC_ROOT="../data/favc_features"
CSV_ROOT="csv_metadata/favc"

for MODEL in A0 A1 A2; do
    CKPT=$(ls outputs_${MODEL}/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "  WARNING: no checkpoint found for $MODEL, skipping."
        continue
    fi
    echo ""
    echo "--- Running perturbation robustness for $MODEL ---"
    echo "  Checkpoint: $CKPT"
    python eval_perturbation_robustness.py \
        --ckpt_path      "$CKPT" \
        --model          ablation \
        --config         configs/${MODEL}.yaml \
        --favc_root      "$FAVC_ROOT" \
        --csv_root_path  "$CSV_ROOT" \
        --split          test \
        --batch_size     32 \
        --seed           42 \
        --output_dir     outputs_${MODEL}/results \
        --device         cuda
done

# ── outputs_causal (AVH_Causal — passive spurious probe) ─────────────────────
CAUSAL_CKPT="outputs_causal/ckpts/model-epoch=03-v1.ckpt"
if [[ -f "$CAUSAL_CKPT" ]]; then
    echo ""
    echo "--- Running perturbation robustness for outputs_causal ---"
    echo "  Checkpoint: $CAUSAL_CKPT"
    python eval_perturbation_robustness.py \
        --ckpt_path      "$CAUSAL_CKPT" \
        --model          causal \
        --config         configs/A1.yaml \
        --favc_root      "$FAVC_ROOT" \
        --csv_root_path  "$CSV_ROOT" \
        --split          test \
        --batch_size     32 \
        --seed           42 \
        --output_dir     outputs_causal/results \
        --device         cuda
else
    echo "  WARNING: causal checkpoint not found at $CAUSAL_CKPT, skipping."
fi
# ── outputs_baseline (AVH_Sup — paper benchmark model) ────────────────────
BASELINE_CKPT=$(ls outputs_baseline/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
if [[ -n "$BASELINE_CKPT" ]]; then
    echo ""
    echo "--- Running perturbation robustness for baseline (AVH_Sup) ---"
    echo "  Checkpoint: $BASELINE_CKPT"
    python eval_perturbation_robustness.py \
        --ckpt_path      "$BASELINE_CKPT" \
        --model          baseline \
        --config         configs/baseline.yaml \
        --favc_root      "$FAVC_ROOT" \
        --csv_root_path  "$CSV_ROOT" \
        --split          test \
        --batch_size     32 \
        --seed           42 \
        --output_dir     outputs_baseline/results \
        --device         cuda
else
    echo "  WARNING: no baseline checkpoint found, skipping."
fi
echo ""
echo "============================================================"
echo "  Perturbation robustness eval done: $(date)"
echo "============================================================"
