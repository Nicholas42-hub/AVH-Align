#!/bin/bash
#SBATCH --job-name=perturb_causal_hyp
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/perturb_causal_hyp_%j.out
#SBATCH --error=logs/perturb_causal_hyp_%j.err

# ============================================================
#  Experiment 1 — Causal Direction Hypothesis
#
#  Hypothesis: audio is the cause, visual is the effect.
#    - RealVideo-FakeAudio: causal path (audio) is broken
#      → Causal head should be HIGHLY sensitive to audio noise
#    - FakeVideo-RealAudio: effect (visual) is broken
#      → Causal head should be MORE STABLE under audio noise
#
#  Models tested: A1 (epoch=03, AUC=0.78), A2 (epoch=09, AUC=0.79)
# ============================================================

echo "============================================================"
echo "  Causal Direction Hypothesis — Per-Category Perturbation"
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

mkdir -p avh_sup/outputs_A1/results avh_sup/outputs_A2/results logs

cd avh_sup

FAVC_ROOT="../data/favc_features"
CSV_ROOT="csv_metadata/favc"

# ── A1 (epoch=03, the best-performing A1 checkpoint) ─────────────────────────

echo ""
echo "=== A1 (epoch=03) | RealVideo-FakeAudio (audio=broken causal) ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A1/ckpts/model-epoch=03.ckpt \
    --model         ablation \
    --config        configs/A1.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A1/results \
    --device        cuda

echo ""
echo "=== A1 (epoch=03) | FakeVideo-RealAudio (visual=broken effect) ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A1/ckpts/model-epoch=03.ckpt \
    --model         ablation \
    --config        configs/A1.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A1/results \
    --device        cuda

# ── A2 (epoch=09) ─────────────────────────────────────────────────────────────

echo ""
echo "=== A2 (epoch=09) | RealVideo-FakeAudio (audio=broken causal) ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A2/ckpts/model-epoch=09.ckpt \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results \
    --device        cuda

echo ""
echo "=== A2 (epoch=09) | FakeVideo-RealAudio (visual=broken effect) ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A2/ckpts/model-epoch=09.ckpt \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results \
    --device        cuda

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
