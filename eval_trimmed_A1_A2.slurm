#!/bin/bash
#SBATCH --job-name=eval_trimmed
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_trimmed_%j.out
#SBATCH --error=logs/eval_trimmed_%j.err

# ============================================================
#  Shortcut Analysis — Evaluate A1 & A2 on TRIMMED FakeAVCeleb
#
#  Goal: Compare AUC on trimmed vs non-trimmed features to check
#        whether the high Causal AUC is partly due to dataset-level
#        shortcuts (opening seconds encoding recording environment).
#
#  If trimmed AUC drops significantly → shortcut contribution exists
#  If trimmed AUC stays stable       → causal feature explanation holds
# ============================================================

echo "============================================================"
echo "  Shortcut Analysis — Trimmed FakeAVCeleb Evaluation"
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

FAVC_TRIMMED="../data/favc_features_trimmed"
CSV_ROOT="csv_metadata/favc"

# ── A1 (epoch=03) — full test set ────────────────────────────────────────────
echo ""
echo "=== A1 (epoch=03) | Trimmed — full test set ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A1/ckpts/model-epoch=03.ckpt \
    --model         ablation \
    --config        configs/A1.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A1/results/trimmed \
    --device        cuda

# ── A1 (epoch=03) — RealVideo-FakeAudio ──────────────────────────────────────
echo ""
echo "=== A1 (epoch=03) | Trimmed — RealVideo-FakeAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A1/ckpts/model-epoch=03.ckpt \
    --model         ablation \
    --config        configs/A1.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A1/results/trimmed \
    --device        cuda

# ── A1 (epoch=03) — FakeVideo-RealAudio ──────────────────────────────────────
echo ""
echo "=== A1 (epoch=03) | Trimmed — FakeVideo-RealAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A1/ckpts/model-epoch=03.ckpt \
    --model         ablation \
    --config        configs/A1.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A1/results/trimmed \
    --device        cuda

# ── A2 (epoch=09) — full test set ────────────────────────────────────────────
echo ""
echo "=== A2 (epoch=09) | Trimmed — full test set ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A2/ckpts/model-epoch=09.ckpt \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results/trimmed \
    --device        cuda

# ── A2 (epoch=09) — RealVideo-FakeAudio ──────────────────────────────────────
echo ""
echo "=== A2 (epoch=09) | Trimmed — RealVideo-FakeAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A2/ckpts/model-epoch=09.ckpt \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results/trimmed \
    --device        cuda

# ── A2 (epoch=09) — FakeVideo-RealAudio ──────────────────────────────────────
echo ""
echo "=== A2 (epoch=09) | Trimmed — FakeVideo-RealAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     outputs_A2/ckpts/model-epoch=09.ckpt \
    --model         ablation \
    --config        configs/A2.yaml \
    --favc_root     "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A2/results/trimmed \
    --device        cuda

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
