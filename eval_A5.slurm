#!/bin/bash
#SBATCH --job-name=eval_A5
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_A5_%j.out
#SBATCH --error=logs/eval_A5_%j.err

echo "============================================================"
echo "  A5 Evaluation — FakeAVCeleb (trimmed + non-trimmed)"
echo "  + Per-category perturbation robustness"
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

mkdir -p avh_sup/outputs_A5/results/trimmed avh_sup/outputs_A5/results logs

cd avh_sup

# Auto-detect best checkpoint saved by ModelCheckpoint
CKPT=$(ls outputs_A5/ckpts/*.ckpt | sort | tail -1)
echo "Using checkpoint: $CKPT"
FAVC_ROOT="../data/favc_features"
FAVC_TRIMMED="../data/favc_features_trimmed"
CSV_ROOT="csv_metadata/favc"

# ── 1. Cross-domain eval — non-trimmed (full test set) ────────────────────────
echo ""
echo "=== A5 | FakeAVCeleb non-trimmed ==="
python eval_favc.py \
    --ckpt          "$CKPT" \
    --model         fcd \
    --features_path "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    outputs_A5/results \
    --device        cuda

# ── 2. Cross-domain eval — trimmed (match paper baseline) ────────────────────
echo ""
echo "=== A5 | FakeAVCeleb trimmed ==="
python eval_favc.py \
    --ckpt          "$CKPT" \
    --model         fcd \
    --features_path "$FAVC_TRIMMED" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --output_dir    outputs_A5/results/trimmed \
    --device        cuda

# ── 3. Perturbation robustness — full test set ────────────────────────────────
echo ""
echo "=== A5 | Perturbation — full test set ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         fcd \
    --config        configs/A5.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A5/results \
    --device        cuda

# ── 4. Per-category perturbation — RealVideo-FakeAudio ───────────────────────
echo ""
echo "=== A5 | Perturbation — RealVideo-FakeAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         fcd \
    --config        configs/A5.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category RealVideo-FakeAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A5/results \
    --device        cuda

# ── 5. Per-category perturbation — FakeVideo-RealAudio ───────────────────────
echo ""
echo "=== A5 | Perturbation — FakeVideo-RealAudio ==="
python eval_perturbation_robustness.py \
    --ckpt_path     "$CKPT" \
    --model         fcd \
    --config        configs/A5.yaml \
    --favc_root     "$FAVC_ROOT" \
    --csv_root_path "$CSV_ROOT" \
    --split         test \
    --fake_category FakeVideo-RealAudio \
    --batch_size    32 \
    --seed          42 \
    --output_dir    outputs_A5/results \
    --device        cuda

echo ""
echo "============================================================"
echo "  All done: $(date)"
echo "============================================================"
