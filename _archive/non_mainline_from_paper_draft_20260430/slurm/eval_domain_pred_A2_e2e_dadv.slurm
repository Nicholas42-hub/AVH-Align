#!/bin/bash
#SBATCH --job-name=domain_A2_dadv
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/domain_A2_e2e_dadv_%j.out
#SBATCH --error=logs/domain_A2_e2e_dadv_%j.err

# ============================================================
#  Domain Predictability Probe — A2_e2e_dadv
#
#  Goal: Measure how well Z_c vs Z_s can predict dataset identity
#        AV1M (0) vs FakeAVCeleb (1) using logistic regression.
#
#  Expected result:
#    Z_c domain AUC ≈ 0.5   (GRL pushed Z_c domain-invariant)
#    Z_s domain AUC >> 0.5  (spurious branch absorbed domain artefacts)
# ============================================================

echo "============================================================"
echo "  Domain Predictability Probe — A2_e2e_dadv"
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
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

BASE=/data/projects/punim2637/nnliang/AVH-Align
CKPT_DIR="$BASE/avh_sup/outputs_A2_e2e_dadv/ckpts"

# Auto-select newest-by-mtime best checkpoint
BEST=$(ls -t "$CKPT_DIR"/model-epoch=*.ckpt 2>/dev/null | grep -v "last" | head -1)
CKPT="${BEST:-$CKPT_DIR/last.ckpt}"
echo ">>> Using checkpoint: $CKPT"

mkdir -p "$BASE/avh_sup/outputs_A2_e2e_dadv/results" "$BASE/logs"

cd "$BASE/avh_sup"

echo ">>> Running domain predictability probe (Z_c vs Z_s, AV1M vs FAVC)..."
$PYTHON eval_e2e_domain_a2_dadv.py \
    --ckpt       "$CKPT" \
    --config     "$BASE/avh_sup/configs/A2_e2e_dadv.yaml" \
    --av1m_raw   /scratch/punim2637/nnliang/avd1m_preprocessed/val \
    --av1m_csv   "$BASE/avh_sup/csv_metadata/av1m_e2e" \
    --favc_raw   /scratch/punim2637/nnliang/favc_preprocessed \
    --output_dir "$BASE/avh_sup/outputs_A2_e2e_dadv/results" \
    --max_clips  500 \
    --batch_size 4 \
    --num_workers 2

echo ""
echo "============================================================"
echo "  Domain probe done: $(date)"
echo "  Results: $BASE/avh_sup/outputs_A2_e2e_dadv/results/eval_e2e_domain.json"
echo "============================================================"
cat "$BASE/avh_sup/outputs_A2_e2e_dadv/results/eval_e2e_domain.json" 2>/dev/null
