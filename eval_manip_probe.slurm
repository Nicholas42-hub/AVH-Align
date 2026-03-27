#!/bin/bash
#SBATCH --job-name=manip_probe
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/manip_probe_%j.out
#SBATCH --error=logs/manip_probe_%j.err

# ============================================================
#  Manipulation-Type Probe — A0 / A1 / A2 / A3
#
#  Task: FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)
#  Both categories have the SAME video manipulation.
#  Only difference: real audio vs synthesised/swapped audio.
#
#  Expected result (if disentanglement works):
#    Z_s AUC >> raw AUC  (spurious branch captures audio style)
#    Z_c AUC ≈  raw AUC  (causal/sync branch blind to audio type)
#
#  Unlike domain probe (AV1M vs FAVC), raw AUC is NOT expected
#  to be ~99% here — this is a meaningful, model-attributable probe.
# ============================================================

echo "============================================================"
echo "  Manipulation-Type Probe — A0 / A1 / A2 / A3"
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

FAVC_ROOT="data/favc_features"

mkdir -p logs \
    avh_sup/outputs_A0/results \
    avh_sup/outputs_A1/results \
    avh_sup/outputs_A2/results \
    avh_sup/outputs_A3/results

cd avh_sup

for MODEL in A0 A1 A2 A3; do
    CKPT=$(ls outputs_${MODEL}/ckpts/*.ckpt 2>/dev/null | sort | tail -1)
    if [[ -z "$CKPT" ]]; then
        echo "WARNING: no checkpoint for $MODEL, skipping."
        continue
    fi

    echo ""
    echo "=== Manipulation probe: $MODEL | $(basename $CKPT) ==="
    python eval_manipulation_probe.py \
        --ckpt_path     "$CKPT" \
        --config        configs/${MODEL}.yaml \
        --favc_root     "../$FAVC_ROOT" \
        --max_per_class 1000 \
        --n_folds       5 \
        --seed          42 \
        --output_dir    outputs_${MODEL}/results \
        --device        cuda
done

echo ""
echo "============================================================"
echo "  All manipulation probes done: $(date)"
echo "============================================================"
