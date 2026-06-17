#!/bin/bash
#SBATCH --job-name=avlips_mask_s44
#SBATCH --partition=gpu-a100-mig
#SBATCH --account=punim2637
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/avlips_mask_s44_%A_%a.out
#SBATCH --error=logs/avlips_mask_s44_%A_%a.err

# AVLips visual + audio masking sweep at seed=44 for the three model families.
# levels = {0.0, 0.5, 1.0} for both visual_masking and audio_masking.
# Generalisation strengthening for #16: AVLips perturbation companion to FAVC.

BASE=/data/projects/punim2637/nnliang/AVH-Align
AVLIPS="$BASE/data/avlips_features"
SEED=44

# Index → (model_arg, ckpt, label)
case $SLURM_ARRAY_TASK_ID in
  0)
    MODEL=ablation
    CKPT="$BASE/avh_sup/outputs_A2_seed44/ckpts/model-epoch=13.ckpt"
    OUT="$BASE/avh_sup/outputs_A2_seed44/results_avlips_masking"
    LABEL="A2 two-way baseline"
    ;;
  1)
    MODEL=fcd
    CKPT="$BASE/avh_sup/outputs_A5_seed44/ckpts/model-epoch=03.ckpt"
    OUT="$BASE/avh_sup/outputs_A5_seed44/results_avlips_masking"
    LABEL="A5 three-way no push-pull"
    ;;
  2)
    MODEL=fcd_a6
    CKPT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_100ep_seed44/ckpts/favc_fvra/best-favc-fvra-epoch=60-auc=0.8372.ckpt"
    OUT="$BASE/avh_sup/outputs_A6_av1m_pseudodomain_v2_100ep_seed44/results_avlips_masking"
    LABEL="A6 TriRoute 100ep best"
    ;;
  *)
    echo "Bad array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing ckpt: $CKPT" >&2
  exit 1
fi

echo "============================================================"
echo "  AVLips masking sweep — $LABEL — seed=$SEED"
echo "  Ckpt   : $CKPT"
echo "  Out    : $OUT"
echo "  Start  : $(date)"
echo "============================================================"

cd "$BASE"
mkdir -p logs "$OUT"
module purge
module load Miniconda3/23.10.0-1
module load GCC/11.3.0
module load CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

cd avh_sup
python eval_avlips_masking.py \
    --ckpt          "$CKPT" \
    --model         "$MODEL" \
    --features_path "$AVLIPS" \
    --output_dir    "$OUT" \
    --seed          "$SEED"

echo "Done $LABEL: $(date)"
