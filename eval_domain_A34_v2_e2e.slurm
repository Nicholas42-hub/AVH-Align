#!/bin/bash
#SBATCH --job-name=eval_domain_A34_v2_e2e
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=32G --time=2:00:00
#SBATCH --output=logs/eval_domain_A34_v2_e2e_%j.out
#SBATCH --error=logs/eval_domain_A34_v2_e2e_%j.err
cd /data/projects/punim2637/nnliang/AVH-Align
module purge; module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/punim2637/nnliang/avh_env
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export PYTHONUNBUFFERED=1; export CUDA_VISIBLE_DEVICES=0
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python
BASE=/data/projects/punim2637/nnliang/AVH-Align
BEST=$(ls -t "$BASE/avh_sup/outputs_A34_v2_e2e/ckpts"/model-epoch=*.ckpt 2>/dev/null | grep -v last | head -1)
CKPT="${BEST:-$BASE/avh_sup/outputs_A34_v2_e2e/ckpts/last.ckpt}"
echo ">>> Using checkpoint: $CKPT"
mkdir -p "$BASE/avh_sup/outputs_A34_v2_e2e/results" "$BASE/logs"
cd "$BASE/avh_sup"
$PYTHON eval_domain_a34_e2e.py --ckpt "$CKPT" --config "$BASE/avh_sup/configs/A34_v2_e2e.yaml" \
    --av1m_raw /scratch/punim2637/nnliang/avd1m_preprocessed/val \
    --av1m_csv "$BASE/avh_sup/csv_metadata/av1m_e2e" \
    --favc_raw /scratch/punim2637/nnliang/favc_preprocessed \
    --output_dir "$BASE/avh_sup/outputs_A34_v2_e2e/results" --max_clips 500 --batch_size 4 --num_workers 2
cat "$BASE/avh_sup/outputs_A34_v2_e2e/results/eval_e2e_domain.json" 2>/dev/null
