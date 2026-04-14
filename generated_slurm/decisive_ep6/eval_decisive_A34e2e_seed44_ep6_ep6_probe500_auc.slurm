#!/bin/bash
#SBATCH --job-name=ev_decisive_A34e2e_44_ep6_ep6_probe500_auc
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.out
#SBATCH --error=/data/projects/punim2637/nnliang/AVH-Align/logs/%x_%j.err

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

mkdir -p /data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_decisive_A34e2e_seed44/results_ep6_probe500_auc /data/projects/punim2637/nnliang/AVH-Align/logs
cd /data/projects/punim2637/nnliang/AVH-Align/avh_sup
CKPT=$(ls "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_decisive_A34e2e_seed44/ckpts"/model-epoch=05-*.ckpt 2>/dev/null | head -1)
CKPT="${CKPT:-/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_decisive_A34e2e_seed44/ckpts/last.ckpt}"
echo ">>> ev_decisive_A34e2e_44_ep6_ep6_probe500_auc using checkpoint: $CKPT"
$PYTHON eval_e2e_auc.py \
    --model_module train_e2e_a34 \
    --model_class E2EModelA34 \
    --use_fullpath_csv \
    --ckpt "$CKPT" \
    --config "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/configs/decisive/decisive_A34e2e_seed44.yaml" \
    --av1m_raw /scratch/punim2637/nnliang/avd1m_preprocessed/val \
    --av1m_csv "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m_e2e" \
    --favc_raw /scratch/punim2637/nnliang/favc_preprocessed \
    --output_dir "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/outputs_decisive_A34e2e_seed44/results_ep6_probe500_auc" \
    --max_clips 500 \
    --batch_size 4 \
    --num_workers 2
