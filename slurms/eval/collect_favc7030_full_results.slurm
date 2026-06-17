#!/bin/bash
#SBATCH --job-name=collect_7030_full
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/collect_7030_full_%j.out
#SBATCH --error=logs/collect_7030_full_%j.err

set -euo pipefail

BASE=/data/projects/punim2637/nnliang/AVH-Align
PYTHON=/data/projects/punim2637/nnliang/avh_env/bin/python

cd "$BASE"

module purge
module load Miniconda3/23.10.0-1

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONUNBUFFERED=1

$PYTHON submission_prep/collect_favc7030_full_results.py
