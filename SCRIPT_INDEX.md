# Script Index

This file gives a lightweight map of the most important scripts and config areas in this repository.

## Top-Level Training and Evaluation Scripts

- `train_A*.slurm`: submit training jobs for the A-series experiments.
- `eval_auc_A*.slurm`: run AUC evaluation for trained checkpoints.
- `eval_domain_A*.slurm`: run domain and label probe evaluation for trained checkpoints.
- `eval_domain_pred_A*.slurm`: run domain predictability evaluation for older experiments.
- `eval_label_probe*.slurm`: run label probe experiments on selected checkpoints.
- `eval_perturb*.slurm`: run perturbation robustness evaluation.
- `train_B*.slurm`: submit the modal-balance B-series experiments.
- `eval_favc_B*.slurm`: evaluate B-series checkpoints on FakeAVCeleb.

## Core Python Modules

- `avh_sup/train_e2e_*.py`: end-to-end training entry points for the newer A-series experiments.
- `avh_sup/eval_auc_*.py`: AUC evaluation entry points for newer end-to-end experiments.
- `avh_sup/eval_domain_*.py`: domain and label probe evaluation entry points.
- `avh_sup/mlp_causal_ablation.py`: causal/spurious backbone used by many A-series ablations.
- `avh_sup/mlp_fcd_a6.py`: FCD-style backbone used in the A6 line.
- `avh_sup/mlp_sad_a8.py` and `avh_sup/mlp_sad_a9.py`: synchrony/appearance decomposition variants.
- `avh_sup/mlp_causal_modal.py`: modal-balance extension for the B-series experiments.
- `avh_sup/datasets.py`: shared dataset definitions and data loading helpers.

## Config Directories

- `avh_sup/configs/A*.yaml`: experiment configs for the A-series.
- `avh_sup/configs/B*.yaml`: modal-balance configs for the B-series.

## Data Preparation and Utilities

- `deepfake_preprocess.py`: preprocessing entry point for raw videos.
- `deepfake_feature_extraction.py`: feature extraction pipeline for processed videos.
- `generate_favc_metadata.py` and `regenerate_favc_metadata.py`: FakeAVCeleb metadata generation helpers.
- `batch_download_extract_features.py` and related batch scripts: large-scale download and feature extraction utilities.
- `check_favc_progress.sh`: quick progress summary for FakeAVCeleb processing.

## Notes

- Generated outputs under `avh_sup/outputs_*` are experiment artifacts and are not part of the hand-maintained code structure.
- When adding a new experiment, try to keep the naming consistent across:
  - config file
  - training script
  - evaluation script
  - slurm launcher
