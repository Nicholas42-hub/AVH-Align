# TRIROUTE — Supplementary Code

This package contains the core runnable code for **TRIROUTE**, the unsupervised
Audio-Visual deepfake detection method described in the NeurIPS 2026 submission.

---

## Method Overview

TRIROUTE decomposes frozen AV-HuBERT features into three components per modality:

| Symbol | Name | Role |
|--------|------|------|
| `s_v`, `s_a` | Shared / synergistic | Cross-modal AV consistency cue |
| `u_v`, `u_a` | Unique manipulation | Modality-specific deepfake artefacts |
| `r_v`, `r_a` | Residual / redundant | Domain shortcuts (excluded at inference) |

The detection head uses only `Z_task = cat(s_v, u_v, u_Δ, s_a, u_a)` where
`u_Δ` is a cross-modal inconsistency term.  Training is **fully unsupervised**:
the model learns audio-visual synchrony from real unlabelled clips; no fake
labels are needed during training.

---

## File Structure

```
triroute_code/
├── README.md                          # this file
├── requirements.txt                   # Python dependencies
├── train_eval_triroute_unsup_sync.py  # main train + eval entry-point
├── mlp_fcd.py                         # CCD / URD / SDA / Sinkhorn modules
├── mlp_fcd_triroute.py                 # CrossModalInconsistency module
├── avhubert_wrapper.py                # AV-HuBERT feature extractor wrapper
├── datasets.py                        # Dataset classes (AV1M, FakeAVCeleb)
├── configs/
│   └── triroute_paper.yaml            # Hyperparameters used in paper experiments
└── run_triroute_paper.slurm           # Example Slurm job script
```

---

## Prerequisites

1. **AV-HuBERT** — download the `self_large_vox_433h.pt` checkpoint from the
   [AV-HuBERT model zoo](https://github.com/facebookresearch/av_hubert).
   Follow the fairseq-based installation in `av_hubert/README.md`.

2. **Pre-extracted features** — run `deepfake_feature_extraction.py` (repo root)
   to extract `visual` and `audio` AV-HuBERT features as `.npz` files for
   each video clip.  The resulting directory structure expected by the
   datasets is:
   ```
   <features_root>/
     train/<speaker_id>/<clip_id>.npz   # keys: "visual" [T,1024], "audio" [T,1024]
     val/...
     test/...
   ```

3. **CSV metadata** — each split needs a `train_labels.csv`, `val_labels.csv`,
   and `test_labels.csv` with at minimum a `path` column (relative .mp4 path)
   and a `label` column (0=real, 1=fake).

---

## Training

```bash
python train_eval_triroute_unsup_sync.py \
    --output_dir  outputs/triroute_seed43 \
    --av1m_features   /path/to/av1m_features \
    --av1m_csv_root   /path/to/av1m_csv \
    --favc_features   /path/to/favc_features \
    --favc_csv_root   /path/to/favc_csv \
    --epochs 100 \
    --frames_per_clip 4 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 43
```

The script trains on **real unlabelled clips only** (audio-visual sync
objective) and periodically evaluates cross-domain AUC on FakeAVCeleb.

---

## Evaluation Only

```bash
python train_eval_triroute_unsup_sync.py \
    --eval_only \
    --ckpt  outputs/triroute_seed43/triroute_unsup_sync_best_auc.pt \
    --output_dir outputs/triroute_seed43 \
    --favc_features  /path/to/favc_features \
    --favc_csv_root  /path/to/favc_csv \
    --av1m_features  /path/to/av1m_features \
    --av1m_csv_root  /path/to/av1m_csv
```

Results are written to `outputs/triroute_seed43/results_favc_7030/eval_results.txt`
and `outputs/triroute_seed43/results_av1m_val/`.

---

## Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--tau` | 15 | Temporal context window (±τ frames) for sync task |
| `--syn_dim` | 256 | Synergistic subspace dimension |
| `--spec_dim` | 256 | Modality-specific subspace dimension |
| `--lambda_sda` | 0.5 | Synergistic Distribution Alignment weight |
| `--lambda_dis` | 1.0 | Modality discrimination (residual) weight |
| `--lambda_orth` | 0.1 | Orthogonality penalty weight |
| `--lambda_delta_sparse` | 0.01 | Cross-modal inconsistency sparsity weight |
| `--epochs` | 100 | Training epochs (paper: 100) |
| `--frames_per_clip` | 4 | Samples per clip per epoch |

---

## Architecture Modules (`mlp_fcd.py`)

| Class | Role |
|-------|------|
| `CCD` | Causality Components Decomposition — splits `z` into `(s, h)` |
| `URD` | Unique-Redundant Decomposition — splits `h` into `(u, r = h − u)` |
| `SDA` | Shared projection for Synergistic Distribution Alignment |
| `_sinkhorn_divergence` | Debiased Sinkhorn divergence (pure PyTorch, no external OT library) |

Cross-modal inconsistency `u_Δ` is computed by `CrossModalInconsistency`
in `mlp_fcd_triroute.py` via cross-attention between video and audio features.

---

## Citation

If you use this code, please cite our paper:

```
@inproceedings{triroute2026,
  title  = {TRIROUTE: Unsupervised Audio-Visual Deepfake Detection via
             Three-Way Feature Routing},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year   = {2026},
}
```
