"""
Per-Branch Label Probe Under Perturbation
==========================================
Evaluates how much fake/real label information is retained in each branch
(Z_c and Z_s) of a model when the input features are perturbed.

This is the KEY analysis experiment for Karen's request:
  Show that FCD (A6) forces branch specialisation — Z_c retains label signal
  under audio masking (because visual unique channel survives), while Z_s
  collapses. Binary (A2) Z_c also collapses under audio masking because it
  is audio-dominated.

Protocol:
  For each perturbation type and level:
    1. Load AV1M val clips as full [T, D] sequences (NOT pre-mean-pooled).
    2. Apply perturbation to raw features.
    3. Run model._encode() → Z_c [B,T,d_c], Z_s [B,T,d_s].
    4. Mean-pool over T → fixed-size vectors.
    5. 5-fold stratified logistic regression probe on Z_c → label AUC.
    6. 5-fold stratified logistic regression probe on Z_s → label AUC.

Perturbation types covered:
  audio_masking   — zero out fraction of audio frames
  visual_masking  — zero out fraction of visual frames
  audio_noise     — Gaussian noise on audio features  (σ ∈ {0,0.05,0.1,0.2,0.5,1.0})
  visual_noise    — Gaussian noise on visual features (σ ∈ {0,0.05,0.1,0.2,0.5,1.0})

Output JSON structure:
  {
    "audio_masking": {
      "0.00": {"zc_auc": ..., "zc_std": ..., "zs_auc": ..., "zs_std": ...},
      "0.10": {...}, ...
    },
    "visual_masking": { ... }
  }

Usage:
  python eval_branch_probe_perturb.py \\
      --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
      --config     configs/A6.yaml \\
      --av1m_root  ../data/avh_features/val \\
      --av1m_csv   csv_metadata/av1m/val_labels.csv \\
      --output_dir outputs_A6/results \\
      --device     cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mlp_causal_ablation import AVH_Causal_Ablation
from mlp_causal_mi import AVH_Causal_MI
from mlp_fcd import AVH_FCD
from mlp_fcd_a6 import AVH_FCD_A6
from mlp_fcd_a7 import AVH_FCD_A7
from mlp_sad_a8 import AVH_SAD_A8


# ── Perturbation schedule ──────────────────────────────────────────────────────
PERTURB_CONFIG = {
    "audio_masking":  [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "visual_masking": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "audio_noise":    [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "visual_noise":   [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
}


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_from_ckpt(ckpt_path: str):
    """Auto-detect model class from checkpoint hyper_parameters and load it."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]["config"]
    hp   = cfg.get("model_hparams", {})
    if "sync_dim" in hp:
        model = AVH_SAD_A8(config=cfg)
    elif "lambda_ladv" in hp:
        model = AVH_FCD_A7(config=cfg)
    elif "lambda_dadv" in hp:
        model = AVH_FCD_A6(config=cfg)
    elif "syn_dim" in hp:
        model = AVH_FCD(config=cfg)
    elif "lambda_mi_spu" in hp or "lambda_club" in hp:
        model = AVH_Causal_MI(config=cfg)
    else:
        model = AVH_Causal_Ablation(config=cfg)
    model_state = model.state_dict()
    filtered    = {k: v for k, v in ckpt["state_dict"].items()
                   if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    return model, cfg


# ── Sequence dataset (returns full [T,D], no pre-pooling) ─────────────────────

class SequenceDataset(Dataset):
    """Returns (video [T,1024], audio [T,1024], label) — full time dimension."""
    def __init__(self, items: list, apply_l2: bool = True):
        self.items    = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data  = np.load(path, allow_pickle=True)
        video = data["visual"].astype(np.float32)   # [T, 1024]
        audio = data["audio"].astype(np.float32)    # [T, 1024]
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(video), torch.tensor(audio), label


def _seq_collate(batch):
    """batch_size=1 collate: add batch dimension."""
    video, audio, label = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label])


def _load_av1m_items(av1m_root: str, csv_path: str) -> list:
    items = []
    with open(csv_path) as f:
        header    = f.readline().strip().split(",")
        path_col  = header.index("path")
        label_col = header.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rel   = parts[path_col]
            label = int(parts[label_col])
            npz   = os.path.join(av1m_root, rel.replace(".mp4", ".npz"))
            if os.path.exists(npz):
                items.append((npz, label))
    return items


# ── Perturbation helpers ───────────────────────────────────────────────────────

def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def apply_perturbation(video: torch.Tensor, audio: torch.Tensor,
                       perturb_type: str, level: float):
    """
    Apply a perturbation to raw (pre-L2) features and return l2-normalised
    (video, audio) ready for model._encode.

    video/audio shape: [B, T, D]
    level=0.0 returns clean normalised features.
    """
    if level == 0.0:
        return _l2_norm(video), _l2_norm(audio)

    if perturb_type == "audio_masking":
        B, T, D = audio.shape
        mask  = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=audio.device)
        )
        audio = audio * mask
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "visual_masking":
        B, T, D = video.shape
        mask  = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=video.device)
        )
        video = video * mask
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "audio_noise":
        audio = audio + torch.randn_like(audio) * level
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "visual_noise":
        video = video + torch.randn_like(video) * level
        return _l2_norm(video), _l2_norm(audio)

    else:
        raise ValueError(f"Unknown perturb_type: {perturb_type!r}")


# ── Feature extraction under perturbation ─────────────────────────────────────

@torch.no_grad()
def extract_perturbed(model, dataloader: DataLoader,
                      perturb_type: str, level: float,
                      device: torch.device, seed: int = 42):
    """
    For each clip: apply perturbation → _encode → mean-pool Z_c and Z_s.
    Returns (zc [N, d_c], zs [N, d_s], labels [N]).
    """
    model.eval()
    model.to(device)
    torch.manual_seed(seed)

    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc=f"  {perturb_type} lv={level:.2f}",
                      leave=False):
        video_raw, audio_raw, labels = batch
        video_raw = video_raw.to(device)   # [1, T, 1024]
        audio_raw = audio_raw.to(device)

        video_p, audio_p = apply_perturbation(
            video_raw, audio_raw, perturb_type, level
        )

        # _encode returns (causal_repr, spurious_repr, ...) for all model types
        causal_repr, spurious_repr, *_ = model._encode(video_p, audio_p)

        # Mean-pool over time
        zc = causal_repr.mean(dim=1).cpu().numpy()    # [1, d_c]
        zs = spurious_repr.mean(dim=1).cpu().numpy()  # [1, d_s]
        all_zc.append(zc)
        all_zs.append(zs)
        all_labels.extend(labels.tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Logistic probe ────────────────────────────────────────────────────────────

def probe_cv(features: np.ndarray, labels: np.ndarray,
             n_splits: int = 5, seed: int = 42):
    """5-fold stratified CV logistic regression probe. Returns (mean_auc, std_auc)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr_idx, te_idx in skf.split(features, labels):
        scaler = StandardScaler().fit(features[tr_idx])
        clf    = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                    random_state=seed)
        clf.fit(scaler.transform(features[tr_idx]), labels[tr_idx])
        proba  = clf.predict_proba(scaler.transform(features[te_idx]))[:, 1]
        fold_aucs.append(roc_auc_score(labels[te_idx], proba))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-branch label probe under perturbation (A2 vs A5/A6).")
    parser.add_argument("--ckpt_path",  required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config",     required=True,
                        help="YAML config file (e.g. configs/A6.yaml)")
    parser.add_argument("--av1m_root",  required=True,
                        help="Root of AV1M val features (contains .npz files)")
    parser.add_argument("--av1m_csv",   required=True,
                        help="AV1M val labels CSV (path, label columns)")
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config   = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    model, _ = _load_model_from_ckpt(args.ckpt_path)
    device   = torch.device(args.device)
    print(f"Model class : {model.__class__.__name__}")
    print(f"Device      : {device}")

    items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"AV1M val    : {len(items)} clips  "
          f"(real={sum(1 for _,l in items if l==0)}, "
          f"fake={sum(1 for _,l in items if l==1)})")

    ds     = SequenceDataset(items, apply_l2=apply_l2)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=_seq_collate)

    results = {}

    for perturb_type, levels in PERTURB_CONFIG.items():
        print(f"\n── {perturb_type} ──")
        results[perturb_type] = {}

        for level in levels:
            zc, zs, labels = extract_perturbed(model, loader, perturb_type,
                                               level, device, seed=args.seed)
            zc_auc, zc_std = probe_cv(zc, labels, args.n_folds, args.seed)
            zs_auc, zs_std = probe_cv(zs, labels, args.n_folds, args.seed)

            key = f"{level:.2f}"
            results[perturb_type][key] = {
                "zc_auc": zc_auc, "zc_std": zc_std,
                "zs_auc": zs_auc, "zs_std": zs_std,
            }
            print(f"  level={level:.2f}  "
                  f"Z_c={zc_auc:.4f}±{zc_std:.4f}  "
                  f"Z_s={zs_auc:.4f}±{zs_std:.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "branch_probe_perturb.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out_path}")

    return results


if __name__ == "__main__":
    main()
