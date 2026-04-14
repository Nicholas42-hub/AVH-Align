"""
Modality-of-Origin Probe
=========================
Probes whether Z_c can identify *which modality was manipulated*:
  class 0 = FakeVideo-RealAudio  (video synthesised, audio real)
  class 1 = RealVideo-FakeAudio  (audio synthesised, video real)

This tests whether the causal branch encodes BOTH modality-specific fake
signals in a balanced way, or is skewed toward one modality.

Hypotheses:
  Z_c^FCD (A5/A6):
    High probe AUC — unique channels u_v and u_a separately capture
    video vs audio synthesis artefacts. Z_c therefore has enough
    modality-specific information to distinguish FV-RA from RV-FA.

  Z_c^binary (A2):
    Audio-dominated Z_c makes RV-FA easy to detect (audio fake signal)
    but FV-RA harder (Z_c audio channel looks "real" for video-only fakes).
    Results in a biased or lower probe AUC.

Also evaluates Z_s as sanity check:
  Z_s^FCD redundant channels are shared correlates —
  they should carry less modality-specific distinction.

Data: FakeAVCeleb test — FV-RA clips (video fake) vs RV-FA clips (audio fake).
  N ~ 522 FV-RA + 52 RV-FA from the N=1114 balanced test subset.
  Note: class imbalance is expected — the probe uses AUC (rank-based).

Usage:
  python eval_modality_origin_probe.py \\
      --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
      --config     configs/A6.yaml \\
      --favc_root  ../data/favc_features \\
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


# FAVC category directory names
_CATEGORY_FVRA = "FakeVideo-RealAudio"   # video fake only → class 0
_CATEGORY_RVFA = "RealVideo-FakeAudio"   # audio fake only → class 1


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


# ── Dataset ────────────────────────────────────────────────────────────────────

class MeanPoolDataset(Dataset):
    """Returns (mean-pooled video [D], mean-pooled audio [D], label)."""
    def __init__(self, items: list, apply_l2: bool = True):
        self.items    = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data  = np.load(path, allow_pickle=True)
        video = data["visual"].astype(np.float32)
        audio = data["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        # Mean-pool over time → fixed-size vectors
        return torch.tensor(video.mean(0)), torch.tensor(audio.mean(0)), label


def _load_favc_items_by_category(favc_root: str, categories: dict) -> list:
    """
    categories: {category_dir_name: label_int}
    Returns list of (npz_path, label).
    """
    items = []
    for cat, label in categories.items():
        cat_path = os.path.join(favc_root, cat)
        if not os.path.isdir(cat_path):
            print(f"  WARNING: category dir not found: {cat_path}")
            continue
        for dirpath, _, fnames in os.walk(cat_path):
            for fname in sorted(fnames):
                if fname.endswith(".npz"):
                    items.append((os.path.join(dirpath, fname), label))
    return items


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_repr(model, dataloader: DataLoader, device: torch.device):
    """Extract mean-pooled Z_c and Z_s for all clips. Returns (zc, zs, labels)."""
    model.eval()
    model.to(device)
    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        video, audio, labels = batch
        video = video.unsqueeze(1).to(device)   # [B, 1, D] — single‑frame equivalent
        audio = audio.unsqueeze(1).to(device)

        causal_repr, spurious_repr, *_ = model._encode(video, audio)
        all_zc.append(causal_repr.squeeze(1).cpu().numpy())
        all_zs.append(spurious_repr.squeeze(1).cpu().numpy())
        all_labels.extend(labels.tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Probe ──────────────────────────────────────────────────────────────────────

def probe_cv(features: np.ndarray, labels: np.ndarray,
             n_splits: int = 5, seed: int = 42):
    """5-fold stratified CV logistic regression. Returns (mean_auc, std_auc)."""
    if len(set(labels)) < 2:
        return None, None
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
        description="Modality-of-origin probe: FV-RA (class 0) vs RV-FA (class 1).")
    parser.add_argument("--ckpt_path",  required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--config",     required=True,
                        help="YAML config file (e.g. configs/A6.yaml)")
    parser.add_argument("--favc_root",  required=True,
                        help="Root of FakeAVCeleb features (category subdirs)")
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

    # FV-RA = class 0 (video fake only), RV-FA = class 1 (audio fake only)
    categories = {_CATEGORY_FVRA: 0, _CATEGORY_RVFA: 1}
    items = _load_favc_items_by_category(args.favc_root, categories)

    n_fvra = sum(1 for _, l in items if l == 0)
    n_rvfa = sum(1 for _, l in items if l == 1)
    print(f"FV-RA clips : {n_fvra}")
    print(f"RV-FA clips : {n_rvfa}")
    print(f"Total       : {len(items)}")

    if len(items) == 0:
        print("No clips found — check favc_root and category directory names.")
        return

    ds     = MeanPoolDataset(items, apply_l2=apply_l2)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    zc, zs, labels = extract_repr(model, loader, device)

    zc_auc, zc_std = probe_cv(zc, labels, args.n_folds, args.seed)
    zs_auc, zs_std = probe_cv(zs, labels, args.n_folds, args.seed)

    print(f"\nModality-of-origin probe (FV-RA=0 vs RV-FA=1):")
    print(f"  Z_c AUC = {zc_auc:.4f} ± {zc_std:.4f}")
    print(f"  Z_s AUC = {zs_auc:.4f} ± {zs_std:.4f}")
    print("  (AUC > 0.5 = can predict which modality was faked from this branch)")

    results = {
        "task":     "modality_origin (FV-RA=0 vs RV-FA=1)",
        "n_fvra":   n_fvra,
        "n_rvfa":   n_rvfa,
        "zc_auc":   zc_auc,
        "zc_std":   zc_std,
        "zs_auc":   zs_auc,
        "zs_std":   zs_std,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "modality_origin_probe.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {out_path}")

    return results


if __name__ == "__main__":
    main()
