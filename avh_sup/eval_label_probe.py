"""
Label Predictability Probe
============================
Trains a logistic regression probe to predict fake/real labels separately
from Z_c and Z_s, on both in-domain (AV1M val) and OOD (FakeAVCeleb) data.

This verifies that disentanglement is working as intended:
  - Z_c should retain label-predictive signal (A-V sync = informative for fake detection)
  - Z_s should carry label signal in-domain (it absorbs shortcut features)
      BUT fail to generalise OOD (shortcut breaks across domains)

Two evaluation modes
---------------------
1. In-domain  (AV1M val, 5-fold stratified CV):
     Reports Z_c AUC and Z_s AUC on the training domain.
     Expected: both high for A0 (no disentanglement pressure on Z_s).
               A1/A2: Z_s may drop as adversarial loss penalises label info in Z_s.

2. OOD transfer (train probe on AV1M val → evaluate on FakeAVCeleb test):
     Trains probe once on all AV1M val features, evaluates on FAVC.
     Expected: Z_c OOD AUC > Z_s OOD AUC (A1/A2) — the spurious shortcut fails OOD.
     This is the critical evidence for the paper's central claim.

Domain classification check (bonus)
-------------------------------------
Also probes domain predictability (AV1M=0 vs FAVC=1) from Z_c and Z_s.
Cross-entropy from domain probe ≈ measure of domain information in each branch.
Expected for A1/A2: Z_s carries domain info (shortcut ≈ dataset-specific artefact).
Note: raw AV-HuBERT features are ~99% domain-separable; Z_c/Z_s deltas vs raw
      are the meaningful signal.

A0 note
--------
A0 only trains full_head(Z_c + Z_s) — neither causal_head nor spurious_head
is directly supervised.  The logistic regression probe here operates on the
*frozen* Z_c and Z_s representations, not on trained classifier heads.
This is the correct way to evaluate A0 without changing its training objective.

Usage:
    python eval_label_probe.py \\
        --ckpt_path   outputs_A0/ckpts/model-epoch=09.ckpt \\
        --config      configs/A0.yaml \\
        --av1m_root   ../data/avh_features/val \\
        --av1m_csv    csv_metadata/av1m/val_labels.csv \\
        --favc_root   ../data/favc_features \\
        --output_dir  outputs_A0/results
"""

import argparse
import json
import os
import random
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


def _load_model_from_ckpt(ckpt_path: str):
    """Auto-detect model class from checkpoint hyper_parameters and load it."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]["config"]
    hp   = cfg.get("model_hparams", {})
    # A8 uses sync_dim (SAD); A7 uses lambda_ladv; A6 uses lambda_dadv; A5 uses syn_dim
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
    model_state    = model.state_dict()
    filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    return model, cfg

# FAVC categories: "real" = RealVideo-RealAudio only; everything else = "fake"
_FAVC_REAL_CATEGORY = "RealVideo-RealAudio"
_FAVC_FAKE_CATEGORIES = {"FakeVideo-RealAudio", "FakeVideo-FakeAudio", "RealVideo-FakeAudio"}


# ── Dataset helpers ───────────────────────────────────────────────────────────

class LabelDataset(Dataset):
    """
    (visual_feat, audio_feat, label) dataset.
    Items: list of (abs_npz_path, int_label).
    Features are mean-pooled over time before being returned.
    """
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
        video = video.mean(axis=0)
        audio = audio.mean(axis=0)
        return torch.tensor(video), torch.tensor(audio), label


def _load_av1m_items(av1m_root: str, csv_path: str) -> list:
    """
    Returns list of (abs_npz_path, label) for AV1M val using the labels CSV.
    CSV format: path (relative, mp4 extension), label (0=real, 1=fake).
    """
    items = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        path_col = header.index("path")
        label_col = header.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rel_path = parts[path_col]
            label    = int(parts[label_col])
            npz_path = os.path.join(av1m_root, rel_path.replace(".mp4", ".npz"))
            if os.path.exists(npz_path):
                items.append((npz_path, label))
    return items


def _load_favc_items(favc_root: str) -> list:
    """
    Returns list of (abs_npz_path, label) for FakeAVCeleb.
    label: 0 = RealVideo-RealAudio, 1 = all fake categories.
    """
    items = []
    for category in os.listdir(favc_root):
        cat_path = os.path.join(favc_root, category)
        if not os.path.isdir(cat_path):
            continue
        if category == _FAVC_REAL_CATEGORY:
            label = 0
        elif category in _FAVC_FAKE_CATEGORIES:
            label = 1
        else:
            continue  # unknown category
        for dirpath, _, filenames in os.walk(cat_path):
            for fname in sorted(filenames):
                if fname.endswith(".npz"):
                    items.append((os.path.join(dirpath, fname), label))
    return items


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_repr(model: AVH_Causal_Ablation, dataloader: DataLoader,
                 device: torch.device):
    """Extract Z_c and Z_s for every clip. Returns (zc, zs, labels)."""
    model.eval()
    model.to(device)
    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        video_feats, audio_feats, labels = batch
        video_feats = video_feats.unsqueeze(1).to(device)   # [B,1,D]
        audio_feats = audio_feats.unsqueeze(1).to(device)

        causal_repr, spurious_repr, *_ = model._encode(video_feats, audio_feats)
        all_zc.append(causal_repr.squeeze(1).cpu().numpy())
        all_zs.append(spurious_repr.squeeze(1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else labels.tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


def extract_raw(dataloader: DataLoader):
    """Baseline: concatenate raw AV-HuBERT features (no model)."""
    all_raw, all_labels = [], []
    for batch in tqdm(dataloader, desc="Raw baseline", leave=False):
        video_feats, audio_feats, labels = batch
        raw = torch.cat([video_feats, audio_feats], dim=-1)
        all_raw.append(raw.numpy())
        all_labels.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else labels.tolist())
    return (np.concatenate(all_raw, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Logistic regression probe ─────────────────────────────────────────────────

def probe_cv(features: np.ndarray, labels: np.ndarray,
             n_splits: int = 5, seed: int = 42):
    """5-fold stratified CV logistic regression probe. Returns (mean, std, folds)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, test_idx in skf.split(features, labels):
        X_tr, X_te = features[train_idx], features[test_idx]
        y_tr, y_te = labels[train_idx],   labels[test_idx]
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        fold_aucs.append(roc_auc_score(y_te, proba))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), [float(a) for a in fold_aucs]


def probe_transfer(train_features: np.ndarray, train_labels: np.ndarray,
                   test_features: np.ndarray, test_labels: np.ndarray,
                   seed: int = 42):
    """
    Train probe on source domain, evaluate on target domain.
    Returns single AUC (no CV — full train set used for best generalisation).
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_features)
    X_te = scaler.transform(test_features)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             random_state=seed)
    clf.fit(X_tr, train_labels)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(test_labels, proba))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Label predictability probe: in-domain CV + OOD transfer.")
    parser.add_argument("--ckpt_path",    required=True)
    parser.add_argument("--config",       required=True)
    parser.add_argument("--av1m_root",    required=True,
                        help="Root of AV1M val features")
    parser.add_argument("--av1m_csv",     required=True,
                        help="AV1M val labels CSV (path, label columns)")
    parser.add_argument("--favc_root",    required=True,
                        help="Root of FakeAVCeleb features (category/id/clip.npz)")
    parser.add_argument("--n_folds",      type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--max_av1m",     type=int, default=None,
                        help="Subsample AV1M to this many clips (default: all)")
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load config & model ──────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)
    model, _loaded_config = _load_model_from_ckpt(args.ckpt_path)
    print(f"  Model class: {model.__class__.__name__}")
    device   = torch.device(args.device)

    # ── AV1M val items ───────────────────────────────────────────────────────
    print(f"\nLoading AV1M val from: {args.av1m_root}")
    av1m_items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"  Found {len(av1m_items)} clips  "
          f"(real={sum(1 for _,l in av1m_items if l==0)}, "
          f"fake={sum(1 for _,l in av1m_items if l==1)})")
    if args.max_av1m and len(av1m_items) > args.max_av1m:
        av1m_items = rng.sample(av1m_items, args.max_av1m)
        print(f"  Subsampled to {len(av1m_items)}")

    # ── FAVC items ───────────────────────────────────────────────────────────
    print(f"\nLoading FakeAVCeleb from: {args.favc_root}")
    favc_items = _load_favc_items(args.favc_root)
    print(f"  Found {len(favc_items)} clips  "
          f"(real={sum(1 for _,l in favc_items if l==0)}, "
          f"fake={sum(1 for _,l in favc_items if l==1)})")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    av1m_ds = LabelDataset(av1m_items, apply_l2=apply_l2)
    favc_ds = LabelDataset(favc_items, apply_l2=apply_l2)
    av1m_dl = DataLoader(av1m_ds, batch_size=64, shuffle=False, num_workers=4)
    favc_dl = DataLoader(favc_ds, batch_size=64, shuffle=False, num_workers=4)

    # ── Extract representations ───────────────────────────────────────────────
    print("\nExtracting AV1M Z_c / Z_s …")
    av1m_zc, av1m_zs, av1m_labels = extract_repr(model, av1m_dl, device)
    print("Extracting FAVC Z_c / Z_s …")
    favc_zc, favc_zs, favc_labels = extract_repr(model, favc_dl, device)

    print("Extracting raw baselines …")
    av1m_raw, av1m_labels_raw = extract_raw(av1m_dl)
    favc_raw, favc_labels_raw = extract_raw(favc_dl)

    assert (av1m_labels == av1m_labels_raw).all()
    assert (favc_labels == favc_labels_raw).all()

    # ── In-domain probe (AV1M 5-fold CV) ────────────────────────────────────
    print("\n── In-domain probe (AV1M val, 5-fold CV) ──────────────────────────")
    raw_id_mean,  raw_id_std,  raw_id_folds  = probe_cv(av1m_raw,  av1m_labels, args.n_folds, args.seed)
    zc_id_mean,   zc_id_std,   zc_id_folds   = probe_cv(av1m_zc,   av1m_labels, args.n_folds, args.seed)
    zs_id_mean,   zs_id_std,   zs_id_folds   = probe_cv(av1m_zs,   av1m_labels, args.n_folds, args.seed)

    print(f"  Raw AV-HuBERT  AUC = {raw_id_mean:.4f} ± {raw_id_std:.4f}  (baseline)")
    print(f"  Z_c (causal)   AUC = {zc_id_mean:.4f} ± {zc_id_std:.4f}  Δraw={zc_id_mean - raw_id_mean:+.4f}")
    print(f"  Z_s (spurious) AUC = {zs_id_mean:.4f} ± {zs_id_std:.4f}  Δraw={zs_id_mean - raw_id_mean:+.4f}")

    # ── OOD transfer probe (train AV1M → test FAVC) ──────────────────────────
    print("\n── OOD transfer probe (train on AV1M val → test on FAVC) ──────────")
    raw_ood  = probe_transfer(av1m_raw,  av1m_labels, favc_raw,  favc_labels, args.seed)
    zc_ood   = probe_transfer(av1m_zc,   av1m_labels, favc_zc,   favc_labels, args.seed)
    zs_ood   = probe_transfer(av1m_zs,   av1m_labels, favc_zs,   favc_labels, args.seed)

    print(f"  Raw AV-HuBERT  OOD AUC = {raw_ood:.4f}  (baseline)")
    print(f"  Z_c (causal)   OOD AUC = {zc_ood:.4f}   Δraw={zc_ood - raw_ood:+.4f}")
    print(f"  Z_s (spurious) OOD AUC = {zs_ood:.4f}   Δraw={zs_ood - raw_ood:+.4f}")

    # ── Domain probe (AV1M vs FAVC, for domain CE check) ────────────────────
    print("\n── Domain probe (AV1M=0 vs FAVC=1, 5-fold CV on balanced sample) ──")
    n_dom = min(len(av1m_items), len(favc_items))
    dom_rng = np.random.default_rng(args.seed)
    av1m_idx = dom_rng.choice(len(av1m_labels), n_dom, replace=False)
    favc_idx = dom_rng.choice(len(favc_labels), n_dom, replace=False)
    dom_raw  = np.concatenate([av1m_raw[av1m_idx], favc_raw[favc_idx]], axis=0)
    dom_zc   = np.concatenate([av1m_zc[av1m_idx],  favc_zc[favc_idx]],  axis=0)
    dom_zs   = np.concatenate([av1m_zs[av1m_idx],  favc_zs[favc_idx]],  axis=0)
    dom_labels = np.array([0]*n_dom + [1]*n_dom, dtype=np.int32)

    raw_dom_mean, raw_dom_std, _ = probe_cv(dom_raw, dom_labels, args.n_folds, args.seed)
    zc_dom_mean,  zc_dom_std,  _ = probe_cv(dom_zc,  dom_labels, args.n_folds, args.seed)
    zs_dom_mean,  zs_dom_std,  _ = probe_cv(dom_zs,  dom_labels, args.n_folds, args.seed)

    print(f"  Raw AV-HuBERT  domain AUC = {raw_dom_mean:.4f} ± {raw_dom_std:.4f}  (baseline)")
    print(f"  Z_c (causal)   domain AUC = {zc_dom_mean:.4f} ± {zc_dom_std:.4f}  Δraw={zc_dom_mean - raw_dom_mean:+.4f}")
    print(f"  Z_s (spurious) domain AUC = {zs_dom_mean:.4f} ± {zs_dom_std:.4f}  Δraw={zs_dom_mean - raw_dom_mean:+.4f}")
    print(f"\n  NOTE: Raw≈99% means AV-HuBERT itself encodes domain; Δraw is the")
    print(f"        only model-attributable signal.")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "ckpt_path":         args.ckpt_path,
        "n_av1m_clips":      len(av1m_labels),
        "n_favc_clips":      len(favc_labels),
        "n_folds":           args.n_folds,
        "seed":              args.seed,
        "in_domain": {
            "description": "AV1M val fake/real, 5-fold stratified CV",
            "raw":      {"auc_mean": raw_id_mean, "auc_std": raw_id_std,  "folds": raw_id_folds},
            "zc":       {"auc_mean": zc_id_mean,  "auc_std": zc_id_std,   "folds": zc_id_folds,  "delta_raw": round(zc_id_mean - raw_id_mean, 6)},
            "zs":       {"auc_mean": zs_id_mean,  "auc_std": zs_id_std,   "folds": zs_id_folds,  "delta_raw": round(zs_id_mean - raw_id_mean, 6)},
        },
        "ood_transfer": {
            "description": "Probe trained on AV1M val, evaluated on FakeAVCeleb",
            "raw":      {"auc": raw_ood},
            "zc":       {"auc": zc_ood,  "delta_raw": round(zc_ood - raw_ood, 6)},
            "zs":       {"auc": zs_ood,  "delta_raw": round(zs_ood - raw_ood, 6)},
        },
        "domain_probe": {
            "description": "AV1M (label=0) vs FakeAVCeleb (label=1), balanced, 5-fold CV",
            "n_per_domain":  n_dom,
            "raw":      {"auc_mean": raw_dom_mean, "auc_std": raw_dom_std},
            "zc":       {"auc_mean": zc_dom_mean,  "auc_std": zc_dom_std,  "delta_raw": round(zc_dom_mean - raw_dom_mean, 6)},
            "zs":       {"auc_mean": zs_dom_mean,  "auc_std": zs_dom_std,  "delta_raw": round(zs_dom_mean - raw_dom_mean, 6)},
        },
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "label_probe.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
