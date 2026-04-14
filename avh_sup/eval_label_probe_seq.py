"""
Sequence-Based Label & Domain Probe
=====================================
Re-runs the label predictability probe and domain probe using FULL per-frame
sequences instead of pre-mean-pooled features.  This is the "correct" way to
query the model's representation because:

  * Each model was trained with [B, T, D] input.
  * Pre-pooling before _encode() passes a fake T=1 frame, which the projection
    layers see as a single timestep rather than a sequence → the output Z_c/Z_s
    may differ, especially for models with intra-sequence normalisation.

Protocol (same as eval_label_probe.py, different data pipeline):
  1.  Load raw [T, D] visual/audio sequences (no pre-pooling in the dataset).
  2.  Pass full sequence through model._encode(video [B,T,D], audio [B,T,D]).
  3.  Mean-pool the output Z_c [B,T,d_c] → [B,d_c] and Z_s → [B,d_s].
  4.  5-fold stratified CV logistic regression on AV1M val (in-domain).
  5.  Train-on-AV1M / eval-on-FAVC OOD transfer probe.
  6.  Domain probe (binary: AV1M=0 vs FAVC=1) on combined pool.

Output JSON: label_probe_seq.json  (output_dir/label_probe_seq.json)

Usage:
  python eval_label_probe_seq.py \\
      --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
      --config     configs/A6.yaml \\
      --av1m_root  ../data/avh_features/val \\
      --av1m_csv   csv_metadata/av1m/val_labels.csv \\
      --favc_root  ../data/favc_features \\
      --output_dir outputs_A6/results \\
      --device     cuda
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


# ── Category constants ────────────────────────────────────────────────────────
_FAVC_REAL_CATEGORY = "RealVideo-RealAudio"
_FAVC_FAKE_CATEGORIES = {"FakeVideo-RealAudio", "FakeVideo-FakeAudio",
                          "RealVideo-FakeAudio"}


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model_from_ckpt(ckpt_path: str):
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
    """Returns (video [T,1024], audio [T,1024], label) with full time axis."""
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
        return torch.tensor(video), torch.tensor(audio), label


def _seq_collate(batch):
    """batch_size=1: add batch dim without padding."""
    video, audio, label = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label])


# ── Item loaders ──────────────────────────────────────────────────────────────

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


def _load_favc_items(favc_root: str) -> list:
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
            continue
        for dirpath, _, filenames in os.walk(cat_path):
            for fname in sorted(filenames):
                if fname.endswith(".npz"):
                    items.append((os.path.join(dirpath, fname), label))
    return items


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_repr(model, dataloader: DataLoader, device: torch.device):
    """
    Pass each clip through model._encode() with its full [B,T,D] sequence.
    Mean-pool Z_c and Z_s over time.  Returns (zc [N,d_c], zs [N,d_s], labels).
    """
    model.eval()
    model.to(device)
    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="  Extracting", leave=False):
        video, audio, labels = batch
        video = video.to(device)   # [1, T, D]
        audio = audio.to(device)

        # Full sequence → _encode → mean-pool time axis
        causal_repr, spurious_repr, *_ = model._encode(video, audio)
        zc = causal_repr.mean(dim=1).cpu().numpy()    # [1, d_c]
        zs = spurious_repr.mean(dim=1).cpu().numpy()  # [1, d_s]
        all_zc.append(zc)
        all_zs.append(zs)
        all_labels.extend(labels.tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Probe helpers ──────────────────────────────────────────────────────────────

def probe_cv(features: np.ndarray, labels: np.ndarray,
             n_splits: int = 5, seed: int = 42):
    """5-fold stratified CV logistic regression. Returns (mean_auc, std_auc, folds)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr_idx, te_idx in skf.split(features, labels):
        scaler = StandardScaler().fit(features[tr_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(scaler.transform(features[tr_idx]), labels[tr_idx])
        proba = clf.predict_proba(scaler.transform(features[te_idx]))[:, 1]
        fold_aucs.append(roc_auc_score(labels[te_idx], proba))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), [float(a) for a in fold_aucs]


def probe_transfer(train_feat: np.ndarray, train_labels: np.ndarray,
                   test_feat: np.ndarray, test_labels: np.ndarray,
                   seed: int = 42) -> float:
    """Train on source, evaluate on target. Returns single AUC."""
    scaler = StandardScaler().fit(train_feat)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             random_state=seed)
    clf.fit(scaler.transform(train_feat), train_labels)
    proba = clf.predict_proba(scaler.transform(test_feat))[:, 1]
    return float(roc_auc_score(test_labels, proba))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sequence-based label + domain probe (A2/A5/A6).")
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--av1m_root",  required=True,
                        help="Root of AV1M val features (.npz)")
    parser.add_argument("--av1m_csv",   required=True,
                        help="AV1M val labels CSV")
    parser.add_argument("--favc_root",  required=True,
                        help="Root of FakeAVCeleb features")
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max_av1m",   type=int, default=None,
                        help="Subsample AV1M val to N clips (default: all)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.config) as f:
        config   = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    model, _ = _load_model_from_ckpt(args.ckpt_path)
    device   = torch.device(args.device)
    print(f"Model class : {model.__class__.__name__}")
    print(f"Device      : {device}")

    # ── Load items ─────────────────────────────────────────────────────────
    av1m_items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"\nAV1M val   : {len(av1m_items)} clips  "
          f"(real={sum(1 for _,l in av1m_items if l==0)}, "
          f"fake={sum(1 for _,l in av1m_items if l==1)})")
    if args.max_av1m and len(av1m_items) > args.max_av1m:
        av1m_items = rng.sample(av1m_items, args.max_av1m)
        print(f"  Subsampled to {len(av1m_items)}")

    favc_items = _load_favc_items(args.favc_root)
    print(f"FAVC test  : {len(favc_items)} clips  "
          f"(real={sum(1 for _,l in favc_items if l==0)}, "
          f"fake={sum(1 for _,l in favc_items if l==1)})")

    # ── DataLoaders ────────────────────────────────────────────────────────
    av1m_ds  = SequenceDataset(av1m_items,  apply_l2=apply_l2)
    favc_ds  = SequenceDataset(favc_items,  apply_l2=apply_l2)
    av1m_loader = DataLoader(av1m_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=_seq_collate)
    favc_loader = DataLoader(favc_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=_seq_collate)

    # ── Extract representations ─────────────────────────────────────────────
    print("\n[1/2] Extracting AV1M val representations...")
    av1m_zc, av1m_zs, av1m_lbl = extract_repr(model, av1m_loader, device)
    print(f"  Z_c shape : {av1m_zc.shape}  Z_s shape : {av1m_zs.shape}")

    print("\n[2/2] Extracting FAVC representations...")
    favc_zc, favc_zs, favc_lbl = extract_repr(model, favc_loader, device)
    print(f"  Z_c shape : {favc_zc.shape}  Z_s shape : {favc_zs.shape}")

    # ── In-domain label probe (AV1M val, 5-fold CV) ─────────────────────────
    print("\n── In-domain label probe (AV1M val, 5-fold CV) ──")
    zc_auc, zc_std, zc_folds = probe_cv(av1m_zc, av1m_lbl, args.n_folds, args.seed)
    zs_auc, zs_std, zs_folds = probe_cv(av1m_zs, av1m_lbl, args.n_folds, args.seed)
    print(f"  Z_c : {zc_auc:.4f} ± {zc_std:.4f}")
    print(f"  Z_s : {zs_auc:.4f} ± {zs_std:.4f}")

    # ── OOD transfer probe (train AV1M, eval FAVC) ──────────────────────────
    print("\n── OOD transfer probe (train AV1M → eval FAVC) ──")
    ood_zc = probe_transfer(av1m_zc, av1m_lbl, favc_zc, favc_lbl, args.seed)
    ood_zs = probe_transfer(av1m_zs, av1m_lbl, favc_zs, favc_lbl, args.seed)
    print(f"  Z_c OOD : {ood_zc:.4f}")
    print(f"  Z_s OOD : {ood_zs:.4f}")

    # ── Domain probe (AV1M=0 vs FAVC=1, 5-fold CV on pooled set) ───────────
    print("\n── Domain probe (AV1M=0 vs FAVC=1, 5-fold CV) ──")
    domain_zc = np.concatenate([av1m_zc, favc_zc], axis=0)
    domain_zs = np.concatenate([av1m_zs, favc_zs], axis=0)
    domain_lbl = np.array([0] * len(av1m_zc) + [1] * len(favc_zc), dtype=np.int32)

    dom_zc_auc, dom_zc_std, _ = probe_cv(domain_zc, domain_lbl, args.n_folds, args.seed)
    dom_zs_auc, dom_zs_std, _ = probe_cv(domain_zs, domain_lbl, args.n_folds, args.seed)
    print(f"  Z_c domain AUC : {dom_zc_auc:.4f} ± {dom_zc_std:.4f}")
    print(f"  Z_s domain AUC : {dom_zs_auc:.4f} ± {dom_zs_std:.4f}")

    # ── Collate results ────────────────────────────────────────────────────
    results = {
        "ckpt_path": args.ckpt_path,
        "note": "sequence-based: full [T,D] → _encode → mean-pool",
        "in_domain_label_probe": {
            "zc_auc": zc_auc, "zc_std": zc_std, "zc_folds": zc_folds,
            "zs_auc": zs_auc, "zs_std": zs_std, "zs_folds": zs_folds,
            "n_av1m": len(av1m_items),
        },
        "ood_transfer_probe": {
            "zc_auc": ood_zc,
            "zs_auc": ood_zs,
            "n_av1m_train": len(av1m_items),
            "n_favc_test":  len(favc_items),
        },
        "domain_probe": {
            "zc_auc": dom_zc_auc, "zc_std": dom_zc_std,
            "zs_auc": dom_zs_auc, "zs_std": dom_zs_std,
            "n_av1m":  len(av1m_items),
            "n_favc":  len(favc_items),
        },
    }

    print("\n── Summary ──")
    print(f"  In-domain Z_c label AUC : {zc_auc:.4f} ± {zc_std:.4f}")
    print(f"  In-domain Z_s label AUC : {zs_auc:.4f} ± {zs_std:.4f}")
    print(f"  OOD Z_c AUC             : {ood_zc:.4f}")
    print(f"  OOD Z_s AUC             : {ood_zs:.4f}")
    print(f"  Domain Z_c AUC          : {dom_zc_auc:.4f} ± {dom_zc_std:.4f}")
    print(f"  Domain Z_s AUC          : {dom_zs_auc:.4f} ± {dom_zs_std:.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "label_probe_seq.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out}")

    return results


if __name__ == "__main__":
    main()
