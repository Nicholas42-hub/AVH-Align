"""
FAVC Per-Category Branch Probe Under Perturbation
==================================================
The AV1M branch probe (eval_branch_probe_perturb.py) is limited because AV1M
is an audio-only-fake dataset — visual masking never hurts any model.

This script uses FakeAVCeleb, split by manipulation category, which provides
four complementary test conditions:

  FV-RA  (FakeVideo-RealAudio):  only video is synthesised → detectable via
           visual channel. Under audio masking, Z_c^FCD should RETAIN signal
           (visual-unique u_v channel), while Z_c^binary COLLAPSES (audio-led).

  RV-FA  (RealVideo-FakeAudio):  only audio is synthesised → detectable via
           audio channel. Under visual masking, signal should survive.
           Under audio masking, all models collapse.

  FV-FA  (FakeVideo-FakeAudio):  both synthesised → both channels carry signal.

  ALL    (all three fake categories vs RV-RA real):  overall performance.

Real reference: RealVideo-RealAudio (RV-RA) — 100 clips.

Protocol (per fake category, per perturbation):
  1.  Load full [T,D] sequences for the category's fake clips + all RV-RA real.
  2.  Apply perturbation to raw features.
  3.  model._encode() → mean-pool Z_c [T,d_c] → [d_c] and Z_s [T,d_s] → [d_s].
  4.  5-fold stratified CV logistic regression probe → label AUC.

Perturbation types:
  audio_masking   — frame drop rate ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}
  visual_masking  — frame drop rate ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}
  audio_noise     — Gaussian σ     ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}
  visual_noise    — Gaussian σ     ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}

Output JSON: favc_branch_probe_perturb.json
  {
    "FakeVideo-RealAudio": {
      "audio_masking": {
        "0.00": {"zc_auc": ..., "zc_std": ..., "zs_auc": ..., "zs_std": ...,
                 "n_fake": ..., "n_real": ...},
        "0.10": {...}, ...
      },
      "visual_masking": {...},
      "audio_noise":    {...},
      "visual_noise":   {...},
    },
    "RealVideo-FakeAudio": { ... },
    "FakeVideo-FakeAudio": { ... },
    "overall": { ... },
  }

Usage:
  python eval_favc_branch_probe_perturb.py \\
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


# ── Perturbation schedule ──────────────────────────────────────────────────────
PERTURB_CONFIG = {
    "audio_masking":  [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "visual_masking": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "audio_noise":    [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "visual_noise":   [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
}

_REAL_CATEGORY  = "RealVideo-RealAudio"
_FAKE_CATEGORIES = ["FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]


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


# ── Sequence dataset ───────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
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
    video, audio, label = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label])


# ── FAVC item loader ───────────────────────────────────────────────────────────

def _load_favc_by_category(favc_root: str) -> dict:
    """
    Returns dict: category_name → list of (abs_npz_path, int_label).
    Real clips (RV-RA) have label=0; fake clips have label=1.
    """
    cat_items = {}
    for category in os.listdir(favc_root):
        cat_path = os.path.join(favc_root, category)
        if not os.path.isdir(cat_path):
            continue
        label = 0 if category == _REAL_CATEGORY else 1
        items = []
        for dirpath, _, filenames in os.walk(cat_path):
            for fname in sorted(filenames):
                if fname.endswith(".npz"):
                    items.append((os.path.join(dirpath, fname), label))
        if items:
            cat_items[category] = items
    return cat_items


# ── Perturbation helpers ───────────────────────────────────────────────────────

def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def apply_perturbation(video: torch.Tensor, audio: torch.Tensor,
                       perturb_type: str, level: float, seed_offset: int = 0):
    """Apply perturbation and return L2-normed (video, audio) for _encode."""
    if level == 0.0:
        return _l2_norm(video), _l2_norm(audio)

    if perturb_type == "audio_masking":
        B, T, D = audio.shape
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=audio.device))
        audio = audio * mask
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "visual_masking":
        B, T, D = video.shape
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=video.device))
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


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_perturbed(model, dataloader: DataLoader,
                      perturb_type: str, level: float,
                      device: torch.device, seed: int = 42):
    """Extract mean-pooled Z_c and Z_s after applying perturbation."""
    model.eval()
    model.to(device)
    torch.manual_seed(seed)

    all_zc, all_zs, all_labels = [], [], []
    for batch in dataloader:
        video_raw, audio_raw, labels = batch
        video_raw = video_raw.to(device)
        audio_raw = audio_raw.to(device)

        video_p, audio_p = apply_perturbation(video_raw, audio_raw,
                                              perturb_type, level)
        causal_repr, spurious_repr, *_ = model._encode(video_p, audio_p)
        all_zc.append(causal_repr.mean(dim=1).cpu().numpy())
        all_zs.append(spurious_repr.mean(dim=1).cpu().numpy())
        all_labels.extend(labels.tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Logistic probe (5-fold CV) ─────────────────────────────────────────────────

def probe_cv(features: np.ndarray, labels: np.ndarray,
             n_splits: int = 5, seed: int = 42):
    """Returns (mean_auc, std_auc). Falls back gracefully for tiny sets."""
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0
    # Cap n_splits to smallest class count
    n_splits = min(n_splits, np.bincount(labels).min())
    if n_splits < 2:
        # Too few samples per class for CV — single train/test split
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            features, labels, test_size=0.3, random_state=seed, stratify=labels)
        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(scaler.transform(X_tr), y_tr)
        proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
        return float(roc_auc_score(y_te, proba)), 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for tr_idx, te_idx in skf.split(features, labels):
        scaler = StandardScaler().fit(features[tr_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(scaler.transform(features[tr_idx]), labels[tr_idx])
        proba = clf.predict_proba(scaler.transform(features[te_idx]))[:, 1]
        fold_aucs.append(roc_auc_score(labels[te_idx], proba))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FAVC per-category branch probe under perturbation.")
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--favc_root",  required=True,
                        help="Root of FakeAVCeleb features (category/id/clip.npz)")
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
    print(f"FAVC root   : {args.favc_root}")

    # ── Load all FAVC clips by category ────────────────────────────────────
    cat_items = _load_favc_by_category(args.favc_root)
    real_items = cat_items.get(_REAL_CATEGORY, [])
    print(f"\nFAVC clips by category:")
    for cat, items in cat_items.items():
        print(f"  {cat}: {len(items)}")

    # ── Build per-category probe datasets ──────────────────────────────────
    # For each fake category, pool it with the real clips
    probe_configs = {}
    for fake_cat in _FAKE_CATEGORIES:
        if fake_cat not in cat_items:
            print(f"  WARNING: {fake_cat} not found, skipping")
            continue
        combined = cat_items[fake_cat] + real_items  # fake(1) + real(0)
        probe_configs[fake_cat] = combined

    # "overall": all fake categories combined vs real
    all_fake = []
    for fake_cat in _FAKE_CATEGORIES:
        all_fake.extend(cat_items.get(fake_cat, []))
    probe_configs["overall"] = all_fake + real_items

    # ── Run probes ─────────────────────────────────────────────────────────
    results = {}

    for cat_name, items in probe_configs.items():
        n_fake = sum(1 for _, l in items if l == 1)
        n_real = sum(1 for _, l in items if l == 0)
        print(f"\n{'='*60}")
        print(f"  Category: {cat_name}  (fake={n_fake}, real={n_real})")
        print(f"{'='*60}")

        ds     = SequenceDataset(items, apply_l2=apply_l2)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=_seq_collate)

        results[cat_name] = {}

        for perturb_type, levels in PERTURB_CONFIG.items():
            print(f"\n  ── {perturb_type} ──")
            results[cat_name][perturb_type] = {}

            for level in levels:
                zc, zs, labels = extract_perturbed(
                    model, loader, perturb_type, level, device, seed=args.seed)
                zc_auc, zc_std = probe_cv(zc, labels, args.n_folds, args.seed)
                zs_auc, zs_std = probe_cv(zs, labels, args.n_folds, args.seed)

                key = f"{level:.2f}"
                results[cat_name][perturb_type][key] = {
                    "zc_auc": zc_auc, "zc_std": zc_std,
                    "zs_auc": zs_auc, "zs_std": zs_std,
                    "n_fake": int(n_fake), "n_real": int(n_real),
                }
                print(f"    level={level:.2f}  "
                      f"Z_c={zc_auc:.4f}±{zc_std:.4f}  "
                      f"Z_s={zs_auc:.4f}±{zs_std:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "favc_branch_probe_perturb.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out}")

    # ── Print summary: key cells ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY — Z_c AUC at 100% masking / noise=1.0")
    print("="*60)
    for cat_name in ["FakeVideo-RealAudio", "RealVideo-FakeAudio",
                     "FakeVideo-FakeAudio", "overall"]:
        if cat_name not in results:
            continue
        r = results[cat_name]
        am = r.get("audio_masking", {}).get("1.00", {})
        vm = r.get("visual_masking", {}).get("1.00", {})
        an = r.get("audio_noise",    {}).get("1.00", {})
        vn = r.get("visual_noise",   {}).get("1.00", {})
        print(f"  {cat_name[:25]:25s}: "
              f"amask={am.get('zc_auc', 0):.4f}  "
              f"vmask={vm.get('zc_auc', 0):.4f}  "
              f"anoise={an.get('zc_auc', 0):.4f}  "
              f"vnoise={vn.get('zc_auc', 0):.4f}")

    return results


if __name__ == "__main__":
    main()
