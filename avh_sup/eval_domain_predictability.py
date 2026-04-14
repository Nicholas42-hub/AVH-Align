"""
Domain Predictability Probe
============================
Experiment 2: Measure how well each branch can predict dataset identity
(AV1M = 0  vs  FakeAVCeleb = 1).

This directly tests whether the domain adversarial objective (Component 3,
A3 only) is successfully suppressing domain information from Z_s.

Architecture recap:
  A1/A2 — no domain constraint on either branch.
  A3     — domain_head(GRL_domain(Z_s)): GRL reverses domain gradient into
           Z_s only, pushing Z_s to be domain-INVARIANT.

Expected probe outcomes:
  A1/A2 (no domain adversarial):
    Z_c AUC >> 0.5   — causal branch freely encodes domain artefacts
    Z_s AUC >> 0.5   — spurious branch also freely encodes domain artefacts
    (both should be similarly high; no domain constraint on either)

  A3 (domain adversarial on Z_s):
    Z_s AUC ≈ 0.5    — GRL successfully suppressed domain info from Z_s
                        (the target behavior: Z_s low domain AUC)
    Z_c AUC >> 0.5   — causal branch is unconstrained; can still encode domain

For fairness the probe uses a balanced subsample: equal numbers of clips
from each dataset (up to --max_per_dataset, default 2000).

Method:
  1. Load trained checkpoint.
  2. Sample up to N clips from AV1M val split (label 0) and N clips from
     FakeAVCeleb (label 1, all categories merged).
     Each clip: mean-pool over time → fixed-size vector.
  3. Train a logistic regression binary probe (5-fold stratified CV).
  4. Report standard binary AUC mean ± std.

Usage:
  python eval_domain_predictability.py \\
      --ckpt_path     outputs_A1/ckpts/model-epoch=27.ckpt \\
      --config        configs/A1.yaml \\
      --av1m_root     ../data/avh_features/val \\
      --favc_root     ../data/favc_features \\
      --output_dir    outputs_A1/results
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
from mlp_fcd import AVH_FCD
from mlp_fcd_a6 import AVH_FCD_A6
from mlp_fcd_a7 import AVH_FCD_A7
from mlp_sad_a8 import AVH_SAD_A8
from mlp_sad_a9 import AVH_SAD_A9

RANDOM_CHANCE = 0.5  # binary classification baseline


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _collect_npz(root: str) -> list:
    """Recursively collect all .npz paths under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith(".npz"):
                paths.append(os.path.join(dirpath, fname))
    return paths


class DomainDataset(Dataset):
    """
    Pairs (visual_feat, audio_feat, domain_label) for domain probe.
      domain_label = 0  →  AV1M
      domain_label = 1  →  FakeAVCeleb

    Clips are mean-pooled over time before being returned, which avoids
    the need for sequence-length collation.
    """
    def __init__(self, items: list, apply_l2: bool = True):
        """
        items: list of (abs_npz_path, domain_label)
        """
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
        video = video.mean(axis=0)   # [feat_dim]
        audio = audio.mean(axis=0)   # [feat_dim]
        return torch.tensor(video), torch.tensor(audio), label


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_raw(dataloader: DataLoader):
    """
    Baseline: extract raw AV-HuBERT features WITHOUT passing through the model.
    Concatenates mean-pooled video + audio → [B, 2*feat_dim].

    If this probe AUC is already ~99%, the domain signal lives in the upstream
    AV-HuBERT features themselves, not in anything the model learned.
    Compare model Z_c/Z_s AUC against this baseline to measure the model's
    actual contribution to domain (de)mixing.
    """
    all_raw, all_labels = [], []
    for batch in tqdm(dataloader, desc="Raw features", leave=False):
        video_feats, audio_feats, labels = batch   # [B, feat_dim] each
        raw = torch.cat([video_feats, audio_feats], dim=-1)  # [B, 2*feat_dim]
        all_raw.append(raw.numpy())
        all_labels.extend(labels.numpy().tolist())
    return (np.concatenate(all_raw, axis=0),
            np.array(all_labels, dtype=np.int32))


@torch.no_grad()
def extract_repr(model: AVH_Causal_Ablation, dataloader: DataLoader, device: torch.device):
    """Extract mean-pooled Z_c and Z_s for every clip."""
    model.eval()
    model.to(device)
    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        video_feats, audio_feats, labels = batch
        # video_feats: [B, feat_dim] — add dummy time dim for _encode
        video_feats = video_feats.unsqueeze(1).to(device)   # [B, 1, feat_dim]
        audio_feats = audio_feats.unsqueeze(1).to(device)   # [B, 1, feat_dim]

        causal_repr, spurious_repr, *_ = model._encode(video_feats, audio_feats)
        all_zc.append(causal_repr.squeeze(1).cpu().numpy())   # [B, fused_dim]
        all_zs.append(spurious_repr.squeeze(1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Linear probe (5-fold CV, binary AUC) ────────────────────────────────────

def linear_probe_binary_auc(features: np.ndarray, labels: np.ndarray,
                             n_splits: int = 5, seed: int = 42):
    """
    Binary logistic regression probe with stratified k-fold CV.
    Reports standard binary AUC (random baseline = 0.5).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx],   labels[test_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(X_train, y_train)

        # Binary: use probability of the positive class (label=1)
        proba = clf.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)

    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Domain predictability probe: AV1M vs FakeAVCeleb.")
    parser.add_argument("--ckpt_path",         required=True,
                        help="Path to trained .ckpt file")
    parser.add_argument("--config",            required=True,
                        help="Path to model config yaml (e.g. configs/A1.yaml)")
    parser.add_argument("--av1m_root",         required=True,
                        help="Root of AV1M val features (flat id/clip.npz layout)")
    parser.add_argument("--favc_root",         required=True,
                        help="Root of FakeAVCeleb features (category/id/clip.npz)")
    parser.add_argument("--max_per_dataset",   type=int, default=2000,
                        help="Max clips per domain for balanced probe (default 2000)")
    parser.add_argument("--n_folds",           type=int, default=5)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--output_dir",        default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load config & model ──────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]
    # Auto-detect model class:
    #   A9: sync_dim + lambda_mmd  (SAD + MMD)
    #   A8: sync_dim (SAD + GRL)
    #   A7: syn_dim + lambda_ladv  (FCD + label adversarial)
    #   A6: syn_dim + lambda_dadv  (FCD + domain adversarial)
    #   A5/earlier: syn_dim or no distinguishing key
    #   A44: proj_dim + sda_dim    (binary + Sinkhorn + domain)
    #   A42/A43: proj_dim + lambda_dadv (binary + domain, no syn_dim)
    _hp = ckpt_cfg.get("model_hparams", {})
    _ablation_id = ckpt_cfg.get("ablation_id", "")
    if "sync_dim" in _hp and "lambda_mmd" in _hp:
        model = AVH_SAD_A9(config=ckpt_cfg)
    elif "sync_dim" in _hp:
        model = AVH_SAD_A8(config=ckpt_cfg)
    elif "lambda_ladv" in _hp and "syn_dim" in _hp:
        model = AVH_FCD_A7(config=ckpt_cfg)
    elif "lambda_dadv" in _hp and "syn_dim" in _hp:
        model = AVH_FCD_A6(config=ckpt_cfg)
    elif "syn_dim" in _hp:
        model = AVH_FCD(config=ckpt_cfg)
    elif _ablation_id == "A44" or ("sda_dim" in _hp and "proj_dim" in _hp):
        from mlp_causal_a44 import AVH_Causal_A44
        model = AVH_Causal_A44(config=ckpt_cfg)
    elif _ablation_id == "A43" or (_ablation_id == "" and "proj_dim" in _hp and "lambda_dadv" in _hp):
        from mlp_causal_a43 import AVH_Causal_A43
        model = AVH_Causal_A43(config=ckpt_cfg)
    elif _ablation_id == "A42":
        from mlp_causal_a42 import AVH_Causal_A42
        model = AVH_Causal_A42(config=ckpt_cfg)
    else:
        model = AVH_Causal_Ablation(config=ckpt_cfg)
    print(f"  Model class: {model.__class__.__name__}")
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    device = torch.device(args.device)

    # ── Collect clip paths ───────────────────────────────────────────────────
    print(f"\nCollecting AV1M clips from: {args.av1m_root}")
    av1m_all = _collect_npz(args.av1m_root)
    print(f"  Found {len(av1m_all)} clips total")

    print(f"\nCollecting FakeAVCeleb clips from: {args.favc_root}")
    favc_all = _collect_npz(args.favc_root)
    print(f"  Found {len(favc_all)} clips total")

    # ── Balance: subsample to min(max_per_dataset, available) ────────────────
    n = min(args.max_per_dataset, len(av1m_all), len(favc_all))
    av1m_sample = rng.sample(av1m_all, n)
    favc_sample = rng.sample(favc_all, n)
    print(f"\nUsing {n} clips per domain ({2*n} total) for balanced probe.")

    items = [(p, 0) for p in av1m_sample] + [(p, 1) for p in favc_sample]
    rng.shuffle(items)

    ds = DomainDataset(items, apply_l2=apply_l2)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    # ── Extract raw baseline features (no model) ────────────────────────────
    print("\nExtracting raw AV-HuBERT features (upstream baseline, no model) …")
    raw_feats, domain_labels_raw = extract_raw(dl)

    # ── Extract model representations ────────────────────────────────────────
    print("Extracting model representations (Z_c, Z_s) …")
    zc, zs, domain_labels = extract_repr(model, dl, device)
    assert (domain_labels == domain_labels_raw).all(), \
        "Label mismatch between raw and model extraction"
    print(f"Extracted {len(zc)} clips  (AV1M={int((domain_labels==0).sum())}, "
          f"FAVC={int((domain_labels==1).sum())})")

    # ── Linear probe ────────────────────────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold linear probe (binary AUC) …")
    print(f"  Task: AV1M (0) vs FakeAVCeleb (1)")
    print(f"  Random chance = {RANDOM_CHANCE:.3f}\n")

    print("  Probing raw AV-HuBERT features (upstream baseline) …")
    raw_mean, raw_std, raw_folds = linear_probe_binary_auc(
        raw_feats, domain_labels, args.n_folds, args.seed)

    print("  Probing Z_c (causal branch) …")
    zc_mean, zc_std, zc_folds = linear_probe_binary_auc(
        zc, domain_labels, args.n_folds, args.seed)

    print("  Probing Z_s (spurious branch) …")
    zs_mean, zs_std, zs_folds = linear_probe_binary_auc(
        zs, domain_labels, args.n_folds, args.seed)

    # ── Print results ────────────────────────────────────────────────────────
    # The raw AUC is the ceiling set by the upstream AV-HuBERT encoder.
    # Only deviations from this baseline (Δraw) are attributable to the model.
    def _delta(auc):
        d = auc - raw_mean
        return f"(Δraw {d:+.4f})"

    print("\n" + "=" * 72)
    print(f"  DOMAIN PREDICTABILITY  ({args.n_folds}-fold CV, binary AUC)")
    print(f"  Probe: AV1M (0) vs FakeAVCeleb (1)   random chance = {RANDOM_CHANCE:.3f}")
    print("=" * 72)
    print(f"  Raw AV-HuBERT (upstream baseline)  AUC = {raw_mean:.4f} ± {raw_std:.4f}")
    print(f"  Causal   branch  Z_c               AUC = {zc_mean:.4f} ± {zc_std:.4f}  {_delta(zc_mean)}")
    print(f"  Spurious branch  Z_s               AUC = {zs_mean:.4f} ± {zs_std:.4f}  {_delta(zs_mean)}")
    print("=" * 72)
    print()
    print("  Interpretation:")
    print(f"    Δraw(Z_c) = {zc_mean - raw_mean:+.4f}  "
          "(negative = model suppressed domain from Z_c)")
    print(f"    Δraw(Z_s) = {zs_mean - raw_mean:+.4f}  "
          "(positive = model concentrated domain into Z_s)")
    print()

    results = {
        "probe_design":         "binary: AV1M (0) vs FakeAVCeleb (1)",
        "ckpt_path":            args.ckpt_path,
        "n_clips_per_domain":   n,
        "n_clips_total":        int(len(zc)),
        "n_folds":              args.n_folds,
        "seed":                 args.seed,
        "random_chance":        RANDOM_CHANCE,
        "raw_auc_mean":         raw_mean,
        "raw_auc_std":          raw_std,
        "raw_auc_folds":        raw_folds,
        "causal_auc_mean":      zc_mean,
        "causal_auc_std":       zc_std,
        "causal_auc_folds":     zc_folds,
        "causal_delta_raw":     round(zc_mean - raw_mean, 6),
        "spurious_auc_mean":    zs_mean,
        "spurious_auc_std":     zs_std,
        "spurious_auc_folds":   zs_folds,
        "spurious_delta_raw":   round(zs_mean - raw_mean, 6),
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "domain_predictability.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
