"""
Domain Predictability Evaluation
=================================
Experiment 1 (from supervisor): Measure how well each branch can predict
the SOURCE DOMAIN (AV1M=0 vs FakeAVCeleb=1) from its representations.

Expected result if disentanglement is working:
  spurious AUC_domain >> 0.5   (Z_s captures domain-specific artifacts)
  causal  AUC_domain ≈  0.5   (Z_c is domain-invariant)

Method:
  1. Load trained A1 (adversarial-only) checkpoint.
  2. Extract Z_c and Z_s for AV1M val samples (domain=0)
     and FakeAVCeleb val/test samples (domain=1).
     Each clip: mean-pool over time → fixed-size vector.
  3. Train a logistic regression linear probe (5-fold stratified CV)
     on top of Z_c and Z_s separately.
  4. Report mean AUC ± std for domain prediction.

Usage:
  python eval_domain_predictability.py \\
      --ckpt_path outputs_ablation_a1/ckpts/model-epoch=03.ckpt \\
      --a1_config  configs/ablation_a1.yaml \\
      --favc_root  ../data/favc_features \\
      --favc_csv   csv_metadata/favc \\
      --output_dir outputs_ablation_a1/results
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow importing sibling modules when run from avh_sup/
sys.path.insert(0, os.path.dirname(__file__))
from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_causal_ablation import AVH_Causal_Ablation


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_repr(model: AVH_Causal_Ablation, dataloader: DataLoader, device: torch.device):
    """
    Run the model encoder + projection heads and return mean-pooled
    causal and spurious representations for every clip.

    Returns
    -------
    zc : np.ndarray  [N, fused_dim]
    zs : np.ndarray  [N, fused_dim]
    """
    model.eval()
    model.to(device)

    all_zc, all_zs = [], []

    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        # Handle both 4-tuple (no domain) and 5-tuple (with domain) batches
        if len(batch) == 4:
            video_feats, audio_feats, _, _ = batch
        else:
            video_feats, audio_feats, _, _, _ = batch

        video_feats = video_feats.to(device)   # [B, T, feat_dim]
        audio_feats = audio_feats.to(device)   # [B, T, feat_dim]

        causal_repr, spurious_repr = model._encode(video_feats, audio_feats)

        # Mean-pool over the time dimension → [B, fused_dim]
        zc = causal_repr.mean(dim=1).cpu().numpy()
        zs = spurious_repr.mean(dim=1).cpu().numpy()

        all_zc.append(zc)
        all_zs.append(zs)

    return np.concatenate(all_zc, axis=0), np.concatenate(all_zs, axis=0)


# ── Linear probe (5-fold CV) ──────────────────────────────────────────────────

def linear_probe_auc(features: np.ndarray, labels: np.ndarray, n_splits: int = 5, seed: int = 42):
    """
    Fit a logistic regression linear probe with stratified k-fold CV.

    Returns
    -------
    mean_auc  : float
    std_auc   : float
    fold_aucs : list[float]
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx],   labels[test_idx]

        # Normalise features (fit scaler on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)

    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Domain predictability linear probe.")
    parser.add_argument("--ckpt_path",  required=True,
                        help="Path to trained A1 .ckpt file")
    parser.add_argument("--a1_config",  required=True,
                        help="Path to ablation_a1.yaml (for AV1M data paths)")
    parser.add_argument("--favc_root",  required=True,
                        help="Root directory of FakeAVCeleb NPZ features")
    parser.add_argument("--favc_csv",   default=None,
                        help="CSV metadata directory for FakeAVCeleb split (optional)")
    parser.add_argument("--favc_split", default="test",
                        choices=["train", "val", "test"],
                        help="Which FakeAVCeleb split to use (default: test)")
    parser.add_argument("--av1m_split", default="val",
                        choices=["train", "val"],
                        help="Which AV1M split to use (default: val)")
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save results JSON (optional)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load config & model ──────────────────────────────────────────────────
    with open(args.a1_config) as f:
        config = yaml.safe_load(f)

    print(f"Loading checkpoint: {args.ckpt_path}")
    model = AVH_Causal_Ablation.load_from_checkpoint(args.ckpt_path)
    device = torch.device(args.device)

    # ── Build datasets ───────────────────────────────────────────────────────
    print(f"Loading AV1M [{args.av1m_split}] …")
    av1m_ds = AV1M_trainval_dataset(config["data_info"], split=args.av1m_split)
    av1m_dl = DataLoader(av1m_ds, batch_size=1, shuffle=False, num_workers=2)

    favc_cfg = {
        "root_path": args.favc_root,
        "apply_l2": config["data_info"].get("apply_l2", True),
    }
    if args.favc_csv:
        favc_cfg["csv_root_path"] = args.favc_csv

    print(f"Loading FakeAVCeleb [{args.favc_split}] …")
    favc_ds = FakeAVCeleb_NPZ_Dataset(favc_cfg, split=args.favc_split if args.favc_csv else None)
    favc_dl = DataLoader(favc_ds, batch_size=1, shuffle=False, num_workers=2)

    # ── Extract representations ──────────────────────────────────────────────
    print("Extracting AV1M representations …")
    zc_av1m, zs_av1m = extract_repr(model, av1m_dl, device)

    print("Extracting FakeAVCeleb representations …")
    zc_favc, zs_favc = extract_repr(model, favc_dl, device)

    n_av1m = len(zc_av1m)
    n_favc = len(zc_favc)
    print(f"\nSample counts — AV1M: {n_av1m}  |  FakeAVCeleb: {n_favc}")

    # Domain labels: AV1M=0, FAVC=1
    domain_labels = np.array([0] * n_av1m + [1] * n_favc, dtype=np.int32)

    zc_all = np.concatenate([zc_av1m, zc_favc], axis=0)
    zs_all = np.concatenate([zs_av1m, zs_favc], axis=0)

    # ── Linear probe ────────────────────────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold linear probe for domain prediction …")

    print("  Probing Z_c (causal branch) …")
    zc_mean, zc_std, zc_folds = linear_probe_auc(zc_all, domain_labels, args.n_folds, args.seed)

    print("  Probing Z_s (spurious branch) …")
    zs_mean, zs_std, zs_folds = linear_probe_auc(zs_all, domain_labels, args.n_folds, args.seed)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DOMAIN PREDICTABILITY (linear probe, {}-fold CV)".format(args.n_folds))
    print("=" * 55)
    print(f"  Causal  branch  Z_c  AUC = {zc_mean:.4f} ± {zc_std:.4f}")
    print(f"  Spurious branch Z_s  AUC = {zs_mean:.4f} ± {zs_std:.4f}")
    print("=" * 55)
    print()
    print("  Interpretation:")
    print(f"    Z_c → {'domain-invariant ✓' if zc_mean < 0.60 else 'still domain-sensitive ✗'}")
    print(f"    Z_s → {'encodes domain info ✓' if zs_mean > 0.60 else 'domain info not captured ✗'}")
    print()

    results = {
        "ckpt_path": args.ckpt_path,
        "n_av1m": n_av1m,
        "n_favc": n_favc,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "causal_auc_mean": zc_mean,
        "causal_auc_std":  zc_std,
        "causal_auc_folds": zc_folds,
        "spurious_auc_mean": zs_mean,
        "spurious_auc_std":  zs_std,
        "spurious_auc_folds": zs_folds,
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
