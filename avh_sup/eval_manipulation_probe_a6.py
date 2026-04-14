"""
Manipulation-Type Probe — FCD Family (A5, A6, A7)
===================================================
Task: distinguish FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)
      within FakeAVCeleb, using Z_c and Z_s representations.

Uses the FCD model auto-detection logic:
  A7: syn_dim + lambda_ladv
  A6: syn_dim + lambda_dadv
  A5: syn_dim (no domain heads)

Usage:
    python eval_manipulation_probe_a6.py \\
        --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
        --config     configs/A6.yaml \\
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

FAKE_VIDEO_REAL_AUDIO = "FakeVideo-RealAudio"   # label 0
FAKE_VIDEO_FAKE_AUDIO = "FakeVideo-FakeAudio"   # label 1
RANDOM_CHANCE = 0.5


# ── Dataset ───────────────────────────────────────────────────────────────────

def _collect_npz(root: str) -> list:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.endswith(".npz"):
                paths.append(os.path.join(dirpath, fname))
    return paths


class ManipDataset(Dataset):
    """Mean-pool clips from two FakeVideo categories."""
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


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_raw(dataloader: DataLoader):
    all_raw, all_labels = [], []
    for batch in tqdm(dataloader, desc="Raw features", leave=False):
        video_feats, audio_feats, labels = batch
        raw = torch.cat([video_feats, audio_feats], dim=-1)
        all_raw.append(raw.numpy())
        all_labels.extend(labels.numpy().tolist())
    return (np.concatenate(all_raw, axis=0),
            np.array(all_labels, dtype=np.int32))


@torch.no_grad()
def extract_repr(model, dataloader: DataLoader, device: torch.device):
    """Extract Z_c and Z_s (first two return values of _encode)."""
    model.eval()
    model.to(device)
    all_zc, all_zs, all_labels = [], [], []

    for batch in tqdm(dataloader, desc="Model repr", leave=False):
        video_feats, audio_feats, labels = batch
        video_feats = video_feats.unsqueeze(1).to(device)
        audio_feats = audio_feats.unsqueeze(1).to(device)
        result = model._encode(video_feats, audio_feats)
        causal_repr   = result[0]
        spurious_repr = result[1]
        all_zc.append(causal_repr.squeeze(1).cpu().numpy())
        all_zs.append(spurious_repr.squeeze(1).cpu().numpy())
        all_labels.extend(labels.numpy().tolist())

    return (np.concatenate(all_zc, axis=0),
            np.concatenate(all_zs, axis=0),
            np.array(all_labels, dtype=np.int32))


# ── Linear probe ─────────────────────────────────────────────────────────────

def linear_probe_auc(features, labels, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, test_idx in skf.split(features, labels):
        scaler  = StandardScaler().fit(features[train_idx])
        X_train = scaler.transform(features[train_idx])
        X_test  = scaler.transform(features[test_idx])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                 random_state=seed)
        clf.fit(X_train, labels[train_idx])
        proba = clf.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(labels[test_idx], proba))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manipulation-type probe — FCD models (A5/A6/A7)")
    parser.add_argument("--ckpt_path",       required=True)
    parser.add_argument("--config",          required=True)
    parser.add_argument("--favc_root",       required=True,
                        help="Root of favc_features/ (contains FakeVideo-* subdirs)")
    parser.add_argument("--max_per_class",   type=int, default=1000,
                        help="Max clips per class for balanced probe (default 1000)")
    parser.add_argument("--n_folds",         type=int, default=5)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output_dir",      default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    # ── Load model (auto-detect FCD variant) ──────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt     = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]
    _hp      = ckpt_cfg.get("model_hparams", {})

    if "lambda_ladv" in _hp and "syn_dim" in _hp:
        from mlp_fcd_a7 import AVH_FCD_A7
        model = AVH_FCD_A7(config=ckpt_cfg)
        model_name = "AVH_FCD_A7"
    elif "lambda_dadv" in _hp and "syn_dim" in _hp:
        from mlp_fcd_a6 import AVH_FCD_A6
        model = AVH_FCD_A6(config=ckpt_cfg)
        model_name = "AVH_FCD_A6"
    elif "syn_dim" in _hp:
        from mlp_fcd import AVH_FCD
        model = AVH_FCD(config=ckpt_cfg)
        model_name = "AVH_FCD"
    else:
        raise ValueError(
            "Checkpoint does not appear to be an FCD model (no syn_dim in config). "
            "Use eval_manipulation_probe.py for binary models."
        )

    model_state    = model.state_dict()
    filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    print(f"  Model class : {model_name}")
    device = torch.device(args.device)

    # ── Collect clips ─────────────────────────────────────────────────────────
    fvra_root = os.path.join(args.favc_root, FAKE_VIDEO_REAL_AUDIO)
    fvfa_root = os.path.join(args.favc_root, FAKE_VIDEO_FAKE_AUDIO)

    fvra_all = _collect_npz(fvra_root)
    fvfa_all = _collect_npz(fvfa_root)
    print(f"FakeVideo-RealAudio clips : {len(fvra_all)}")
    print(f"FakeVideo-FakeAudio clips : {len(fvfa_all)}")

    n = min(args.max_per_class, len(fvra_all), len(fvfa_all))
    items = ([(p, 0) for p in rng.sample(fvra_all, n)] +
             [(p, 1) for p in rng.sample(fvfa_all, n)])
    rng.shuffle(items)
    print(f"Using {n} clips per class ({2*n} total, balanced).\n")

    ds = ManipDataset(items, apply_l2=apply_l2)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    # ── Extract ───────────────────────────────────────────────────────────────
    print("Extracting raw AV-HuBERT features (upstream baseline) …")
    raw_feats, labels_raw = extract_raw(dl)

    print("Extracting model representations (Z_c, Z_s) …")
    zc, zs, labels = extract_repr(model, dl, device)
    assert (labels == labels_raw).all()
    print(f"Extracted {len(zc)} clips  "
          f"(FakeVideo-RealAudio={int((labels==0).sum())}, "
          f"FakeVideo-FakeAudio={int((labels==1).sum())})\n")

    # ── Probe ─────────────────────────────────────────────────────────────────
    print(f"Running {args.n_folds}-fold linear probe …\n")

    raw_mean, raw_std, raw_folds = linear_probe_auc(raw_feats, labels, args.n_folds, args.seed)
    zc_mean,  zc_std,  zc_folds  = linear_probe_auc(zc,        labels, args.n_folds, args.seed)
    zs_mean,  zs_std,  zs_folds  = linear_probe_auc(zs,        labels, args.n_folds, args.seed)

    def _delta(auc):
        return f"(Δraw {auc - raw_mean:+.4f})"

    print("=" * 72)
    print(f"  MANIPULATION-TYPE PROBE  ({args.n_folds}-fold CV, binary AUC)")
    print(f"  Task: FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)")
    print(f"  Model: {model_name}")
    print(f"  Random chance = {RANDOM_CHANCE:.3f}")
    print("=" * 72)
    print(f"  Raw AV-HuBERT (upstream baseline)  AUC = {raw_mean:.4f} ± {raw_std:.4f}")
    print(f"  Causal   branch  Z_c               AUC = {zc_mean:.4f} ± {zc_std:.4f}  {_delta(zc_mean)}")
    print(f"  Spurious branch  Z_s               AUC = {zs_mean:.4f} ± {zs_std:.4f}  {_delta(zs_mean)}")
    print("=" * 72)
    print()
    print("  Expected if disentanglement works:")
    print("    Δraw(Z_s) >> 0  (Z_s sensitive to audio manipulation style)")
    print("    Δraw(Z_c) ≈  0  (Z_c blind to audio swap type)")
    print()

    results = {
        "probe_design":       "FakeVideo-RealAudio (0) vs FakeVideo-FakeAudio (1)",
        "ckpt_path":          args.ckpt_path,
        "model_class":        model_name,
        "n_clips_per_class":  n,
        "n_clips_total":      int(len(zc)),
        "n_folds":            args.n_folds,
        "seed":               args.seed,
        "random_chance":      RANDOM_CHANCE,
        "raw_auc_mean":       raw_mean,
        "raw_auc_std":        raw_std,
        "raw_auc_folds":      raw_folds,
        "causal_auc_mean":    zc_mean,
        "causal_auc_std":     zc_std,
        "causal_auc_folds":   zc_folds,
        "causal_delta_raw":   round(zc_mean - raw_mean, 6),
        "spurious_auc_mean":  zs_mean,
        "spurious_auc_std":   zs_std,
        "spurious_auc_folds": zs_folds,
        "spurious_delta_raw": round(zs_mean - raw_mean, 6),
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "manipulation_probe.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
