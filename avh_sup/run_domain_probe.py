"""
Domain probe for frozen-encoder causal models (A42, A43, ...).

For a given checkpoint, loads:
  - 200 AV1M val clips (domain=0)
  - FAVC val-real clips (domain=1, up to 200)

Extracts Z_c and Z_s (mean-pooled over time), then runs:
  - zc_raw:       5-fold logistic probe on Z_c (want < 0.65)
  - zc_centered:  same after per-domain mean subtraction (covariance residual)
  - zs_raw:       5-fold logistic probe on Z_s (want > 0.80)

Usage:
  python run_domain_probe.py --ckpt outputs_A42/ckpts/model-epoch=05.ckpt --model causal_a42
  python run_domain_probe.py --ckpt outputs_A43/ckpts/model-epoch=05.ckpt --model causal_a43
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from datasets import AV1M_trainval_dataset


def _probe(X, y, n_folds=5):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    try:
        return roc_auc_score(y, probs)
    except Exception:
        return 0.5


def load_model(ckpt_path, model_type):
    if model_type == "causal_a42":
        from mlp_causal_a42 import AVH_Causal_A42
        model = AVH_Causal_A42.load_from_checkpoint(ckpt_path)
    elif model_type == "causal_a43":
        from mlp_causal_a43 import AVH_Causal_A43
        model = AVH_Causal_A43.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()
    return model


@torch.no_grad()
def extract_representations(model, loader, device):
    zc_list, zs_list, dom_list = [], [], []
    for batch in loader:
        video, audio, _, domain, _ = batch
        video = video.to(device)
        audio = audio.to(device)
        zc, zs = model._encode(video, audio)
        zc_list.append(zc.mean(1).cpu().numpy())   # mean-pool over time → [B, D]
        zs_list.append(zs.mean(1).cpu().numpy())
        dom_list.extend(domain.tolist())
    return (
        np.concatenate(zc_list),
        np.concatenate(zs_list),
        np.array(dom_list),
    )


def collate_fn(batch):
    import torch.nn.functional as F
    videos, audios, cls_labels, domain_labels, paths = zip(*batch)
    max_T = max(v.shape[0] for v in videos)
    videos_padded = torch.stack([
        F.pad(v, (0, 0, 0, max_T - v.shape[0])) for v in videos
    ])
    audios_padded = torch.stack([
        F.pad(a, (0, 0, 0, max_T - a.shape[0])) for a in audios
    ])
    return (
        videos_padded,
        audios_padded,
        torch.tensor([int(c) if not isinstance(c, torch.Tensor) else c.item() for c in cls_labels]),
        torch.stack([torch.tensor(d) if not isinstance(d, torch.Tensor) else d for d in domain_labels]),
        list(paths),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True)
    parser.add_argument("--model", required=True, choices=["causal_a42", "causal_a43"])
    parser.add_argument("--av1m_root",  default="/data/projects/punim2637/nnliang/AVH-Align/data/avh_features")
    parser.add_argument("--av1m_csv",   default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m")
    parser.add_argument("--favc_root",  default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    parser.add_argument("--favc_csv",   default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc")
    parser.add_argument("--n_probe",    type=int, default=200)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"\nLoading {args.model} from: {args.ckpt}", flush=True)
    model = load_model(args.ckpt, args.model)
    model.to(device)

    # ── AV1M val clips (domain=0) ─────────────────────────────────────────────
    from datasets import AV1M_trainval_dataset
    import pandas as pd

    av1m_config = {
        "root_path":     args.av1m_root,
        "csv_root_path": args.av1m_csv,
        "apply_l2":      True,
    }
    av1m_ds = AV1M_trainval_dataset(config=av1m_config, split="val")
    n_av1m = min(args.n_probe, len(av1m_ds))
    av1m_indices = list(range(n_av1m))

    class AV1M_DomainWrapper(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            video, audio, cls_label, path = self.base[self.indices[i]]
            return video, audio, cls_label, torch.tensor(0.0), path

    av1m_probe = AV1M_DomainWrapper(av1m_ds, av1m_indices)
    print(f"AV1M probe clips: {len(av1m_probe)}", flush=True)

    # ── FAVC val-real clips (domain=1) ────────────────────────────────────────
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from train_test_causal_a42 import FAVC_RealDomainDataset

    favc_config = {
        "root_path":     args.favc_root,
        "csv_root_path": args.favc_csv,
        "split":         "val",
        "real_only":     True,
        "apply_l2":      True,
    }
    favc_ds = FAVC_RealDomainDataset(favc_config)
    n_favc = min(args.n_probe, len(favc_ds))
    favc_indices = list(range(n_favc))

    class FAVC_Subset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    favc_probe = FAVC_Subset(favc_ds, favc_indices)
    print(f"FAVC probe clips: {len(favc_probe)}", flush=True)

    from torch.utils.data import ConcatDataset
    probe_ds = ConcatDataset([av1m_probe, favc_probe])
    probe_loader = DataLoader(
        probe_ds, batch_size=32, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    # ── Extract ───────────────────────────────────────────────────────────────
    print("\nExtracting Z_c / Z_s ...", flush=True)
    zc_all, zs_all, dom_all = extract_representations(model, probe_loader, device)
    print(f"  Z_c shape: {zc_all.shape},  domains: {np.bincount(dom_all.astype(int))}", flush=True)

    # ── Probe ─────────────────────────────────────────────────────────────────
    auc_zc_raw = _probe(zc_all, dom_all)
    auc_zs_raw = _probe(zs_all, dom_all)

    # Centered Z_c: remove per-domain mean (test covariance residual only)
    zc_centered = zc_all.copy()
    for d in [0, 1]:
        idx_d = (dom_all == d)
        if idx_d.sum() > 1:
            zc_centered[idx_d] -= zc_centered[idx_d].mean(0)
    auc_zc_centered = _probe(zc_centered, dom_all)

    print(f"""
============================================================
  Domain Probe Results — {args.model}
  Checkpoint: {args.ckpt}
============================================================
  Z_c raw AUC      = {auc_zc_raw:.4f}   (want < 0.65 — domain expelled)
  Z_c centered AUC = {auc_zc_centered:.4f}   (covariance residual)
  Z_s raw AUC      = {auc_zs_raw:.4f}   (want > 0.80 — domain concentrated)
============================================================
""", flush=True)


if __name__ == "__main__":
    main()
