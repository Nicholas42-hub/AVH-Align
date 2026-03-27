"""
Standalone cross-validated domain probe for A29 (or any E2EModelA29 checkpoint).

The in-training probe (fit+eval on same 400 samples, d=512) always returns
AUC=1.0 due to n < d overfitting. This script uses 5-fold StratifiedKFold CV
to get the TRUE out-of-sample AUC.

Usage:
    python eval_cv_probe_a29.py --config configs/A29_e2e.yaml \
                                 --ckpt   outputs_A29_e2e/ckpts/last.ckpt \
                                 --n_probe 400
"""

import argparse, os, sys
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    enc_list, zs_list, dom_list = [], [], []
    for batch in loader:
        if batch is None:
            continue
        video_raw, audio_raw, _, domain_labels, _ = batch
        video_raw  = video_raw.to(device)
        audio_raw  = audio_raw.to(device)
        video_feats, audio_feats = model.encoder(video_raw, audio_raw)
        z_sync, _, _ = model.head._encode(video_feats, audio_feats)
        enc_feats = torch.cat([video_feats.mean(1), audio_feats.mean(1)], dim=-1)
        enc_list.append(enc_feats.cpu().numpy())
        zs_list.append(z_sync.mean(1).cpu().numpy())
        dom_list.extend(domain_labels.tolist())
    return (np.concatenate(enc_list),
            np.concatenate(zs_list),
            np.array(dom_list))


def cv_auc(X, y, n_folds=5):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"),
    )
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--ckpt",    required=True)
    ap.add_argument("--n_probe", type=int, default=400,
                    help="samples per domain for probe (default 400 each)")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    config = load_config(args.config)
    hp     = config["model_hparams"]

    # ── Imports that depend on path ────────────────────────────────────────────
    from train_e2e_a29 import E2EModelA29, DomainLabeledDataset
    from datasets_e2e  import AV1M_E2E_FullPathDataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn

    # ── Build model ────────────────────────────────────────────────────────────
    model = E2EModelA29(config=config)
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval_cv_probe] Loaded {args.ckpt}")
    print(f"  Missing: {missing[:5]}  Unexpected: {unexpected[:3]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ── Build probe loader ─────────────────────────────────────────────────────
    dp         = config["data_paths"]
    max_frames = int(hp.get("max_frames", 150))
    n_probe    = args.n_probe

    val_csv = dp["e2e_val_csv"]
    av1m_base   = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    av1m_probe  = DomainLabeledDataset(
        Subset(av1m_base, list(range(min(n_probe, len(av1m_base))))),
        domain_label=0,
    )

    favc_root   = dp.get("favc_raw_root", "")
    favc_base   = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
    rng         = np.random.default_rng(0)
    favc_idx    = rng.choice(len(favc_base), size=min(n_probe, len(favc_base)), replace=False)
    favc_probe  = DomainLabeledDataset(Subset(favc_base, favc_idx.tolist()), domain_label=1)

    from torch.utils.data import ConcatDataset
    probe_ds  = ConcatDataset([av1m_probe, favc_probe])
    probe_loader = DataLoader(
        probe_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=e2e_collate_fn,
        drop_last=False,
    )

    print(f"[eval_cv_probe] Extracting embeddings ({len(probe_ds)} samples)...")
    enc_all, zs_all, dom_all = extract_embeddings(model, probe_loader, device)
    print(f"  enc shape: {enc_all.shape}  zs shape: {zs_all.shape}  "
          f"domain counts: {(dom_all==0).sum()} vs {(dom_all==1).sum()}")

    # ── Cross-validated probes ─────────────────────────────────────────────────
    print(f"\n[eval_cv_probe] Running {args.n_folds}-fold CV probes...", flush=True)

    auc_enc = cv_auc(enc_all, dom_all, args.n_folds)
    print(f"  enc_feats  (2048d):  CV AUC = {auc_enc:.4f}")

    auc_zs  = cv_auc(zs_all,  dom_all, args.n_folds)
    print(f"  z_sync_pool (512d):  CV AUC = {auc_zs:.4f}  ← PRIMARY (target <0.65)")

    # centered (residual beyond mean)
    zs_c = zs_all.copy()
    for d in [0, 1]:
        idx = (dom_all == d)
        if idx.sum() > 1:
            zs_c[idx] -= zs_c[idx].mean(0)
    auc_zs_c = cv_auc(zs_c, dom_all, args.n_folds)
    print(f"  z_sync_pool CENTERED: CV AUC = {auc_zs_c:.4f}")

    # mean distance diagnostic
    mu0 = zs_all[dom_all == 0].mean(0)
    mu1 = zs_all[dom_all == 1].mean(0)
    mean_dist = np.linalg.norm(mu0 - mu1)
    within_std = np.concatenate([zs_all[dom_all==0] - mu0, zs_all[dom_all==1] - mu1]).std()
    print(f"\n  ||mu0 - mu1|| = {mean_dist:.4f}  within-class std ≈ {within_std:.4f}  "
          f"(effect size d≈{mean_dist/within_std:.4f})")

    print(f"\n{'='*60}")
    print(f"  RESULT:  z_sync_pool CV AUC = {auc_zs:.4f}")
    if auc_zs < 0.65:
        print(f"  ✅ TARGET MET (<0.65)!")
    else:
        print(f"  ❌ Not yet (<0.65 needed, got {auc_zs:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
