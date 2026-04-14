"""
E2E Domain & Label Probe Evaluation
=====================================
Evaluates an E2EModel checkpoint on:
  1. Domain predictability (AV1M=0 vs FakeAVCeleb=1) from Z_c and Z_s
     → tests whether Z_s is domain-insensitive
  2. Label AUC (real=0 vs fake=1) from Z_c and Z_s on FAVC (OOD)
     → tests whether Z_c generalises out-of-domain

Unlike the frozen-encoder eval scripts (which use pre-extracted .npz files),
this script feeds raw preprocessed video/audio through the full e2e pipeline
so the outputs reflect the fine-tuned encoder.

Usage
-----
  python eval_e2e_domain.py \\
      --ckpt  avh_sup/outputs_A2_e2e/ckpts/model-epoch=14-val_auc_causal=0.9949.ckpt \\
      --config avh_sup/configs/A2_e2e.yaml \\
      --av1m_raw   /data/scratch/.../avd1m_preprocessed \\
      --favc_raw   /data/scratch/.../favc_preprocessed \\
      --av1m_csv   avh_sup/csv_metadata/av1m \\
      --output_dir avh_sup/outputs_A2_e2e/results \\
      --max_clips  500 --batch_size 8
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import importlib
from datasets_e2e import FakeAVCeleb_E2E_Dataset, e2e_collate_fn


# ── Representation extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_zc_zs(model, dataloader: DataLoader, device: torch.device,
                  domain_label: int):
    """
    Run raw video/audio through the full e2e pipeline.
    Returns mean-pooled Z_c, Z_s [N, D] and per-sample cls/domain labels.
    """
    model.eval()
    model.to(device)

    all_zc, all_zs, all_cls, all_dom = [], [], [], []

    for batch in tqdm(dataloader, leave=False):
        if batch is None:
            continue
        # e2e_collate_fn returns (video [B,T,1,H,W], audio [B,T,104], labels, paths)
        video, audio, cls_labels, _ = batch
        video = video.to(device)
        audio = audio.to(device)

        # encoder → [B,T,1024] each
        video_feats, audio_feats = model.encoder(video, audio)

        # head._encode → (causal_repr, spurious_repr, vae_extras)  [B,T,D] each
        causal_repr, spurious_repr, _ = model.head._encode(video_feats, audio_feats)

        # mean-pool over time → [B, D]
        zc = causal_repr.mean(dim=1).cpu().numpy()
        zs = spurious_repr.mean(dim=1).cpu().numpy()

        all_zc.append(zc)
        all_zs.append(zs)
        all_cls.extend(cls_labels.tolist())
        all_dom.extend([domain_label] * len(cls_labels))

    return (np.concatenate(all_zc),
            np.concatenate(all_zs),
            np.array(all_cls, dtype=np.int32),
            np.array(all_dom, dtype=np.int32))


# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe_auc(features: np.ndarray, labels: np.ndarray,
                     n_splits: int = 5, seed: int = 42) -> tuple:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(features, labels):
        sc = StandardScaler().fit(features[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(features[tr]), labels[tr])
        prob = clf.predict_proba(sc.transform(features[te]))[:, 1]
        try:
            aucs.append(roc_auc_score(labels[te], prob))
        except ValueError:
            pass
    return float(np.mean(aucs)), float(np.std(aucs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--av1m_raw",    required=True,
                        help="avd1m_preprocessed/ root")
    parser.add_argument("--favc_raw",    required=True,
                        help="favc_preprocessed/ root")
    parser.add_argument("--av1m_csv",    required=True,
                        help="csv_metadata/av1m/ dir with val_labels.csv")
    parser.add_argument("--output_dir",  default="outputs_A2_e2e/results")
    parser.add_argument("--max_clips",   type=int, default=500,
                        help="Max clips per dataset for domain probe (balanced)")
    parser.add_argument("--max_frames",  type=int, default=150)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--holdout_only", action="store_true",
                        help="Evaluate only on the 20%% holdout not seen during training")
    parser.add_argument("--train_seed",  type=int, default=42,
                        help="Seed used for 80/20 split during training (default 42)")
    parser.add_argument("--model_module", default="train_e2e",
                        help="Python module containing the model class (default: train_e2e)")
    parser.add_argument("--model_class",  default="E2EModel",
                        help="Model class name to load (default: E2EModel)")
    parser.add_argument("--use_fullpath_csv", action="store_true",
                        help="Use AV1M_E2E_FullPathDataset with val_e2e_full.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    _mod = importlib.import_module(args.model_module)
    ModelCls = getattr(_mod, args.model_class)
    model = ModelCls.load_from_checkpoint(args.ckpt, config=config, strict=False)
    model.eval()
    model.to(device)
    print("Model loaded OK")

    # ── AV1M val dataset ──────────────────────────────────────────────────────
    if args.use_fullpath_csv:
        from datasets_e2e import AV1M_E2E_FullPathDataset
        av1m_ds = AV1M_E2E_FullPathDataset(
            os.path.join(args.av1m_csv, "val_e2e_full.csv"),
            max_frames=args.max_frames)
    else:
        from datasets_e2e import AV1M_E2E_Dataset
        av1m_ds = AV1M_E2E_Dataset(
            args.av1m_raw,
            os.path.join(args.av1m_csv, "val_labels.csv"),
            split="val", max_frames=args.max_frames)

    if args.holdout_only:
        # Reproduce the exact 80/20 split used during training so we only
        # evaluate on the 400 clips that were NEVER seen by the model.
        from torch.utils.data import random_split as _rs
        n_total   = len(av1m_ds)
        n_train   = int(n_total * 0.8)
        n_holdout = n_total - n_train
        _, holdout_split = _rs(
            av1m_ds, [n_train, n_holdout],
            generator=torch.Generator().manual_seed(args.train_seed))
        av1m_sub = holdout_split
        print(f"[holdout_only] using {n_holdout} clips not seen during training")
    else:
        n_av1m   = min(args.max_clips, len(av1m_ds))
        av1m_sub = Subset(av1m_ds, list(range(n_av1m)))
    av1m_loader = DataLoader(av1m_sub, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=e2e_collate_fn)

    # ── FakeAVCeleb dataset ───────────────────────────────────────────────────
    favc_ds = FakeAVCeleb_E2E_Dataset(args.favc_raw, max_frames=args.max_frames)
    n_favc = min(args.max_clips, len(favc_ds))
    rng = np.random.default_rng(42)
    favc_idx = rng.choice(len(favc_ds), size=n_favc, replace=False).tolist()
    favc_sub = Subset(favc_ds, favc_idx)
    favc_loader = DataLoader(favc_sub, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=e2e_collate_fn)

    # ── Extract representations ───────────────────────────────────────────────
    print("Extracting AV1M Z_c / Z_s...")
    zc_av1m, zs_av1m, cls_av1m, dom_av1m = extract_zc_zs(
        model, av1m_loader, device, domain_label=0)
    print(f"  AV1M: {len(zc_av1m)} clips")

    print("Extracting FAVC Z_c / Z_s...")
    zc_favc, zs_favc, cls_favc, dom_favc = extract_zc_zs(
        model, favc_loader, device, domain_label=1)
    print(f"  FAVC: {len(zc_favc)} clips")

    # ── Combined for domain probe ─────────────────────────────────────────────
    zc_all   = np.concatenate([zc_av1m, zc_favc])
    zs_all   = np.concatenate([zs_av1m, zs_favc])
    dom_all  = np.concatenate([dom_av1m, dom_favc])

    print("\n--- Domain Probe (AV1M vs FAVC) ---")
    dom_zc_mean, dom_zc_std = linear_probe_auc(zc_all, dom_all)
    dom_zs_mean, dom_zs_std = linear_probe_auc(zs_all, dom_all)
    print(f"  Z_c domain AUC: {dom_zc_mean:.4f} ± {dom_zc_std:.4f}")
    print(f"  Z_s domain AUC: {dom_zs_mean:.4f} ± {dom_zs_std:.4f}")
    print(f"  → Z_s domain-insensitive? {'YES' if dom_zs_mean < 0.65 else 'NO'} "
          f"(target < 0.65, got {dom_zs_mean:.4f})")

    # ── Label probe on AV1M (in-domain) ──────────────────────────────────────
    print("\n--- Label Probe: In-domain (AV1M val) ---")
    lbl_zc_id_mean, lbl_zc_id_std = linear_probe_auc(zc_av1m, cls_av1m)
    lbl_zs_id_mean, lbl_zs_id_std = linear_probe_auc(zs_av1m, cls_av1m)
    print(f"  Z_c label AUC (in-domain):  {lbl_zc_id_mean:.4f} ± {lbl_zc_id_std:.4f}")
    print(f"  Z_s label AUC (in-domain):  {lbl_zs_id_mean:.4f} ± {lbl_zs_id_std:.4f}")

    # ── Label probe on FAVC (OOD) ─────────────────────────────────────────────
    # FAVC has many fake + some real clips; keep only clips with valid label (0 or 1)
    valid = (cls_favc == 0) | (cls_favc == 1)
    zc_favc_v = zc_favc[valid]
    zs_favc_v = zs_favc[valid]
    cls_favc_v = cls_favc[valid]
    n_real = (cls_favc_v == 0).sum()
    n_fake = (cls_favc_v == 1).sum()
    print(f"\n--- Label Probe: OOD (FAVC, real={n_real}, fake={n_fake}) ---")
    if n_real > 0 and n_fake > 0 and len(np.unique(cls_favc_v)) > 1:
        lbl_zc_ood_mean, lbl_zc_ood_std = linear_probe_auc(zc_favc_v, cls_favc_v)
        lbl_zs_ood_mean, lbl_zs_ood_std = linear_probe_auc(zs_favc_v, cls_favc_v)
        print(f"  Z_c label AUC (OOD FAVC):   {lbl_zc_ood_mean:.4f} ± {lbl_zc_ood_std:.4f}")
        print(f"  Z_s label AUC (OOD FAVC):   {lbl_zs_ood_mean:.4f} ± {lbl_zs_ood_std:.4f}")
    else:
        print("  Skipping: need both real and fake FAVC clips. "
              "Check FakeAVCeleb dataset has RealVideo-RealAudio.")
        lbl_zc_ood_mean = lbl_zc_ood_std = lbl_zs_ood_mean = lbl_zs_ood_std = None

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "checkpoint":              args.ckpt,
        "n_av1m_clips":            int(len(zc_av1m)),
        "n_favc_clips":            int(len(zc_favc)),
        "domain_probe": {
            "Z_c_auc_mean":        dom_zc_mean,
            "Z_c_auc_std":         dom_zc_std,
            "Z_s_auc_mean":        dom_zs_mean,
            "Z_s_auc_std":         dom_zs_std,
        },
        "label_probe_indomain": {
            "Z_c_auc_mean":        lbl_zc_id_mean,
            "Z_c_auc_std":         lbl_zc_id_std,
            "Z_s_auc_mean":        lbl_zs_id_mean,
            "Z_s_auc_std":         lbl_zs_id_std,
        },
        "label_probe_ood_favc": {
            "Z_c_auc_mean":        lbl_zc_ood_mean,
            "Z_c_auc_std":         lbl_zc_ood_std,
            "Z_s_auc_mean":        lbl_zs_ood_mean,
            "Z_s_auc_std":         lbl_zs_ood_std,
        },
    }

    out_path = os.path.join(args.output_dir,
                            "eval_e2e_domain_holdout.json" if args.holdout_only
                            else "eval_e2e_domain.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Metric':<35} {'Z_c':>8}  {'Z_s':>8}")
    print(f"{'Domain AUC (AV1M vs FAVC)':<35} {dom_zc_mean:>8.4f}  {dom_zs_mean:>8.4f}")
    print(f"{'Label AUC in-domain (AV1M val)':<35} {lbl_zc_id_mean:>8.4f}  {lbl_zs_id_mean:>8.4f}")
    if lbl_zc_ood_mean is not None:
        print(f"{'Label AUC OOD (FAVC)':<35} {lbl_zc_ood_mean:>8.4f}  {lbl_zs_ood_mean:>8.4f}")
    print("="*60)
    print("\nFrozen A2 baseline (for reference):")
    print("  Z_c domain AUC ≈ N/A   Z_s domain AUC ≈ N/A")
    print("  Z_c label OOD  ≈ 0.706  Z_s label OOD  ≈ 0.739")


if __name__ == "__main__":
    main()
