"""
Direct Model AUC Evaluation (Causal & Spurious heads)
======================================================
Evaluates an E2EModel checkpoint using the model's own trained classification
heads (NOT a linear probe). For each head, computes AUC on:
  - In-domain  : AV1M val set   (real=0 vs fake=1)
  - Cross-domain: FakeAVCeleb   (real=0 vs fake=1)

Contrast with eval_e2e_domain.py which fits a *new* logistic regression on
mean-pooled Z_c/Z_s representations. This script uses the learned MLP heads
directly, reflecting the model's actual detection performance.

Usage
-----
  python eval_e2e_auc.py \\
      --ckpt    outputs_A1_e2e_dadv/ckpts/model-epoch=14-val_auc_causal=1.0000.ckpt \\
      --config  configs/A1_e2e_dadv.yaml \\
      --av1m_raw   /scratch/punim2637/nnliang/avd1m_preprocessed/val \\
      --av1m_csv   csv_metadata/av1m_e2e \\
      --favc_raw   /scratch/punim2637/nnliang/favc_preprocessed \\
      --output_dir outputs_A1_e2e_dadv/results \\
      --max_clips  500 --batch_size 4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_e2e import E2EModel
from datasets_e2e import (
    AV1M_E2E_Dataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn
)


# ── Score extraction (uses model's own classification heads) ──────────────────

@torch.no_grad()
def extract_scores(model: E2EModel, dataloader: DataLoader,
                   device: torch.device):
    """
    Forward pass through encoder + causal/spurious heads.

    Returns
    -------
    causal_scores  : np.ndarray [N]  – logsumexp-pooled causal head logits
    spurious_scores: np.ndarray [N]  – logsumexp-pooled spurious head logits
    labels         : np.ndarray [N]  – 0=real, 1=fake
    """
    model.eval()
    model.to(device)

    all_causal, all_spurious, all_labels = [], [], []

    for batch in tqdm(dataloader, leave=False):
        if batch is None:
            continue
        # e2e_collate_fn returns (video [B,T,1,H,W], audio [B,T,104], labels, paths)
        video, audio, cls_labels, _ = batch
        video = video.to(device)
        audio = audio.to(device)

        # encoder → [B,T,1024]
        video_feats, audio_feats = model.encoder(video, audio)

        # head.forward → logsumexp-pooled clip score [B]
        s_causal   = model.head.forward(video_feats, audio_feats, mode="causal")
        s_spurious = model.head.forward(video_feats, audio_feats, mode="spurious")

        all_causal.append(s_causal.cpu().numpy())
        all_spurious.append(s_spurious.cpu().numpy())
        all_labels.extend(cls_labels.tolist())

    return (
        np.concatenate(all_causal),
        np.concatenate(all_spurious),
        np.array(all_labels, dtype=np.int32),
    )


def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC, returning nan if only one class present."""
    valid = (labels == 0) | (labels == 1)
    s, l = scores[valid], labels[valid]
    if len(np.unique(l)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true=l, y_score=s))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Direct model AUC eval: causal & spurious heads, in-domain & cross-domain"
    )
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--av1m_raw",    required=True,
                        help="avd1m_preprocessed/val root")
    parser.add_argument("--favc_raw",    required=True,
                        help="favc_preprocessed root")
    parser.add_argument("--av1m_csv",    required=True,
                        help="csv_metadata/av1m_e2e dir with val_e2e_full.csv")
    parser.add_argument("--output_dir",  default="outputs_e2e/results")
    parser.add_argument("--max_clips",   type=int, default=500,
                        help="Max clips per dataset (balanced)")
    parser.add_argument("--max_frames",  type=int, default=150)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model = E2EModel.load_from_checkpoint(args.ckpt, config=config)
    model.eval()
    model.to(device)
    print("Model loaded OK")

    # ── AV1M val dataset (in-domain) ──────────────────────────────────────────
    av1m_ds = AV1M_E2E_Dataset(
        args.av1m_raw,
        os.path.join(args.av1m_csv, "val_labels.csv"),
        split="val", max_frames=args.max_frames)
    n_av1m = min(args.max_clips, len(av1m_ds))
    av1m_sub = Subset(av1m_ds, list(range(n_av1m)))
    av1m_loader = DataLoader(
        av1m_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=e2e_collate_fn)
    print(f"AV1M val clips: {n_av1m}")

    # ── FAVC dataset (cross-domain) ───────────────────────────────────────────
    favc_ds = FakeAVCeleb_E2E_Dataset(args.favc_raw, max_frames=args.max_frames)
    n_favc = min(args.max_clips, len(favc_ds))
    rng = np.random.default_rng(42)
    favc_idx = rng.choice(len(favc_ds), size=n_favc, replace=False).tolist()
    favc_sub = Subset(favc_ds, favc_idx)
    favc_loader = DataLoader(
        favc_sub, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=e2e_collate_fn)
    print(f"FAVC clips:     {n_favc}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nExtracting scores on AV1M val (in-domain)...")
    sc_causal_id, sc_spu_id, lbl_id = extract_scores(model, av1m_loader, device)
    print(f"  Got {len(sc_causal_id)} clips  "
          f"(real={int((lbl_id==0).sum())}, fake={int((lbl_id==1).sum())})")

    print("\nExtracting scores on FAVC (cross-domain)...")
    sc_causal_xd, sc_spu_xd, lbl_xd = extract_scores(model, favc_loader, device)
    n_real_xd = int((lbl_xd == 0).sum())
    n_fake_xd = int((lbl_xd == 1).sum())
    print(f"  Got {len(sc_causal_xd)} clips  "
          f"(real={n_real_xd}, fake={n_fake_xd})")

    # ── Compute AUC ───────────────────────────────────────────────────────────
    causal_id_auc   = safe_auc(sc_causal_id, lbl_id)
    spurious_id_auc = safe_auc(sc_spu_id,    lbl_id)
    causal_xd_auc   = safe_auc(sc_causal_xd, lbl_xd)
    spurious_xd_auc = safe_auc(sc_spu_xd,    lbl_xd)

    # ── Compose results ───────────────────────────────────────────────────────
    results = {
        "checkpoint": args.ckpt,
        "n_av1m_clips": int(len(sc_causal_id)),
        "n_favc_clips": int(len(sc_causal_xd)),
        "n_favc_real": n_real_xd,
        "n_favc_fake": n_fake_xd,
        "in_domain_auc": {
            "causal":   causal_id_auc,
            "spurious": spurious_id_auc,
        },
        "cross_domain_auc": {
            "causal":   causal_xd_auc,
            "spurious": spurious_xd_auc,
        },
    }

    out_path = os.path.join(args.output_dir, "eval_e2e_auc.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  DIRECT MODEL AUC  (trained classification heads)")
    print("=" * 62)
    print(f"  {'Split':<22}  {'Causal head':>12}  {'Spurious head':>14}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*14}")
    print(f"  {'In-domain (AV1M val)':<22}  {causal_id_auc:>12.4f}  {spurious_id_auc:>14.4f}")
    print(f"  {'Cross-domain (FAVC)':<22}  {causal_xd_auc:>12.4f}  {spurious_xd_auc:>14.4f}")
    print("=" * 62)

    gap_c = causal_id_auc - causal_xd_auc
    gap_s = spurious_id_auc - spurious_xd_auc
    print(f"  Domain generalisation gap  (in - cross):")
    print(f"    Causal:   {causal_id_auc:.4f} - {causal_xd_auc:.4f} = {gap_c:+.4f}")
    print(f"    Spurious: {spurious_id_auc:.4f} - {spurious_xd_auc:.4f} = {gap_s:+.4f}")
    print("=" * 62)

    # ── Linear-probe comparison (read existing JSON if available) ─────────────
    probe_path = os.path.join(args.output_dir, "eval_e2e_domain.json")
    if os.path.isfile(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        print("\n  Linear-probe AUC (for comparison, from eval_e2e_domain.json):")
        print(f"  {'Split':<22}  {'Z_c (probe)':>12}  {'Z_s (probe)':>12}")
        lp_id_c  = probe.get("label_probe_indomain",  {}).get("Z_c_auc_mean", float("nan"))
        lp_id_s  = probe.get("label_probe_indomain",  {}).get("Z_s_auc_mean", float("nan"))
        lp_xd_c  = probe.get("label_probe_ood_favc",  {}).get("Z_c_auc_mean", float("nan"))
        lp_xd_s  = probe.get("label_probe_ood_favc",  {}).get("Z_s_auc_mean", float("nan"))
        print(f"  {'In-domain (AV1M val)':<22}  {lp_id_c:>12.4f}  {lp_id_s:>12.4f}")
        print(f"  {'Cross-domain (FAVC)':<22}  {lp_xd_c:>12.4f}  {lp_xd_s:>12.4f}")
        print("=" * 62)


if __name__ == "__main__":
    main()
