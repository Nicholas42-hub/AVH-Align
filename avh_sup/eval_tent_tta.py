"""TENT-style test-time adaptation on FAVC for TriRouteUnsupSync.

Loads a baseline TriRoute-unsup checkpoint, freezes everything except
LayerNorm affine (gamma/beta), then adapts on FAVC test data via entropy
minimization on softmax(sync_logits). Reports AUC before and after.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import FakeAVCeleb_NPZ_Dataset
from train_eval_triroute_unsup_sync import (
    TriRouteUnsupSync,
    eval_loader,
    per_category_metrics,
)


def configure_model_for_tent(model: nn.Module):
    """Freeze all params except LayerNorm affine. Returns LN param list."""
    for p in model.parameters():
        p.requires_grad = False
    ln_params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                m.weight.requires_grad = True
                ln_params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad = True
                ln_params.append(m.bias)
    return ln_params


def adapt_one_pass(model, loader, optimizer, device):
    """One pass over loader doing entropy-min on each clip's sync logits."""
    losses = []
    model.train()
    for video, audio, _label, _path in loader:
        video = video.to(device)
        audio = audio.to(device)
        if video.ndim == 2:
            video = video.unsqueeze(0)
            audio = audio.unsqueeze(0)
        logits = model.logits(video, audio)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        optimizer.zero_grad()
        ent.backward()
        optimizer.step()
        losses.append(float(ent.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def report(label, paths, scores, labels):
    metrics = per_category_metrics(paths, scores, labels)
    print(f"\n=== {label} ===")
    for k, m in metrics.items():
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "N/A"
        ap = f"{m['ap']:.4f}" if m["ap"] is not None else "N/A"
        print(f"  {k:<30} AUC={auc}  AP={ap}  N={m['n']}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--favc_features", default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    parser.add_argument("--favc_csv_root", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1, help="Number of adaptation passes over FAVC.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TriRouteUnsupSync(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded ckpt: {args.ckpt}  (epoch={ckpt.get('epoch')}, overall_auc@save={ckpt.get('overall_auc')})")

    favc_cfg = {"root_path": args.favc_features, "csv_root_path": args.favc_csv_root, "apply_l2": True}
    favc_ds = FakeAVCeleb_NPZ_Dataset(favc_cfg, split="test")
    print(f"FAVC test set: N={len(favc_ds)}")
    favc_loader = DataLoader(favc_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Pre-adaptation eval
    paths, scores, labels = eval_loader(model, favc_loader, device)
    metrics_before = report("BEFORE adaptation", paths, scores, labels)
    pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
        os.path.join(args.output_dir, "predictions_before.csv"), index=False
    )

    # Configure for TENT
    ln_params = configure_model_for_tent(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable LN params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.Adam(ln_params, lr=args.lr)
    favc_loader_shuffle = DataLoader(favc_ds, batch_size=1, shuffle=True, num_workers=args.num_workers)
    for step in range(args.steps):
        mean_ent = adapt_one_pass(model, favc_loader_shuffle, optimizer, device)
        print(f"Adapt pass {step+1}/{args.steps}: mean_entropy={mean_ent:.4f}")

    # Post-adaptation eval
    paths, scores, labels = eval_loader(model, favc_loader, device)
    metrics_after = report("AFTER adaptation", paths, scores, labels)
    pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
        os.path.join(args.output_dir, "predictions_after.csv"), index=False
    )

    # Delta
    print("\n=== Δ (after - before) ===")
    for k in metrics_before:
        d_auc = (metrics_after[k]["auc"] or 0) - (metrics_before[k]["auc"] or 0)
        print(f"  {k:<30} ΔAUC={d_auc:+.4f}")

    with open(os.path.join(args.output_dir, "tent_results.txt"), "w") as f:
        f.write(f"Ckpt: {args.ckpt}\n")
        f.write(f"FAVC N={len(favc_ds)}, lr={args.lr}, steps={args.steps}\n")
        f.write("=" * 60 + "\n")
        f.write("BEFORE:\n")
        for k, m in metrics_before.items():
            f.write(f"  {k:<30} AUC={m['auc']:.4f} AP={m['ap']:.4f}\n")
        f.write("AFTER:\n")
        for k, m in metrics_after.items():
            f.write(f"  {k:<30} AUC={m['auc']:.4f} AP={m['ap']:.4f}\n")
        f.write("Delta:\n")
        for k in metrics_before:
            d_auc = (metrics_after[k]["auc"] or 0) - (metrics_before[k]["auc"] or 0)
            f.write(f"  {k:<30} ΔAUC={d_auc:+.4f}\n")


if __name__ == "__main__":
    main()
