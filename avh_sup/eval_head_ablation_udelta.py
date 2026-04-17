"""
Head ablation evaluation for AVH_FCD_A6 (u_Δ variants).

Runs four conditions per checkpoint:
  1. baseline   — full model (causal head, all factors)
  2. mask_udelta — zero out u_Δ at inference (measure FV-RA drop)
  3. mask_uv    — zero out u_v at inference
  4. mask_ua    — zero out u_a at inference

Reports per-category AUC/AP for each condition on FakeAVCeleb.

Usage:
    python eval_head_ablation_udelta.py \
        --ckpt  outputs_A6_udelta_v2_trainvalreal_seed43/ckpts/model-epoch=XX.ckpt \
        --features_path ../data/favc_features \
        --csv_root_path csv_metadata/favc \
        --output_dir    outputs_A6_udelta_v2_trainvalreal_seed43/results_ablation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from mlp_fcd_a6 import AVH_FCD_A6
from datasets import FakeAVCeleb_NPZ_Dataset

FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]

REAL_CAT = FakeAVCeleb_NPZ_Dataset.REAL_CATEGORY


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true=y_true, y_score=y_score)


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true=y_true, y_score=y_score)


# ── Ablation forward passes ───────────────────────────────────────────────────

def _score_with_mask(model, video, audio, mask_factor: str):
    """
    Run masked forward pass. mask_factor in {'none','udelta','uv','ua'}.
    Returns per-sample causal score tensor.
    """
    model.eval()
    with torch.no_grad():
        causal_repr, spurious_repr, comp = model._encode(video, audio)

        if mask_factor == "udelta":
            # Zero out u_Δ slice in causal_repr
            # causal_repr = [s_v | u_v | u_delta | s_a | u_a]
            # dims: syn_dim=256, spec_dim=256, incon_dim=256, syn_dim=256, spec_dim=256
            hp = model.hparams["config"]["model_hparams"]
            syn_dim   = int(hp.get("syn_dim",   256))
            spec_dim  = int(hp.get("spec_dim",  256))
            start = syn_dim + spec_dim                       # after s_v, u_v
            end   = start + int(hp.get("incon_dim", 256))   # u_delta slice
            causal_repr = causal_repr.clone()
            causal_repr[..., start:end] = 0.0

        elif mask_factor == "uv":
            hp = model.hparams["config"]["model_hparams"]
            syn_dim  = int(hp.get("syn_dim",  256))
            spec_dim = int(hp.get("spec_dim", 256))
            start = syn_dim                  # after s_v
            end   = syn_dim + spec_dim       # u_v slice
            causal_repr = causal_repr.clone()
            causal_repr[..., start:end] = 0.0

        elif mask_factor == "ua":
            hp = model.hparams["config"]["model_hparams"]
            syn_dim   = int(hp.get("syn_dim",   256))
            spec_dim  = int(hp.get("spec_dim",  256))
            incon_dim = int(hp.get("incon_dim", 256))
            # causal_repr = [s_v(256) | u_v(256) | u_delta(256) | s_a(256) | u_a(256)]
            start = syn_dim + spec_dim + incon_dim + syn_dim   # after s_v,u_v,u_delta,s_a
            end   = start + spec_dim                           # u_a slice
            causal_repr = causal_repr.clone()
            causal_repr[..., start:end] = 0.0

        # score via causal_head (logsumexp over time)
        per_frame = model.causal_head(causal_repr)[..., 0]
        score = torch.logsumexp(per_frame, dim=-1)
    return score


def run_ablations(model, loader, device):
    """Returns dict: condition -> (paths, scores, labels)"""
    results = {c: ([], [], []) for c in ["baseline", "mask_udelta", "mask_uv", "mask_ua"]}

    model.eval()
    for video, audio, labels, paths in tqdm.tqdm(loader, desc="Ablation eval"):
        video  = video.to(device)
        audio  = audio.to(device)
        labels_list = labels.numpy().tolist()

        for condition in results:
            mask = "none" if condition == "baseline" else condition.replace("mask_", "")
            scores = _score_with_mask(model, video, audio, mask_factor=mask)
            results[condition][0].extend(paths)
            results[condition][1].extend(scores.cpu().numpy().tolist())
            results[condition][2].extend(labels_list)

    return results


def per_category_metrics(paths, scores, labels):
    out = {}
    out["overall"] = {
        "auc": safe_auc(labels, scores),
        "ap":  safe_ap(labels, scores),
        "n":   len(labels),
    }
    for cat in FAKE_CATEGORIES:
        mask = [(cat in p) or (REAL_CAT in p) for p in paths]
        s = [x for x, m in zip(scores, mask) if m]
        l = [x for x, m in zip(labels, mask) if m]
        out[cat] = {"auc": safe_auc(l, s), "ap": safe_ap(l, s), "n": len(l)}
    return out


def print_and_write(text, fh):
    print(text, flush=True)
    fh.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           required=True)
    parser.add_argument("--features_path",  required=True)
    parser.add_argument("--csv_root_path",  default="csv_metadata/favc")
    parser.add_argument("--split",          default="test")
    parser.add_argument("--output_dir",     default="results/ablation_udelta")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading A6 u_Δ model from: {args.ckpt}", flush=True)
    model = AVH_FCD_A6.load_from_checkpoint(args.ckpt)
    model.to(device)
    model.eval()

    config = {
        "name":          "FAVC_NPZ",
        "root_path":     args.features_path,
        "csv_root_path": args.csv_root_path,
        "split":         args.split,
        "apply_l2":      True,
    }
    dataset = FakeAVCeleb_NPZ_Dataset(config, split=args.split)
    loader  = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

    ablation_results = run_ablations(model, loader, device)

    out_file = os.path.join(args.output_dir, "ablation_results.txt")
    with open(out_file, "w") as fh:
        header = (
            f"A6 u_\u0394 Head Ablation — FakeAVCeleb\n"
            f"Ckpt    : {args.ckpt}\n"
            f"Features: {args.features_path}\n"
            f"Split   : {args.split}\n"
            + "=" * 70
        )
        print_and_write(header, fh)

        fvra_summary = {}

        for condition, (paths, scores, labels) in ablation_results.items():
            print_and_write(f"\n  ── {condition.upper()} ──", fh)
            metrics = per_category_metrics(paths, scores, labels)
            for key, m in metrics.items():
                auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
                ap  = f"{m['ap']:.4f}"  if m["ap"]  is not None else "  N/A "
                print_and_write(f"  {key:<32}  AUC={auc}  AP={ap}  N={m['n']}", fh)
            fv_ra_auc = metrics.get("FakeVideo-RealAudio", {}).get("auc")
            fvra_summary[condition] = fv_ra_auc

        # Summary delta table
        print_and_write("\n" + "=" * 70, fh)
        print_and_write("  FV-RA AUC DELTA (vs baseline):", fh)
        baseline_fvra = fvra_summary.get("baseline")
        for condition, auc in fvra_summary.items():
            if condition == "baseline" or auc is None or baseline_fvra is None:
                delta_str = ""
            else:
                delta_str = f"  Δ={auc - baseline_fvra:+.4f}"
            auc_str = f"{auc:.4f}" if auc is not None else "  N/A "
            print_and_write(f"  {condition:<20}  FV-RA AUC={auc_str}{delta_str}", fh)
        print_and_write("=" * 70, fh)

    # Save raw predictions
    for condition, (paths, scores, labels) in ablation_results.items():
        pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
            os.path.join(args.output_dir, f"predictions_{condition}.csv"), index=False
        )

    print(f"\nDone. Results at: {out_file}")


if __name__ == "__main__":
    main()
