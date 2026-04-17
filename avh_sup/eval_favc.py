"""
Cross-dataset evaluation: AVH-Sup (baseline or causal) → FakeAVCeleb.

Computes AUC and AP:
  • overall          — all fake categories vs real
  • per-category     — RealVideo-FakeAudio / FakeVideo-RealAudio / FakeVideo-FakeAudio
  • (causal model)   — full / causal-head / spurious-head AUC

Usage (baseline):
    python eval_favc.py \
        --ckpt  outputs/ckpts/model-epoch=05.ckpt \
        --features_path ../data/favc_features \
        --csv_root_path csv_metadata/favc \
        --split test \
        --output_dir results/favc_baseline

Usage (causal):
    python eval_favc.py \
        --ckpt  outputs_causal/ckpts/model-epoch=09.ckpt \
        --model causal \
        --features_path ../data/favc_features \
        --csv_root_path csv_metadata/favc \
        --split test \
        --output_dir results/favc_causal
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

from datasets import FakeAVCeleb_NPZ_Dataset

FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]


def safe_auc(y_true, y_score):
    """Return AUC or None if only one class present."""
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true=y_true, y_score=y_score)


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true=y_true, y_score=y_score)


def run_baseline(model, loader, device):
    """Inference with AVH_Sup baseline — returns (paths, scores, labels)."""
    model.eval()
    all_scores, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for video, audio, labels, paths in tqdm.tqdm(loader, desc="Evaluating"):
            video = video.to(device)
            audio = audio.to(device)
            scores = model.predict_scores(video, audio)
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
    return all_paths, all_scores, all_labels


def run_causal(model, loader, device):
    """Inference with AVH_Causal — returns (paths, full_scores, causal_scores,
    spurious_scores, labels).  predict_scores(v, a, mode=...) returns one score per sample."""
    model.eval()
    all_full, all_causal, all_spurious = [], [], []
    all_labels, all_paths = [], []
    with torch.no_grad():
        for video, audio, labels, paths in tqdm.tqdm(loader, desc="Evaluating"):
            video = video.to(device)
            audio = audio.to(device)
            full_s     = model.predict_scores(video, audio, mode="full")
            causal_s   = model.predict_scores(video, audio, mode="causal")
            spur_s     = model.predict_scores(video, audio, mode="spurious")
            all_full.extend(full_s.cpu().numpy().tolist())
            all_causal.extend(causal_s.cpu().numpy().tolist())
            all_spurious.extend(spur_s.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
    return all_paths, all_full, all_causal, all_spurious, all_labels


def per_category_metrics(paths, scores, labels):
    """Compute AUC/AP for each fake category vs. real."""
    results = {}

    # Overall (all clips)
    results["overall"] = {
        "auc": safe_auc(labels, scores),
        "ap":  safe_ap(labels, scores),
        "n":   len(labels),
        "n_real": labels.count(0),
        "n_fake": labels.count(1),
    }

    # Per fake category  (one-vs-real)
    for cat in FAKE_CATEGORIES:
        # mask = real clips OR this fake category
        cat_mask = [
            (cat in p) or (FakeAVCeleb_NPZ_Dataset.REAL_CATEGORY in p)
            for p in paths
        ]
        cat_scores = [s for s, m in zip(scores, cat_mask) if m]
        cat_labels = [l for l, m in zip(labels, cat_mask) if m]
        results[cat] = {
            "auc": safe_auc(cat_labels, cat_scores),
            "ap":  safe_ap(cat_labels, cat_scores),
            "n":   len(cat_labels),
        }

    return results


def print_and_write(text, fh):
    print(text, flush=True)
    fh.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset eval on FakeAVCeleb")
    parser.add_argument("--ckpt",           required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--model",          choices=["baseline", "causal", "ablation", "modal", "fcd", "fcd_a6", "fcd_a6_lite", "fcd_a6_sharedonly", "fcd_a7", "sad_a8", "sad_a9", "causal_mi", "causal_a42", "causal_a43", "causal_a44", "daof"], default="baseline")
    parser.add_argument("--features_path",  required=True, help="Root of favc_features/ dir")
    parser.add_argument("--csv_root_path",  default="csv_metadata/favc",
                        help="Dir with {split}_split.csv files")
    parser.add_argument("--split",          default="test",
                        choices=["train", "val", "test"],
                        help="Which split CSV to use for filtering (default: test)")
    parser.add_argument("--apply_l2",       action="store_true", default=True)
    parser.add_argument("--output_dir",     default="results/favc_eval")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    print(f"Loading {args.model} model from: {args.ckpt}", flush=True)

    if args.model == "baseline":
        from mlp import AVH_Sup
        model = AVH_Sup.load_from_checkpoint(args.ckpt)
    elif args.model == "causal":
        from mlp_causal import AVH_Causal
        model = AVH_Causal.load_from_checkpoint(args.ckpt)
    elif args.model == "modal":
        from mlp_causal_modal import AVH_Causal_Modal
        model = AVH_Causal_Modal.load_from_checkpoint(args.ckpt)
    elif args.model == "fcd":
        from mlp_fcd import AVH_FCD
        model = AVH_FCD.load_from_checkpoint(args.ckpt)
    elif args.model == "fcd_a6":
        from mlp_fcd_a6 import AVH_FCD_A6
        model = AVH_FCD_A6.load_from_checkpoint(args.ckpt)
    elif args.model == "fcd_a6_lite":
        from mlp_fcd_a6_lite import AVH_FCD_A6_Lite
        model = AVH_FCD_A6_Lite.load_from_checkpoint(args.ckpt)
    elif args.model == "fcd_a6_sharedonly":
        from mlp_fcd_a6_sharedonly import AVH_FCD_A6_SharedOnly
        model = AVH_FCD_A6_SharedOnly.load_from_checkpoint(args.ckpt)
    elif args.model == "fcd_a7":
        from mlp_fcd_a7 import AVH_FCD_A7
        model = AVH_FCD_A7.load_from_checkpoint(args.ckpt)
    elif args.model == "sad_a8":
        from mlp_sad_a8 import AVH_SAD_A8
        model = AVH_SAD_A8.load_from_checkpoint(args.ckpt)
    elif args.model == "sad_a9":
        from mlp_sad_a9 import AVH_SAD_A9
        model = AVH_SAD_A9.load_from_checkpoint(args.ckpt)
    elif args.model == "causal_mi":
        # Checkpoint may have been saved from the MINE variant (mi_critic_spu layer)
        # while the current mlp_causal_mi.py uses CLUB (club_predictor layer).
        # Use shape-filtered loading so inference-relevant layers (projections +
        # classification heads) load correctly even if the critic layer differs.
        from mlp_causal_mi import AVH_Causal_MI
        raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = raw_ckpt["hyper_parameters"]["config"]
        model = AVH_Causal_MI(config=cfg)
        target_state = model.state_dict()
        filtered = {k: v for k, v in raw_ckpt["state_dict"].items()
                    if k in target_state and v.shape == target_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        missing = [k for k in target_state if k not in filtered]
        if missing:
            print(f"  [causal_mi] {len(missing)} layers not loaded (critic mismatch): "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}", flush=True)
    elif args.model == "causal_a42":
        from mlp_causal_a42 import AVH_Causal_A42
        model = AVH_Causal_A42.load_from_checkpoint(args.ckpt)
    elif args.model == "causal_a43":
        from mlp_causal_a43 import AVH_Causal_A43
        model = AVH_Causal_A43.load_from_checkpoint(args.ckpt)
    elif args.model == "causal_a44":
        from mlp_causal_a44 import AVH_Causal_A44
        model = AVH_Causal_A44.load_from_checkpoint(args.ckpt)
    elif args.model == "daof":
        from mlp_fcd_daof import AVH_FCD_DAOF
        model = AVH_FCD_DAOF.load_from_checkpoint(args.ckpt)
    else:  # ablation
        # Use shape-filtered loading: the A2 checkpoint was saved with a
        # full_head input dim that may differ from the current model definition
        # (e.g. 1024 in ckpt vs 2048 in model when Z_s projection size changed).
        # Loading with strict=False keeps all matching layers intact.
        from mlp_causal_ablation import AVH_Causal_Ablation
        raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = raw_ckpt["hyper_parameters"]["config"]
        model = AVH_Causal_Ablation(config=cfg)
        target_state = model.state_dict()
        filtered = {k: v for k, v in raw_ckpt["state_dict"].items()
                    if k in target_state and v.shape == target_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        missing = [k for k in target_state if k not in filtered]
        if missing:
            print(f"  [ablation] {len(missing)} layers skipped (shape mismatch): "
                  f"{missing[:4]}{'...' if len(missing) > 4 else ''}", flush=True)

    model.to(device)
    model.eval()

    # ── Build dataset ─────────────────────────────────────────────────────────
    config = {
        "name":          "FAVC_NPZ",
        "root_path":     args.features_path,
        "csv_root_path": args.csv_root_path,
        "split":         args.split,
        "apply_l2":      args.apply_l2,
    }
    dataset = FakeAVCeleb_NPZ_Dataset(config, split=args.split)
    loader  = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

    # ── Inference ─────────────────────────────────────────────────────────────
    out_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(out_file, "w") as fh:
        header = (
            f"FakeAVCeleb Cross-Dataset Evaluation\n"
            f"Model   : {args.model}\n"
            f"Ckpt    : {args.ckpt}\n"
            f"Features: {args.features_path}\n"
            f"Split   : {args.split}\n"
            + "=" * 60
        )
        print_and_write(header, fh)

        if args.model == "baseline":
            paths, scores, labels = run_baseline(model, loader, device)

            # Save raw predictions
            pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
                os.path.join(args.output_dir, "predictions.csv"), index=False
            )

            metrics = per_category_metrics(paths, scores, labels)
            for key, m in metrics.items():
                auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
                ap  = f"{m['ap']:.4f}"  if m["ap"]  is not None else "  N/A "
                n   = m["n"]
                print_and_write(f"  {key:<30}  AUC={auc}  AP={ap}  N={n}", fh)

        else:  # causal or ablation (same predict_scores interface)
            paths, full, causal, spurious, labels = run_causal(model, loader, device)

            pd.DataFrame({
                "path": paths,
                "score_full": full,
                "score_causal": causal,
                "score_spurious": spurious,
                "label": labels,
            }).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

            for head_name, scores in [("full", full), ("causal", causal), ("spurious", spurious)]:
                print_and_write(f"\n  ── {head_name.upper()} HEAD ──", fh)
                metrics = per_category_metrics(paths, scores, labels)
                for key, m in metrics.items():
                    auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
                    ap  = f"{m['ap']:.4f}"  if m["ap"]  is not None else "  N/A "
                    n   = m["n"]
                    print_and_write(f"  {key:<30}  AUC={auc}  AP={ap}  N={n}", fh)

        print_and_write("=" * 60, fh)
        print_and_write(f"Results saved to: {args.output_dir}", fh)

    print(f"\nDone. Results at: {out_file}")


if __name__ == "__main__":
    main()
