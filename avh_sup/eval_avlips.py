"""
Cross-dataset evaluation: AVH-Sup / Causal model → AVLips.

Computes AUC and AP:
  • overall  — all fake clips vs real clips
  • (causal) — full / causal-head / spurious-head AUC

Usage (ablation / baseline):
    python eval_avlips.py \
        --ckpt  outputs_A2_seed42/ckpts/model-epoch=05.ckpt \
        --model ablation \
        --features_path ../data/avlips_features \
        --output_dir results_avlips

Usage (fcd_a6):
    python eval_avlips.py \
        --ckpt  outputs_A6_seed42/ckpts/model-epoch=09.ckpt \
        --model fcd_a6 \
        --features_path ../data/avlips_features \
        --output_dir results_avlips
"""

import argparse
import os
import sys

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from datasets import AVLips_NPZ_Dataset


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true=y_true, y_score=y_score)


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true=y_true, y_score=y_score)


def run_baseline(model, loader, device):
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
    model.eval()
    all_full, all_causal, all_spurious = [], [], []
    all_labels, all_paths = [], []
    with torch.no_grad():
        for video, audio, labels, paths in tqdm.tqdm(loader, desc="Evaluating"):
            video = video.to(device)
            audio = audio.to(device)
            full_s   = model.predict_scores(video, audio, mode="full")
            causal_s = model.predict_scores(video, audio, mode="causal")
            spur_s   = model.predict_scores(video, audio, mode="spurious")
            all_full.extend(full_s.cpu().numpy().tolist())
            all_causal.extend(causal_s.cpu().numpy().tolist())
            all_spurious.extend(spur_s.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
    return all_paths, all_full, all_causal, all_spurious, all_labels


def print_and_write(text, fh):
    print(text, flush=True)
    fh.write(text + "\n")


CAUSAL_MODELS = {"causal", "fcd", "fcd_a6", "fcd_a6_lite", "fcd_a6_sharedonly",
                 "fcd_a7", "sad_a8", "sad_a9", "causal_mi",
                 "causal_a42", "causal_a43", "causal_a44"}


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset eval on AVLips")
    parser.add_argument("--ckpt",          required=True,  help="Path to .ckpt checkpoint")
    parser.add_argument("--model",
                        choices=["baseline", "causal", "ablation", "modal", "fcd", "fcd_a6",
                                 "fcd_a6_lite", "fcd_a6_sharedonly", "fcd_a7",
                                 "sad_a8", "sad_a9", "causal_mi",
                                 "causal_a42", "causal_a43", "causal_a44"],
                        default="baseline")
    parser.add_argument("--features_path", required=True,  help="Root of avlips_features/ dir")
    parser.add_argument("--apply_l2",      action="store_true", default=True)
    parser.add_argument("--output_dir",    default="results/avlips_eval")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
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
        from mlp_causal_mi import AVH_Causal_MI
        raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = raw_ckpt["hyper_parameters"]["config"]
        model = AVH_Causal_MI(config=cfg)
        target_state = model.state_dict()
        filtered = {k: v for k, v in raw_ckpt["state_dict"].items()
                    if k in target_state and v.shape == target_state[k].shape}
        model.load_state_dict(filtered, strict=False)
    elif args.model == "causal_a42":
        from mlp_causal_a42 import AVH_Causal_A42
        model = AVH_Causal_A42.load_from_checkpoint(args.ckpt)
    elif args.model == "causal_a43":
        from mlp_causal_a43 import AVH_Causal_A43
        model = AVH_Causal_A43.load_from_checkpoint(args.ckpt)
    elif args.model == "causal_a44":
        from mlp_causal_a44 import AVH_Causal_A44
        model = AVH_Causal_A44.load_from_checkpoint(args.ckpt)
    else:  # ablation
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
            print(f"  [ablation] {len(missing)} layers skipped: {missing[:4]}", flush=True)

    model.to(device)
    model.eval()

    # ── Build dataset ─────────────────────────────────────────────────────────
    config = {
        "name":       "AVLIPS_NPZ",
        "root_path":  args.features_path,
        "apply_l2":   args.apply_l2,
    }
    dataset = AVLips_NPZ_Dataset(config)
    loader  = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=4)

    # ── Inference ─────────────────────────────────────────────────────────────
    out_file = os.path.join(args.output_dir, "eval_results.txt")
    is_causal = args.model in CAUSAL_MODELS

    with open(out_file, "w") as fh:
        header = (
            f"AVLips Cross-Dataset Evaluation\n"
            f"Model   : {args.model}\n"
            f"Ckpt    : {args.ckpt}\n"
            f"Features: {args.features_path}\n"
            + "=" * 60
        )
        print_and_write(header, fh)

        if is_causal:
            all_paths, all_full, all_causal_s, all_spur_s, all_labels = run_causal(model, loader, device)

            for tag, scores in [("full", all_full), ("causal", all_causal_s), ("spurious", all_spur_s)]:
                auc = safe_auc(all_labels, scores)
                ap  = safe_ap(all_labels, scores)
                n   = len(all_labels)
                n_r = all_labels.count(0)
                n_f = all_labels.count(1)
                line = (
                    f"\n[{tag.upper()}] N={n} (real={n_r}, fake={n_f})\n"
                    f"  AUC : {auc:.4f}" if auc is not None else f"\n[{tag.upper()}] AUC: N/A"
                )
                if auc is not None:
                    line = (
                        f"\n[{tag.upper()}] N={n} (real={n_r}, fake={n_f})\n"
                        f"  AUC : {auc:.4f}\n"
                        f"  AP  : {ap:.4f}"
                    )
                print_and_write(line, fh)
        else:
            all_paths, all_scores, all_labels = run_baseline(model, loader, device)
            auc = safe_auc(all_labels, all_scores)
            ap  = safe_ap(all_labels, all_scores)
            n   = len(all_labels)
            n_r = all_labels.count(0)
            n_f = all_labels.count(1)
            if auc is not None:
                line = (
                    f"\n[OVERALL] N={n} (real={n_r}, fake={n_f})\n"
                    f"  AUC : {auc:.4f}\n"
                    f"  AP  : {ap:.4f}"
                )
            else:
                line = f"\n[OVERALL] AUC: N/A (only one class present)"
            print_and_write(line, fh)

        print_and_write(f"\n{'='*60}", fh)
        print_and_write(f"Results saved to: {out_file}", fh)

    print(f"\nDone. Results written to {out_file}", flush=True)


if __name__ == "__main__":
    main()
