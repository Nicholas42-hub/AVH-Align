"""
Leading Silence Robustness / Trimmed-Feature Evaluation
=======================================================

Smeu et al. (CVPR 2025 Highlight) showed that FakeAVCeleb fake videos can start
with a brief leading silence in the raw audio. This helper script supports two
different evaluation modes:

1. Diagnostic prefix-zeroing on already-extracted AV-HuBERT audio features
   (`features_path=data/favc_features`, `trim_frames > 0`).
   This is useful as a stress test, but it should not be used as the final
   paper-facing leading-silence conclusion.

2. True trimmed-feature evaluation
   (`features_path=data/favc_features_trimmed`, `trim_frames=0`).
   In this mode the raw waveform leading silence has already been removed and
   AV-HuBERT features re-extracted upstream. This is the correct paper-facing
   protocol for the leading-silence question.

Usage:
  python eval_leading_silence_robustness.py \\
      --ckpt avh_sup/outputs_A5_seed43/ckpts/model-epoch=17.ckpt \\
      --model fcd \\
      --features_path data/favc_features \\
      --csv_root_path avh_sup/csv_metadata/favc_npz_matched \\
      --split test \\
      --trim_frames 1 3 5 10 \\
      --output_dir avh_sup/outputs_A5_seed43/results_silence_rob
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from datasets import FakeAVCeleb_NPZ_Dataset

FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
    "RealVideo-RealAudio",  # real clips
]


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper dataset that zeroes out the first K audio feature frames
# ─────────────────────────────────────────────────────────────────────────────

class AudioTrimDataset(Dataset):
    """Wraps FakeAVCeleb_NPZ_Dataset and zeroes out first `trim_frames` audio frames."""

    def __init__(self, base_dataset, trim_frames: int = 0):
        self.base = base_dataset
        self.trim_frames = trim_frames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, label, path = self.base[idx]
        if self.trim_frames > 0:
            # Zero out the leading K frames of audio features
            # audio shape: [T, 1024]
            k = min(self.trim_frames, audio.shape[0])
            audio = audio.clone()
            audio[:k] = 0.0
        return video, audio, label, path


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    model.eval()
    all_paths, all_full, all_causal, all_spur, all_labels = [], [], [], [], []
    with torch.no_grad():
        for video, audio, labels, paths in tqdm.tqdm(loader, desc="  inference", leave=False):
            video = video.to(device)
            audio = audio.to(device)
            full_s   = model.predict_scores(video, audio, mode="full")
            causal_s = model.predict_scores(video, audio, mode="causal")
            spur_s   = model.predict_scores(video, audio, mode="spurious")
            all_full.extend(full_s.cpu().numpy().tolist())
            all_causal.extend(causal_s.cpu().numpy().tolist())
            all_spur.extend(spur_s.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_paths.extend(paths)
    return all_paths, all_full, all_causal, all_spur, all_labels


def safe_auc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(y_true=labels, y_score=scores)


def compute_metrics(paths, scores, labels):
    """Overall + per-category AUC."""
    results = {"overall": safe_auc(labels, scores)}
    for cat in ["RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        real_cat = "RealVideo-RealAudio"
        mask = [(cat in p) or (real_cat in p) for p in paths]
        cat_s = [s for s, m in zip(scores, mask) if m]
        cat_l = [l for l, m in zip(labels, mask) if m]
        results[cat] = safe_auc(cat_l, cat_s)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           required=True)
    parser.add_argument("--model",          required=True,
                        choices=["fcd", "fcd_a6", "fcd_a6_lite", "fcd_a6_sharedonly"])
    parser.add_argument("--features_path",  required=True)
    parser.add_argument("--csv_root_path",  required=True)
    parser.add_argument("--split",          default="test")
    parser.add_argument("--apply_l2",       action="store_true", default=True)
    parser.add_argument("--trim_frames",    type=int, nargs="+", default=[0, 1, 3, 5, 10],
                        help="List of K values: zero out first K audio frames")
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model} from: {args.ckpt}", flush=True)
    if args.model == "fcd":
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

    model.to(device)
    model.eval()

    # ── Base dataset (no trimming) ─────────────────────────────────────────
    cfg = {
        "name":          "FAVC_NPZ",
        "root_path":     args.features_path,
        "csv_root_path": args.csv_root_path,
        "split":         args.split,
        "apply_l2":      args.apply_l2,
    }
    base_ds = FakeAVCeleb_NPZ_Dataset(cfg, split=args.split)

    # ── Iterate over trim values ───────────────────────────────────────────
    rows = []
    trim_values = sorted(set([0] + args.trim_frames))  # always include 0

    for k in trim_values:
        print(f"\n{'='*60}", flush=True)
        print(f"  trim_frames = {k}", flush=True)
        ds     = AudioTrimDataset(base_ds, trim_frames=k)
        loader = DataLoader(ds, shuffle=False, batch_size=1, num_workers=4)

        paths, full, causal, spur, labels = run_inference(model, loader, device)

        for head, scores in [("causal", causal), ("spurious", spur)]:
            m = compute_metrics(paths, scores, labels)
            row = {"trim_frames": k, "head": head}
            row.update({k_: v for k_, v in m.items()})
            rows.append(row)

            ovr = m["overall"]
            rvfa = m.get("RealVideo-FakeAudio", float("nan"))
            fvra = m.get("FakeVideo-RealAudio", float("nan"))
            fvfa = m.get("FakeVideo-FakeAudio", float("nan"))
            print(f"  [{head:8s}]  overall={ovr:.4f}  RV-FA={rvfa:.4f}"
                  f"  FV-RA={fvra:.4f}  FV-FA={fvfa:.4f}", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, "leading_silence_robustness.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}", flush=True)

    # ── Print degradation summary ─────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("  DEGRADATION SUMMARY (trim=K vs trim=0)", flush=True)
    print("=" * 60, flush=True)
    baseline = df[df["trim_frames"] == 0].set_index("head")

    for head in ["causal", "spurious"]:
        base_ovr = baseline.loc[head, "overall"]
        print(f"\n  {head} head (baseline overall = {base_ovr:.4f}):", flush=True)
        for k in [x for x in trim_values if x > 0]:
            row_ = df[(df["trim_frames"] == k) & (df["head"] == head)]
            if row_.empty:
                continue
            ovr_k = float(row_["overall"].values[0])
            delta = ovr_k - base_ovr
            fvra_k = float(row_["FakeVideo-RealAudio"].values[0])
            fvra_base = float(baseline.loc[head, "FakeVideo-RealAudio"])
            fvra_d = fvra_k - fvra_base
            print(f"    trim={k:2d}: overall {ovr_k:.4f} (Δ={delta:+.4f})"
                  f"  FV-RA {fvra_k:.4f} (Δ={fvra_d:+.4f})", flush=True)

    if any(x > 0 for x in trim_values):
        print("\nInterpretation (diagnostic prefix-zeroing only):", flush=True)
        print("  Treat this as a stress test on feature prefixes, not the final", flush=True)
        print("  paper-facing leading-silence result.", flush=True)
        print("  For the final conclusion, compare the same checkpoint against", flush=True)
        print("  a separate true-trimmed feature run (favc_features_trimmed).", flush=True)
    else:
        print("\nInterpretation:", flush=True)
        print("  No additional feature zeroing was applied in this run.", flush=True)
        print("  This CSV should be compared against the matching non-trimmed", flush=True)
        print("  evaluation from the same checkpoint to measure true trimming impact.", flush=True)


if __name__ == "__main__":
    main()
