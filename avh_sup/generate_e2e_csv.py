#!/usr/bin/env python3
"""
Generate comprehensive E2E training/validation CSVs for AV1M.

Scans all available preprocessed directories, infers labels from filenames,
and writes train/val CSVs with full absolute paths for use with
AV1M_E2E_FullPathDataset.

Label inference from *_roi.mp4 filename stem:
  real                    → label=0, manip_type=0
  real_video_fake_audio   → label=1, manip_type=1  (audio-only fake)
  fake_video_real_audio   → label=1, manip_type=2  (video-only fake)
  fake_video_fake_audio   → label=1, manip_type=3  (both faked)

Output CSV columns:  roi_path, wav_path, label, manip_type

Usage:
  python generate_e2e_csv.py [--batch_train_dir ...] [--scratch_val_dir ...]
                              [--output_dir ...] [--val_fraction 0.05]
"""

import argparse
import csv
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_BATCH_TRAIN = os.path.join(
    BASE, "data", "avd1m_preprocessed_batch", "train"
)
DEFAULT_SCRATCH_VAL = "/scratch/punim2637/nnliang/avd1m_preprocessed/val"
DEFAULT_OUTPUT_DIR  = os.path.join(
    BASE, "avh_sup", "csv_metadata", "av1m_e2e"
)

FILENAME_TO_LABEL = {
    "real":                   (0, 0),
    "real_video_fake_audio":  (1, 1),
    "fake_video_real_audio":  (1, 2),
    "fake_video_fake_audio":  (1, 3),
}


def infer_label(roi_fname: str):
    """Return (label, manip_type) from *_roi.mp4 filename, or None if unknown."""
    if roi_fname.endswith("_roi.mp4"):
        stem = roi_fname[:-8]
    elif roi_fname.endswith(".mp4"):
        stem = roi_fname[:-4]
    else:
        stem = roi_fname
    return FILENAME_TO_LABEL.get(stem)


def scan_directory(base_dir: str):
    """
    Walk base_dir recursively; collect (roi_abs, wav_abs, label, manip_type)
    for every *_roi.mp4 that has a matching *.wav sibling.
    """
    items = []
    n_missing_wav  = 0
    n_unknown_lbl  = 0

    for dirpath, _, filenames in os.walk(base_dir):
        for roi_fname in filenames:
            if not roi_fname.endswith("_roi.mp4"):
                continue
            stem = roi_fname[:-8]           # strip _roi.mp4
            wav_fname = stem + ".wav"
            roi_abs = os.path.join(dirpath, roi_fname)
            wav_abs = os.path.join(dirpath, wav_fname)

            if not os.path.isfile(wav_abs):
                n_missing_wav += 1
                continue

            label_info = infer_label(roi_fname)
            if label_info is None:
                n_unknown_lbl += 1
                continue

            label, manip_type = label_info
            items.append((roi_abs, wav_abs, label, manip_type))

    print(
        f"[scan_directory] {base_dir}\n"
        f"  valid pairs  : {len(items)}\n"
        f"  missing wav  : {n_missing_wav}\n"
        f"  unknown label: {n_unknown_lbl}",
        flush=True,
    )
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive AV1M E2E training/val CSVs"
    )
    parser.add_argument(
        "--batch_train_dir", default=DEFAULT_BATCH_TRAIN,
        help="Path to avd1m_preprocessed_batch/train/ directory",
    )
    parser.add_argument(
        "--scratch_val_dir", default=DEFAULT_SCRATCH_VAL,
        help="Path to avd1m_preprocessed/val/ directory on scratch",
    )
    parser.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to write train_e2e_full.csv and val_e2e_full.csv",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.05,
        help="Fraction of all clips to hold out as validation (default 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    all_items = []

    for label, dir_path in [
        ("batch_train", args.batch_train_dir),
        ("scratch_val", args.scratch_val_dir),
    ]:
        if os.path.isdir(dir_path):
            items = scan_directory(dir_path)
            print(f"  [{label}] contributed {len(items)} clips", flush=True)
            all_items.extend(items)
        else:
            print(
                f"WARNING: {label} directory not found: {dir_path}  (skipping)",
                flush=True,
            )

    if not all_items:
        print("ERROR: no clips found — check directory paths.", flush=True)
        sys.exit(1)

    print(f"\nTotal clips found: {len(all_items)}", flush=True)

    # Label distribution summary
    n_real = sum(1 for it in all_items if it[2] == 0)
    n_fake = sum(1 for it in all_items if it[2] == 1)
    manip_dist: dict = {}
    for it in all_items:
        manip_dist[it[3]] = manip_dist.get(it[3], 0) + 1
    print(f"  real={n_real}  fake={n_fake}", flush=True)
    print(f"  manip_type distribution: {manip_dist}", flush=True)

    # Shuffle and split
    random.shuffle(all_items)
    n_val   = max(100, int(len(all_items) * args.val_fraction))
    n_train = len(all_items) - n_val
    train_items = all_items[:n_train]
    val_items   = all_items[n_train:]
    print(f"\nSplit → train={len(train_items)}  val={len(val_items)}", flush=True)

    header = ["roi_path", "wav_path", "label", "manip_type"]

    train_csv = os.path.join(args.output_dir, "train_e2e_full.csv")
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_items)
    print(f"Written: {train_csv}", flush=True)

    val_csv = os.path.join(args.output_dir, "val_e2e_full.csv")
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(val_items)
    print(f"Written: {val_csv}", flush=True)


if __name__ == "__main__":
    main()
