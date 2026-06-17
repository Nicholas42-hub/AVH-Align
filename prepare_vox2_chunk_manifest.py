#!/usr/bin/env python3
"""Create a small VoxCeleb2 chunk manifest from a zipinfo file listing.

The downstream preprocessing code reuses the AV1M path convention, so the CSV
contains relative paths such as ``dev/id00012/video/00001.mp4`` and label 0.
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-list", required=True, help="Text file from `zipinfo -1 vox2_mp4_dev.zip`.")
    parser.add_argument("--start", type=int, default=1, help="1-indexed first mp4 entry to include.")
    parser.add_argument("--count", type=int, default=200, help="Number of mp4 entries to include.")
    parser.add_argument("--selected-list", required=True, help="Output list of selected archive paths.")
    parser.add_argument("--metadata", required=True, help="Output metadata CSV for preprocessing/feature extraction.")
    parser.add_argument("--train-labels", required=True, help="Output train_labels.csv for unsupervised sync training.")
    args = parser.parse_args()

    if args.start < 1:
        raise ValueError("--start must be 1-indexed and >= 1")
    if args.count < 1:
        raise ValueError("--count must be >= 1")

    entries = [
        line.strip()
        for line in Path(args.zip_list).read_text().splitlines()
        if line.strip().endswith(".mp4")
    ]
    begin = args.start - 1
    end = begin + args.count
    selected = entries[begin:end]
    if not selected:
        raise RuntimeError(
            f"No Vox2 mp4 entries selected from {args.zip_list} "
            f"(start={args.start}, count={args.count}, total={len(entries)})"
        )

    for path in (args.selected_list, args.metadata, args.train_labels):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    Path(args.selected_list).write_text("\n".join(selected) + "\n")

    for csv_path, columns in [
        (args.metadata, ["path", "label", "num_frames"]),
        (args.train_labels, ["path", "label"]),
    ]:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for rel_path in selected:
                row = {"path": rel_path, "label": 0}
                if "num_frames" in columns:
                    row["num_frames"] = 0
                writer.writerow(row)

    print(f"Selected {len(selected)} / {len(entries)} Vox2 mp4 entries")
    print(f"First: {selected[0]}")
    print(f"Last : {selected[-1]}")


if __name__ == "__main__":
    main()
