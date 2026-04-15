#!/usr/bin/env python3
"""Materialize FakeAVCeleb 70/30 split CSVs with local canonical directory names.

The public split CSVs in avh_sup/csv_metadata/favc use compact race directory
names such as Caucasian_American. The local FakeAVCeleb_v1.2 tree uses the
canonical dataset names such as Caucasian (American). This script preserves the
existing train/val/test membership while rewriting only those directory tokens.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RACE_DIR_MAP = {
    "Caucasian_American": "Caucasian (American)",
    "Caucasian_European": "Caucasian (European)",
    "Asian_East": "Asian (East)",
    "Asian_South": "Asian (South)",
}


def canonicalize_full_path(full_path: str) -> str:
    parts = full_path.split("/")
    if len(parts) > 2 and parts[2] in RACE_DIR_MAP:
        parts[2] = RACE_DIR_MAP[parts[2]]
    return "/".join(parts)


def rewrite_split(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="") as src, output_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            row["full_path"] = canonicalize_full_path(row["full_path"])
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv_root",
        default="avh_sup/csv_metadata/favc",
        help="Directory containing the original train/val/test split CSVs.",
    )
    parser.add_argument(
        "--output_csv_root",
        default="avh_sup/csv_metadata/favc_7030_canonical",
        help="Directory to write canonicalized split CSVs.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_csv_root)
    output_root = Path(args.output_csv_root)
    for split in ["train", "val", "test"]:
        n_rows = rewrite_split(input_root / f"{split}_split.csv", output_root / f"{split}_split.csv")
        print(f"{split}: wrote {n_rows} rows -> {output_root / f'{split}_split.csv'}", flush=True)


if __name__ == "__main__":
    main()
