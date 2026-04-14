"""
Build explicit FakeAVCeleb CSV splits that match the currently extracted NPZ features.

Why this exists:
  The repository currently mixes three notions of "test protocol":
    1. the original FakeAVCeleb metadata CSVs,
    2. the subset of clips for which NPZ features were actually extracted,
    3. the effective number of samples that match under the repo's flexible filename logic.

  This script materializes the matched split files explicitly, so future evaluations can point
  to a stable CSV root instead of relying on implicit feature coverage.

Default output:
  avh_sup/csv_metadata/favc_npz_matched/{train,val,test}_split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def build_feature_index(root_path: Path) -> dict[str, list[str]]:
    by_dir: dict[str, list[str]] = defaultdict(list)
    for abs_path in sorted(root_path.rglob("*.npz")):
        rel_path = abs_path.relative_to(root_path).as_posix()
        base_path = rel_path[:-4]  # strip .npz
        parts = base_path.rsplit("/", 1)
        if len(parts) != 2:
            continue
        dir_path, full_filename = parts
        by_dir[dir_path].append(full_filename)
    return by_dir


def row_matches_feature(row: dict[str, str], feature_index: dict[str, list[str]]) -> bool:
    base_path = row["full_path"].replace("FakeAVCeleb/", "").replace(".mp4", "")
    parts = base_path.rsplit("/", 1)
    if len(parts) != 2:
        return False

    dir_path, allowed_name = parts
    candidates = feature_index.get(dir_path, [])
    for full_filename in candidates:
        base_filename = full_filename.split("_")[0]
        if (
            allowed_name == full_filename
            or allowed_name == base_filename
            or full_filename.startswith(allowed_name + "_")
        ):
            return True
    return False


def filter_split(csv_path: Path, feature_index: dict[str, list[str]]) -> tuple[list[dict[str, str]], dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    matched = [row for row in rows if row_matches_feature(row, feature_index)]
    cat_counts = Counter(row["category"] for row in matched)
    summary = {
        "source_csv": str(csv_path),
        "n_source_rows": len(rows),
        "n_matched_rows": len(matched),
        "category_counts": dict(cat_counts),
    }
    return matched, summary


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "category", "full_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_root",
        default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features",
    )
    parser.add_argument(
        "--source_csv_root",
        default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc",
    )
    parser.add_argument(
        "--output_csv_root",
        default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_npz_matched",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated list of splits to materialize",
    )
    args = parser.parse_args()

    features_root = Path(args.features_root)
    source_csv_root = Path(args.source_csv_root)
    output_csv_root = Path(args.output_csv_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    feature_index = build_feature_index(features_root)
    total_features = sum(len(v) for v in feature_index.values())
    summary = {
        "features_root": str(features_root),
        "source_csv_root": str(source_csv_root),
        "output_csv_root": str(output_csv_root),
        "n_feature_npz": total_features,
        "splits": {},
    }

    for split in splits:
        source_csv = source_csv_root / f"{split}_split.csv"
        matched_rows, split_summary = filter_split(source_csv, feature_index)
        write_csv(matched_rows, output_csv_root / f"{split}_split.csv")
        summary["splits"][split] = split_summary
        print(
            f"{split}: matched {split_summary['n_matched_rows']} / "
            f"{split_summary['n_source_rows']} rows",
            flush=True,
        )

    output_csv_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_csv_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
