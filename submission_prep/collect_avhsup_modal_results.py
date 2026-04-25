#!/usr/bin/env python3
"""Collect plain AVH_Sup modal baseline results across seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


LINE_RE = re.compile(
    r"^\s*(?P<category>overall|RealVideo-FakeAudio|FakeVideo-RealAudio|FakeVideo-FakeAudio)"
    r"\s+AUC=(?P<auc>[0-9.]+)\s+AP=(?P<ap>[0-9.]+)\s+N=(?P<n>[0-9]+)"
)


def parse_eval(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        row = m.groupdict()
        row["auc"] = float(row["auc"])
        row["ap"] = float(row["ap"])
        row["n"] = int(row["n"])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup",
        help="avh_sup directory containing outputs_avhsup_av1m_*_seed* dirs",
    )
    parser.add_argument(
        "--results_dir",
        default="results_favc_7030",
        help="Result directory name inside each output dir.",
    )
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for eval_path in sorted(root.glob(f"outputs_avhsup_av1m_*_seed*/{args.results_dir}/eval_results.txt")):
        name = eval_path.parts[-3]
        m = re.match(r"outputs_avhsup_av1m_(?P<input_type>both|audio|video)_seed(?P<seed>[0-9]+)", name)
        if not m:
            continue
        for row in parse_eval(eval_path):
            row.update(m.groupdict())
            records.append(row)

    if not records:
        raise SystemExit(f"No completed eval_results.txt files found under {root}")

    df = pd.DataFrame(records)
    df["seed"] = df["seed"].astype(int)
    summary = (
        df.groupby(["input_type", "category"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            seeds=("seed", lambda s: ",".join(map(str, sorted(s.unique())))),
            n=("n", "first"),
        )
        .reset_index()
        .sort_values(["input_type", "category"])
    )

    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
