#!/usr/bin/env python3
"""
Collect A6 variant seed-sweep FakeAVCeleb results and summarize mean±std.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

BASE = Path("/data/projects/punim2637/nnliang/AVH-Align")
VARIANTS = ["A6_lite_trainreal", "A6_sharedonly_trainreal"]
SEEDS = [42, 43, 44]
RESULT_SUBDIR = "results_favc_matched"

LINE_RE = re.compile(
    r"^\s*(overall|RealVideo-FakeAudio|FakeVideo-RealAudio|FakeVideo-FakeAudio)"
    r"\s+AUC=(?P<auc>[0-9.]+)\s+AP=(?P<ap>[0-9.]+)\s+N=(?P<n>[0-9]+)"
)


def parse_eval_results(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    metrics: dict[str, dict[str, float]] = {}
    current_head = None
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if "── FULL HEAD ──" in line:
            current_head = "full"
            continue
        if "── CAUSAL HEAD ──" in line:
            current_head = "causal"
            continue
        if "── SPURIOUS HEAD ──" in line:
            current_head = "spurious"
            continue
        if current_head != "full":
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        metrics[key] = {
            "auc": float(m.group("auc")),
            "ap": float(m.group("ap")),
            "n": float(m.group("n")),
        }
    return metrics or None


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def main() -> None:
    out_dir = BASE / "submission_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    summary_rows = []
    metric_keys = [
        ("overall", "overall_auc"),
        ("RealVideo-FakeAudio", "rv_fa_auc"),
        ("FakeVideo-RealAudio", "fv_ra_auc"),
        ("FakeVideo-FakeAudio", "fv_fa_auc"),
    ]

    for variant in VARIANTS:
        per_metric: dict[str, list[float]] = {label: [] for _, label in metric_keys}
        for seed in SEEDS:
            eval_path = BASE / "avh_sup" / f"outputs_{variant}_seed{seed}" / RESULT_SUBDIR / "eval_results.txt"
            parsed = parse_eval_results(eval_path)
            row = {
                "variant": variant,
                "seed": seed,
                "eval_results_path": str(eval_path),
                "status": "missing" if parsed is None else "ok",
            }
            if parsed is not None:
                for raw_key, label in metric_keys:
                    value = parsed.get(raw_key, {}).get("auc")
                    row[label] = value
                    if value is not None:
                        per_metric[label].append(value)
            raw_rows.append(row)

        summary = {"variant": variant}
        for _, label in metric_keys:
            values = per_metric[label]
            if values:
                mean, std = mean_std(values)
                summary[f"{label}_mean"] = mean
                summary[f"{label}_std"] = std
                summary[f"{label}_n"] = len(values)
            else:
                summary[f"{label}_mean"] = ""
                summary[f"{label}_std"] = ""
                summary[f"{label}_n"] = 0
        summary_rows.append(summary)

    raw_path = out_dir / "a6_variant_seed_results_raw.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = out_dir / "a6_variant_seed_results_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved raw -> {raw_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
