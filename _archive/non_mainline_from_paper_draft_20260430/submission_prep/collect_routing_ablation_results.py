"""
Collect A6 routing-ablation eval_results.txt files into CSV summaries.

Usage:
  python submission_prep/collect_routing_ablation_results.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
SUP = ROOT / "avh_sup"
VARIANTS = ("noudelta", "expose_residual", "ddis_only", "dadv_only", "udelta100")
SEEDS = (43, 44, 45)
SPLITS = ("results_favc_matched", "results_favc_7030")

METRIC_RE = re.compile(r"^\s*(.+?)\s+AUC=([0-9.]+)\s+AP=([0-9.]+)\s+N=(\d+)")


def _parse_eval(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    section = None
    metrics = {}
    for line in path.read_text().splitlines():
        if "FULL HEAD" in line:
            section = "full"
            continue
        if "CAUSAL HEAD" in line:
            section = "causal"
            continue
        if "SPURIOUS HEAD" in line:
            section = "spurious"
            continue
        match = METRIC_RE.match(line)
        if not match or section is None:
            continue
        category, auc, ap, n = match.groups()
        metrics[(section, category.strip())] = {"auc": auc, "ap": ap, "n": n}
    return metrics


def main() -> None:
    rows = []
    for split in SPLITS:
        for variant in VARIANTS:
            for seed in SEEDS:
                result_file = (
                    SUP
                    / f"outputs_A6_trainvalreal_{variant}_seed{seed}"
                    / split
                    / "eval_results.txt"
                )
                if not result_file.exists():
                    rows.append({
                        "split": split,
                        "variant": variant,
                        "seed": seed,
                        "status": "missing",
                    })
                    continue
                metrics = _parse_eval(result_file)
                for (section, category), vals in metrics.items():
                    rows.append({
                        "split": split,
                        "variant": variant,
                        "seed": seed,
                        "status": "ok",
                        "head": section,
                        "category": category,
                        **vals,
                    })

    out_path = ROOT / "submission_prep" / "routing_ablation_results_raw.csv"
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "split",
            "variant",
            "seed",
            "status",
            "head",
            "category",
            "auc",
            "ap",
            "n",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
