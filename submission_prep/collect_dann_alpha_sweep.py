#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


BASE = Path("/data/projects/punim2637/nnliang/AVH-Align/avh_sup")
LINE_RE = re.compile(r"AUC=(?P<auc>[0-9.]+)\s+AP=(?P<ap>[0-9.]+)")


def alpha_from_name(name: str) -> float:
    match = re.search(r"alpha([0-9p]+)_seed", name)
    if not match:
        raise ValueError(f"Cannot parse alpha from {name}")
    return float(match.group(1).replace("p", "."))


def seed_from_name(name: str) -> int:
    match = re.search(r"seed([0-9]+)$", name)
    if not match:
        raise ValueError(f"Cannot parse seed from {name}")
    return int(match.group(1))


def parse_eval(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("overall"):
            m = LINE_RE.search(stripped)
            metrics["overall_auc"] = float(m.group("auc"))
            metrics["overall_ap"] = float(m.group("ap"))
        elif stripped.startswith("FakeVideo-RealAudio"):
            m = LINE_RE.search(stripped)
            metrics["fvra_auc"] = float(m.group("auc"))
            metrics["fvra_ap"] = float(m.group("ap"))
        elif stripped.startswith("RealVideo-FakeAudio"):
            m = LINE_RE.search(stripped)
            metrics["rvfa_auc"] = float(m.group("auc"))
            metrics["rvfa_ap"] = float(m.group("ap"))
        elif stripped.startswith("FakeVideo-FakeAudio"):
            m = LINE_RE.search(stripped)
            metrics["fvfa_auc"] = float(m.group("auc"))
            metrics["fvfa_ap"] = float(m.group("ap"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DANN alpha sweep results")
    parser.add_argument(
        "--glob",
        default="outputs_DANN_alpha*_seed*/results_favc_7030_full/eval_results.txt",
    )
    parser.add_argument(
        "--output_csv",
        default="/data/projects/punim2637/nnliang/AVH-Align/submission_prep/dann_alpha_sweep_summary.csv",
    )
    args = parser.parse_args()

    rows = []
    for eval_path in sorted(BASE.glob(args.glob)):
        exp_name = eval_path.parents[1].name.replace("outputs_", "")
        row = {
            "experiment": exp_name,
            "alpha": alpha_from_name(exp_name),
            "seed": seed_from_name(exp_name),
        }
        row.update(parse_eval(eval_path))
        rows.append(row)

    rows.sort(key=lambda r: (r["alpha"], r["seed"]))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment",
                "alpha",
                "seed",
                "overall_auc",
                "overall_ap",
                "rvfa_auc",
                "rvfa_ap",
                "fvra_auc",
                "fvra_ap",
                "fvfa_auc",
                "fvfa_ap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"alpha={row['alpha']:.1f} seed={row['seed']} "
            f"FV-RA AUC={row.get('fvra_auc', float('nan')):.4f} "
            f"overall AUC={row.get('overall_auc', float('nan')):.4f}"
        )
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()