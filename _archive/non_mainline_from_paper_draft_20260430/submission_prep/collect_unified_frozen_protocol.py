#!/usr/bin/env python3
"""
Parse standard FakeAVCeleb eval_results.txt files into a single CSV.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

BASE = Path("/data/projects/punim2637/nnliang/AVH-Align")
DEFAULT_MODELS = [
    ("A2", BASE / "avh_sup" / "outputs_A2" / "results_favc_matched" / "eval_results.txt"),
    ("A5", BASE / "avh_sup" / "outputs_A5" / "results" / "eval_results.txt"),
    ("A6", BASE / "avh_sup" / "outputs_A6" / "results" / "eval_results.txt"),
]

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
        m = LINE_RE.match(line.rstrip())
        if not m:
            continue
        metrics[m.group(1)] = {
            "auc": float(m.group("auc")),
            "ap": float(m.group("ap")),
            "n": int(m.group("n")),
        }
    return metrics or None


def main() -> None:
    out_dir = BASE / "submission_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, path in DEFAULT_MODELS:
        parsed = parse_eval_results(path)
        row = {
            "model": model,
            "eval_results_path": str(path),
            "protocol": "frozen_favc_npz_matched",
            "status": "missing" if parsed is None else "ok",
            "overall_auc": "",
            "overall_n": "",
            "rv_fa_auc": "",
            "fv_ra_auc": "",
            "fv_fa_auc": "",
        }
        if parsed is not None:
            row["overall_auc"] = parsed["overall"]["auc"]
            row["overall_n"] = parsed["overall"]["n"]
            row["rv_fa_auc"] = parsed["RealVideo-FakeAudio"]["auc"]
            row["fv_ra_auc"] = parsed["FakeVideo-RealAudio"]["auc"]
            row["fv_fa_auc"] = parsed["FakeVideo-FakeAudio"]["auc"]
        rows.append(row)

    out_path = out_dir / "unified_frozen_protocol_main.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
