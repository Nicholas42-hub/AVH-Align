#!/usr/bin/env python3
"""Collect A6 5-seed FakeAVCeleb 70/30 metrics for paper tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULT = "results_favc_7030_full/eval_results.txt"

RUNS = {
    "A6": [ROOT / "avh_sup" / f"outputs_A6_seed{s}" / RESULT for s in range(43, 48)],
    "A6_udelta_trainvalreal": [
        ROOT / "avh_sup" / f"outputs_A6_udelta_trainvalreal_seed{s}" / RESULT
        for s in range(43, 48)
    ],
}

PATTERNS = {
    "overall_auc": re.compile(r"Overall AUC:\s*([0-9.]+)"),
    "overall_ap": re.compile(r"Overall AP:\s*([0-9.]+)"),
    "rv_fa_auc": re.compile(r"RealVideo-FakeAudio:\s*AUC=([0-9.]+),\s*AP=([0-9.]+)"),
    "fv_ra_auc": re.compile(r"FakeVideo-RealAudio:\s*AUC=([0-9.]+),\s*AP=([0-9.]+)"),
    "fv_fa_auc": re.compile(r"FakeVideo-FakeAudio:\s*AUC=([0-9.]+),\s*AP=([0-9.]+)"),
}


def parse_result(path: Path) -> dict[str, float]:
    text = path.read_text()
    out: dict[str, float] = {}

    m = PATTERNS["overall_auc"].search(text)
    if not m:
        raise ValueError(f"Missing Overall AUC in {path}")
    out["overall_auc"] = float(m.group(1))

    m = PATTERNS["overall_ap"].search(text)
    if not m:
        raise ValueError(f"Missing Overall AP in {path}")
    out["overall_ap"] = float(m.group(1))

    for key, pattern in PATTERNS.items():
        if key in {"overall_auc", "overall_ap"}:
            continue
        m = pattern.search(text)
        if not m:
            raise ValueError(f"Missing {key} in {path}")
        prefix = key.removesuffix("_auc")
        out[f"{prefix}_auc"] = float(m.group(1))
        out[f"{prefix}_ap"] = float(m.group(2))
    return out


def summarize(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for variant, paths in RUNS.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{variant} is missing {len(missing)} result file(s):\n"
                + "\n".join(str(p) for p in missing)
            )

        per_seed = []
        for seed, path in zip(range(43, 48), paths):
            row = {"variant": variant, "seed": seed, **parse_result(path)}
            raw_rows.append(row)
            per_seed.append(row)

        metric_keys = [k for k in per_seed[0] if k not in {"variant", "seed"}]
        summary = {"variant": variant, "seeds": "43-47", "n": len(per_seed)}
        for key in metric_keys:
            avg, sd = summarize([float(r[key]) for r in per_seed])
            summary[f"{key}_mean"] = avg
            summary[f"{key}_std"] = sd
        summary_rows.append(summary)

    out_dir = ROOT / "submission_prep"
    raw_path = out_dir / "a6_5seed_favc7030_raw.csv"
    summary_path = out_dir / "a6_5seed_favc7030_summary.csv"
    txt_path = out_dir / "a6_5seed_favc7030_summary.txt"

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = []
    for row in summary_rows:
        lines.append(f"{row['variant']} (seeds {row['seeds']}, n={row['n']})")
        lines.append(
            "  Overall: "
            f"{row['overall_auc_mean']:.4f}/{row['overall_ap_mean']:.4f} "
            f"(std {row['overall_auc_std']:.4f}/{row['overall_ap_std']:.4f})"
        )
        lines.append(
            "  FV-RA:   "
            f"{row['fv_ra_auc_mean']:.4f}/{row['fv_ra_ap_mean']:.4f} "
            f"(std {row['fv_ra_auc_std']:.4f}/{row['fv_ra_ap_std']:.4f})"
        )
    txt_path.write_text("\n".join(lines) + "\n")

    print(txt_path.read_text())
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()

