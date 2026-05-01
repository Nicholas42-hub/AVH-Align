#!/usr/bin/env python3
"""Prepare AVAD input lists and evaluate AVAD scores under our FAVC protocol.

The AVAD official detector reads a plain text file of absolute mp4 paths and
writes ``testing_scores.npy`` in the same order. This helper bridges that format
to our FakeAVCeleb 70/30 category protocol.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


CATEGORY_NAME = {
    "A": "RealVideo-RealAudio",
    "B": "RealVideo-FakeAudio",
    "C": "FakeVideo-RealAudio",
    "D": "FakeVideo-FakeAudio",
}

SHORT_NAME = {
    "B": "RV-FA",
    "C": "FV-RA",
    "D": "FV-FA",
}

FAKE_CATS = {"B", "C", "D"}

ETHNICITY_REMAP = {
    "Caucasian_American": "Caucasian (American)",
    "Caucasian_European": "Caucasian (European)",
    "Asian_South": "Asian (South)",
    "Asian_East": "Asian (East)",
}


def resolve_favc_path(favc_root: Path, full_path: str) -> Optional[Path]:
    rel = full_path.replace("FakeAVCeleb/", "", 1)
    parts = Path(rel).parts
    candidates = [favc_root.joinpath(*parts)]

    if len(parts) > 1 and parts[1] in ETHNICITY_REMAP:
        remapped = list(parts)
        remapped[1] = ETHNICITY_REMAP[parts[1]]
        candidates.append(favc_root.joinpath(*remapped))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score):
    from sklearn.metrics import average_precision_score

    if len(set(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def write_eval_inputs(args):
    favc_root = Path(args.favc_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    with open(args.favc_csv, newline="") as f:
        for row in csv.DictReader(f):
            path = resolve_favc_path(favc_root, row["full_path"])
            if path is None:
                missing.append(row["full_path"])
                continue
            rows.append(
                {
                    "path": str(path),
                    "source": row.get("source", ""),
                    "category": row["category"],
                    "category_name": CATEGORY_NAME[row["category"]],
                    "label": int(row["category"] in FAKE_CATS),
                }
            )

    list_path = out_dir / "favc_7030_test_videos.txt"
    labels_path = out_dir / "favc_7030_test_labels.csv"
    missing_path = out_dir / "missing_favc_paths.txt"

    with open(list_path, "w") as f:
        for row in rows:
            f.write(row["path"] + "\n")

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "source", "category", "category_name", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(missing_path, "w") as f:
        for path in missing:
            f.write(path + "\n")

    summary = {
        "n_total_csv": len(rows) + len(missing),
        "n_resolved": len(rows),
        "n_missing": len(missing),
        "list_path": str(list_path),
        "labels_path": str(labels_path),
        "missing_path": str(missing_path),
    }
    with open(out_dir / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def evaluate_scores(args):
    import numpy as np

    labels = []
    with open(args.labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(row)

    scores = np.load(args.scores_npy).reshape(-1).astype(float)
    if len(scores) != len(labels):
        raise ValueError(
            f"Score count ({len(scores)}) does not match label count ({len(labels)})."
        )

    y_true = [int(row["label"]) for row in labels]
    # AVAD stores anomaly losses; higher score means more fake-like.
    y_score = scores.tolist()

    results = {
        "overall": {
            "n": len(labels),
            "auc": safe_auc(y_true, y_score),
            "ap": safe_ap(y_true, y_score),
        }
    }

    real_idx = [i for i, row in enumerate(labels) if row["category"] == "A"]
    for code, name in SHORT_NAME.items():
        fake_idx = [i for i, row in enumerate(labels) if row["category"] == code]
        idx = real_idx + fake_idx
        results[name] = {
            "n_real": len(real_idx),
            "n_fake": len(fake_idx),
            "auc": safe_auc([y_true[i] for i in idx], [y_score[i] for i in idx]),
            "ap": safe_ap([y_true[i] for i in idx], [y_score[i] for i in idx]),
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for key, value in results.items():
        auc = value["auc"]
        ap = value["ap"]
        auc_s = "N/A" if auc is None else f"{auc:.4f}"
        ap_s = "N/A" if ap is None else f"{ap:.4f}"
        print(f"{key:8s} AUC={auc_s} AP={ap_s}")
    print(f"Saved: {out_path}")


def write_shards(args):
    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    n_shards = int(args.n_shards)
    for shard_id in range(n_shards):
        shard_rows = [row for i, row in enumerate(rows) if i % n_shards == shard_id]
        list_path = shard_dir / f"shard_{shard_id:02d}_videos.txt"
        labels_path = shard_dir / f"shard_{shard_id:02d}_labels.csv"

        with open(list_path, "w") as f:
            for row in shard_rows:
                f.write(row["path"] + "\n")

        with open(labels_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["path", "source", "category", "category_name", "label"]
            )
            writer.writeheader()
            writer.writerows(shard_rows)

    summary = {
        "n_total": len(rows),
        "n_shards": n_shards,
        "shard_dir": str(shard_dir),
    }
    with open(shard_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def merge_shards(args):
    import numpy as np

    out_dir = Path(args.output_dir)
    shard_dir = out_dir / "shards"
    score_dir = Path(args.score_dir)
    all_scores = []
    all_rows = []

    for shard_id in range(int(args.n_shards)):
        labels_path = shard_dir / f"shard_{shard_id:02d}_labels.csv"
        scores_path = score_dir / f"shard_{shard_id:02d}" / "testing_scores.npy"
        if not labels_path.exists():
            raise FileNotFoundError(labels_path)
        if not scores_path.exists():
            raise FileNotFoundError(scores_path)

        with open(labels_path, newline="") as f:
            rows = list(csv.DictReader(f))
        scores = np.load(scores_path).reshape(-1).astype(float)
        if len(scores) != len(rows):
            raise ValueError(
                f"{scores_path}: score count ({len(scores)}) does not match "
                f"label count ({len(rows)})."
            )
        for row, score in zip(rows, scores):
            row = dict(row)
            row["_score"] = score
            all_rows.append(row)

    all_rows.sort(key=lambda row: row["path"])
    all_scores = np.array([row["_score"] for row in all_rows], dtype=float)

    merged_labels = out_dir / "favc_7030_test_labels_merged_from_shards.csv"
    with open(merged_labels, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "source", "category", "category_name", "label"]
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "source": row["source"],
                    "category": row["category"],
                    "category_name": row["category_name"],
                    "label": row["label"],
                }
            )

    merged_scores = out_dir / "favc_7030_testing_scores_merged.npy"
    np.save(merged_scores, all_scores)
    print(json.dumps({"labels_csv": str(merged_labels), "scores_npy": str(merged_scores)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--favc_csv",
        default="avh_sup/csv_metadata/favc_7030_canonical/test_split.csv",
    )
    prep.add_argument(
        "--favc_root",
        default="/data/projects/punim2637/nnliang/Datasets/FakeCeleb/FakeAVCeleb_v1.2",
    )
    prep.add_argument(
        "--output_dir",
        default="submission_prep/avad_matched_baseline",
    )

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument(
        "--labels_csv",
        default="submission_prep/avad_matched_baseline/favc_7030_test_labels.csv",
    )
    eval_p.add_argument(
        "--scores_npy",
        default="external/audio-visual-forensics/outputs_favc_7030/testing_scores.npy",
    )
    eval_p.add_argument(
        "--output_json",
        default="submission_prep/avad_matched_baseline/favc_7030_results.json",
    )

    shard_p = sub.add_parser("shard")
    shard_p.add_argument(
        "--labels_csv",
        default="submission_prep/avad_matched_baseline/favc_7030_test_labels.csv",
    )
    shard_p.add_argument(
        "--output_dir",
        default="submission_prep/avad_matched_baseline",
    )
    shard_p.add_argument("--n_shards", type=int, default=16)

    merge_p = sub.add_parser("merge_shards")
    merge_p.add_argument(
        "--output_dir",
        default="submission_prep/avad_matched_baseline",
    )
    merge_p.add_argument(
        "--score_dir",
        default="external/audio-visual-forensics/outputs_favc_7030_shards",
    )
    merge_p.add_argument("--n_shards", type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "prepare":
        write_eval_inputs(args)
    elif args.cmd == "evaluate":
        evaluate_scores(args)
    elif args.cmd == "shard":
        write_shards(args)
    elif args.cmd == "merge_shards":
        merge_shards(args)


if __name__ == "__main__":
    main()
