"""
Mine FV-RA test clips where A2 (two-way baseline) misclassifies as real
(prob_fake < 0.5) but A6 (TriRoute) correctly identifies as fake (prob_fake > 0.5).

Inputs (predictions.csv schema): path, score_full, score_causal, score_spurious, label
Outputs:
  - qualitative_failures.csv: top-K clips with both model scores and raw paths
  - prints summary table

Run: python mine_qualitative_failures.py
"""

import argparse
import os
import sys
import math
from pathlib import Path

import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_pred(p, tag):
    df = pd.read_csv(p)
    if "score_full" not in df.columns:
        raise ValueError(f"{p}: missing score_full column")
    df = df[df["label"] == 1]  # only fake clips (FV-RA in our filter)
    df["prob_fake"] = df["score_full"].apply(sigmoid)
    df = df.rename(columns={
        "score_full": f"score_full_{tag}",
        "score_causal": f"score_causal_{tag}",
        "score_spurious": f"score_spurious_{tag}",
        "prob_fake": f"prob_fake_{tag}",
    })
    return df[["path", "label", f"score_full_{tag}", f"prob_fake_{tag}"]]


def npz_to_raw(npz_rel, raw_root):
    """Convert e.g. 'FakeVideo-RealAudio/African/men/id00076/00109_1.npz'
    → (.wav path, _roi.mp4 path) under raw_root.
    """
    base = npz_rel[:-4]  # strip .npz
    wav = os.path.join(raw_root, base + ".wav")
    mp4 = os.path.join(raw_root, base + "_roi.mp4")
    return wav, mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2_pred", required=True,
                    help="A2 (two-way baseline) predictions.csv")
    ap.add_argument("--a6_pred", required=True,
                    help="A6 (TriRoute) predictions.csv")
    ap.add_argument("--raw_root", required=True,
                    help="root of favc_preprocessed/")
    ap.add_argument("--top_k", type=int, default=8,
                    help="how many to dump (extra in case some lack raw files)")
    ap.add_argument("--out", default="qualitative_failures.csv")
    args = ap.parse_args()

    a2 = load_pred(args.a2_pred, "a2")
    a6 = load_pred(args.a6_pred, "a6")
    print(f"A2 fake rows: {len(a2)}; A6 fake rows: {len(a6)}", flush=True)

    # filter A2 to FV-RA only and join with A6
    df = a2.merge(a6.drop(columns=["label"]), on="path", how="inner")
    df = df[df["path"].str.startswith("FakeVideo-RealAudio/")]
    print(f"FV-RA fake clips with predictions from both: {len(df)}", flush=True)

    # filter: A2 misclassifies (prob<0.5) AND A6 corrects (prob>0.5)
    cand = df[(df["prob_fake_a2"] < 0.5) & (df["prob_fake_a6"] > 0.5)].copy()
    cand["gap"] = cand["prob_fake_a6"] - cand["prob_fake_a2"]
    cand = cand.sort_values("gap", ascending=False)
    print(f"Candidates (A2 fails, A6 corrects): {len(cand)}", flush=True)

    # add raw paths and verify file existence
    cand["wav_path"] = ""
    cand["mp4_path"] = ""
    cand["wav_exists"] = False
    cand["mp4_exists"] = False
    for i, row in cand.iterrows():
        wav, mp4 = npz_to_raw(row["path"], args.raw_root)
        cand.at[i, "wav_path"] = wav
        cand.at[i, "mp4_path"] = mp4
        cand.at[i, "wav_exists"] = os.path.exists(wav)
        cand.at[i, "mp4_exists"] = os.path.exists(mp4)

    found = cand[cand["wav_exists"] & cand["mp4_exists"]].head(args.top_k)
    print(f"Top {args.top_k} with both raw files present:", flush=True)
    for _, r in found.iterrows():
        print(f"  gap={r['gap']:.3f}  A2={r['prob_fake_a2']:.3f}  "
              f"A6={r['prob_fake_a6']:.3f}  {r['path']}", flush=True)

    found.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(found)} rows)", flush=True)


if __name__ == "__main__":
    main()
