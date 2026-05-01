"""
Select ~50 success and ~50 failure cases of TriRoute for human evaluation.

Success cases: TriRoute correctly classifies clips that the two-way (A2)
and/or three-way no-push-pull (A5) baselines misclassify — stratified
across RV-RA / RV-FA / FV-RA / FV-FA so the survey covers all four
quadrants of FAVC (with FV-RA over-sampled because it is the cross-domain
headline task).

Failure cases: TriRoute is confidently wrong — also stratified across
the four categories.

Outputs (under ./human_eval_cases/):
  - success_cases.csv   (50 rows; +5 spares per stratum)
  - failure_cases.csv   (50 rows; +5 spares per stratum)
  - README.txt          (column key + stratum counts)

Each row carries:
  category, label, prob_fake_a2, prob_fake_a5, prob_fake_a6,
  triroute_correct, baselines_wrong, margin, wav_path, mp4_path
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_pred(path, tag):
    df = pd.read_csv(path)
    if "score_full" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path}: missing score_full / label column")
    df["prob_fake"] = df["score_full"].apply(sigmoid)
    df = df.rename(columns={"prob_fake": f"prob_fake_{tag}"})
    return df[["path", "label", f"prob_fake_{tag}"]]


def npz_to_raw(npz_rel, raw_root):
    base = npz_rel[:-4]
    wav = os.path.join(raw_root, base + ".wav")
    mp4 = os.path.join(raw_root, base + "_roi.mp4")
    return wav, mp4


def category_of(path):
    return path.split("/", 1)[0]


def stratified_pick(df, key_col, ascending, strata_quota, total_target,
                    fill_categories):
    """Pick top rows per stratum from df (after sorting by key_col).

    strata_quota: dict like {"FakeVideo-RealAudio": 25, ...}
    total_target: total primary rows to return (fills shortfall from
                  fill_categories in priority order).
    Returns (picked_df, spares_df).
    """
    df = df.sort_values(key_col, ascending=ascending)
    picked, spares = [], []
    used_paths = set()
    for cat, n in strata_quota.items():
        sub = df[df["category"] == cat]
        head = sub.head(n)
        picked.append(head)
        used_paths.update(head["path"])
        spares.append(sub.iloc[n:n + 5])

    primary = pd.concat(picked, ignore_index=True)
    shortfall = total_target - len(primary)
    if shortfall > 0:
        # Fill from designated categories in priority order, skipping rows
        # already picked or held as spares.
        spare_paths = set(pd.concat(spares)["path"])
        for cat in fill_categories:
            if shortfall <= 0:
                break
            extra_pool = df[(df["category"] == cat)
                            & (~df["path"].isin(used_paths))
                            & (~df["path"].isin(spare_paths))]
            extra = extra_pool.head(shortfall)
            picked.append(extra)
            used_paths.update(extra["path"])
            shortfall -= len(extra)
        primary = pd.concat(picked, ignore_index=True)

    return primary, pd.concat(spares, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2_pred", default="outputs_A2_50ep_seed44/"
                    "results_favc7030_full/predictions.csv")
    ap.add_argument("--a5_pred", default="outputs_A5_seed44/"
                    "results_favc_7030_full/predictions.csv")
    ap.add_argument("--a6_pred", default="outputs_A6_av1m_pseudodomain_v2"
                    "_seed44/results_favc_7030_full/predictions.csv")
    ap.add_argument("--raw_root",
                    default="/data/projects/punim2637/nnliang/AVH-Align/data/"
                            "favc_preprocessed")
    ap.add_argument("--out_dir", default="human_eval_cases")
    args = ap.parse_args()

    a2 = load_pred(args.a2_pred, "a2")
    a5 = load_pred(args.a5_pred, "a5").drop(columns=["label"])
    a6 = load_pred(args.a6_pred, "a6").drop(columns=["label"])
    df = a2.merge(a5, on="path").merge(a6, on="path")
    df["category"] = df["path"].apply(category_of)

    # Per-model correctness on the FAVC binary task (label==1 means fake).
    df["a2_correct"] = ((df["prob_fake_a2"] >= 0.5) == df["label"].astype(bool))
    df["a5_correct"] = ((df["prob_fake_a5"] >= 0.5) == df["label"].astype(bool))
    df["a6_correct"] = ((df["prob_fake_a6"] >= 0.5) == df["label"].astype(bool))
    df["baselines_wrong"] = ~(df["a2_correct"] & df["a5_correct"])

    # Confidence in TriRoute's prediction (closer to 0 or 1 = more confident).
    df["a6_correct_conf"] = df.apply(
        lambda r: r["prob_fake_a6"] if r["label"] == 1
        else 1 - r["prob_fake_a6"], axis=1)
    df["a6_wrong_conf"] = 1 - df["a6_correct_conf"]
    # Margin we use to rank successes: TriRoute correct, baselines wrong,
    # and the gap between TriRoute and worse baseline is large.
    df["success_margin"] = df["a6_correct_conf"] - df.apply(
        lambda r: (r["prob_fake_a2"] if r["label"] == 1 else 1 - r["prob_fake_a2"])
        if not r["a2_correct"] else
        (r["prob_fake_a5"] if r["label"] == 1 else 1 - r["prob_fake_a5"])
        if not r["a5_correct"] else r["a6_correct_conf"], axis=1)

    success_pool = df[df["a6_correct"] & df["baselines_wrong"]].copy()
    failure_pool = df[~df["a6_correct"]].copy()

    print(f"Joined predictions: {len(df)} clips")
    print(f"  TriRoute correct: {df['a6_correct'].sum()} "
          f"({df['a6_correct'].mean():.1%})")
    print(f"  Success pool (TriRoute right, ≥1 baseline wrong): "
          f"{len(success_pool)}")
    print(f"  Failure pool (TriRoute wrong): {len(failure_pool)}")

    # Stratified targets (50 each). FV-RA is the headline cross-domain task,
    # so it gets the largest slice; RV-RA is the smallest pool (151 total)
    # so we cap at what we can realistically draw.
    success_quota = {
        "FakeVideo-RealAudio": 25,
        "FakeVideo-FakeAudio": 10,
        "RealVideo-FakeAudio": 5,
        "RealVideo-RealAudio": 10,
    }
    failure_quota = success_quota.copy()

    fill_order = ["FakeVideo-RealAudio", "FakeVideo-FakeAudio",
                  "RealVideo-RealAudio", "RealVideo-FakeAudio"]
    success_pick, success_spare = stratified_pick(
        success_pool, "success_margin", ascending=False,
        strata_quota=success_quota, total_target=50,
        fill_categories=fill_order)
    failure_pick, failure_spare = stratified_pick(
        failure_pool, "a6_wrong_conf", ascending=False,
        strata_quota=failure_quota, total_target=50,
        fill_categories=fill_order)

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    def attach_raw(picked, spares, label):
        rows = pd.concat([picked.assign(stratum_role="primary"),
                          spares.assign(stratum_role="spare")],
                         ignore_index=True)
        wavs, mp4s, wav_ok, mp4_ok = [], [], [], []
        for p in rows["path"]:
            w, m = npz_to_raw(p, args.raw_root)
            wavs.append(w); mp4s.append(m)
            wav_ok.append(os.path.exists(w))
            mp4_ok.append(os.path.exists(m))
        rows["wav_path"] = wavs
        rows["mp4_path"] = mp4s
        rows["wav_exists"] = wav_ok
        rows["mp4_exists"] = mp4_ok
        cols = ["category", "label", "stratum_role",
                "prob_fake_a2", "prob_fake_a5", "prob_fake_a6",
                "a2_correct", "a5_correct", "a6_correct",
                "success_margin" if label == "success" else "a6_wrong_conf",
                "wav_exists", "mp4_exists", "path", "wav_path", "mp4_path"]
        out_csv = out / f"{label}_cases.csv"
        rows[cols].to_csv(out_csv, index=False)
        print(f"  wrote {out_csv} ({len(picked)} primary + "
              f"{len(spares)} spare)")
        return rows

    s_rows = attach_raw(success_pick, success_spare, "success")
    f_rows = attach_raw(failure_pick, failure_spare, "failure")

    # Quick stratum summary
    print("\nSuccess primary by category:")
    print(s_rows[s_rows["stratum_role"] == "primary"]["category"]
          .value_counts().to_string())
    print("\nFailure primary by category:")
    print(f_rows[f_rows["stratum_role"] == "primary"]["category"]
          .value_counts().to_string())

    readme = out / "README.txt"
    readme.write_text(
        "Human-evaluation case files for AVH-Align / TriRoute\n"
        "=====================================================\n\n"
        "Generated by select_human_eval_cases.py.\n"
        f"Source predictions:\n"
        f"  A2 (two-way baseline): {args.a2_pred}\n"
        f"  A5 (three-way, no push-pull): {args.a5_pred}\n"
        f"  A6 (TriRoute, 100ep pseudodomain v2 seed44): {args.a6_pred}\n\n"
        "success_cases.csv — clips where TriRoute is correct AND at least one\n"
        "  of the two baselines is wrong, ranked by `success_margin` (larger\n"
        "  = TriRoute more confidently corrects the baseline failure).\n\n"
        "failure_cases.csv — clips where TriRoute is wrong, ranked by\n"
        "  `a6_wrong_conf` (closer to 1 = TriRoute is confidently wrong).\n\n"
        "Each file carries 50 primary rows (the survey targets) + 20 spare\n"
        "rows (5 per stratum) drawn from the same ranking, in case some\n"
        "primary clips have missing raw .wav / _roi.mp4 files.\n\n"
        "Stratum quotas (per file):\n"
        "  FakeVideo-RealAudio  25  (headline cross-domain FV-RA)\n"
        "  FakeVideo-FakeAudio  10\n"
        "  RealVideo-FakeAudio   5\n"
        "  RealVideo-RealAudio  10\n\n"
        "Columns:\n"
        "  category           FAVC quadrant\n"
        "  label              ground truth (1=fake, 0=real)\n"
        "  stratum_role       'primary' or 'spare'\n"
        "  prob_fake_a2/a5/a6 sigmoid(score_full) per model\n"
        "  a2_correct/...     bool, prediction at 0.5 threshold\n"
        "  success_margin     TriRoute_correct_conf − failed_baseline_conf\n"
        "                     (success_cases.csv only)\n"
        "  a6_wrong_conf      TriRoute confidence in the wrong class\n"
        "                     (failure_cases.csv only)\n"
        "  wav_exists/mp4_exists  raw asset presence checked at mining time\n"
        "  path / wav_path / mp4_path  npz key + resolved raw paths\n"
    )
    print(f"  wrote {readme}")


if __name__ == "__main__":
    main()
