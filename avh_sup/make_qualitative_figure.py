"""
Compose the qualitative figure (compact, table-style) from clips selected by
either:
  (a) success_cases.csv / failure_cases.csv  (from select_human_eval_cases.py)
  (b) legacy qualitative_failures.csv         (from mine_qualitative_failures.py)

Each row shows: video frame strip | audio waveform | 3-model probability bars
| sub-caption with the Expected Output (ground-truth label + FAVC quadrant).

Outputs:
  fig_qualitative_main.pdf   (top-1 case for the main paper)
  fig_qualitative_appendix.pdf  (the rest, for the appendix)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2
import soundfile as sf


CATEGORY_HUMAN = {
    "FakeVideo-RealAudio": "Fake (FV-RA, cross-domain)",
    "FakeVideo-FakeAudio": "Fake (FV-FA)",
    "RealVideo-FakeAudio": "Fake (RV-FA)",
    "RealVideo-RealAudio": "Real (RV-RA)",
}


def expected_output(row):
    """Plain-language ground truth for the sub-caption."""
    if "category" in row and isinstance(row["category"], str):
        cat = row["category"]
    else:
        cat = row["path"].split("/", 1)[0]
    label_str = CATEGORY_HUMAN.get(cat, "Fake" if row["label"] == 1 else "Real")
    return label_str, cat


def extract_frames(mp4_path, n=3):
    frames = []
    try:
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            return []
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if T <= 0:
            cap.release()
            return []
        idxs = np.linspace(0, T - 1, n, dtype=int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    except Exception as e:
        print(f"WARN frame extraction failed for {mp4_path}: {e}", flush=True)
    return frames


def load_audio_waveform(wav_path):
    try:
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception as e:
        print(f"WARN audio load failed for {wav_path}: {e}", flush=True)
        return None, None


def render_clip(ax_frames, ax_wave, ax_score, row, label_idx):
    """Render frames, waveform, and score panel for one clip."""
    frames = extract_frames(row["mp4_path"], n=3)
    if len(frames) == 0:
        ax_frames.text(0.5, 0.5, "[no video]", ha="center", va="center")
        ax_frames.axis("off")
    else:
        tile = np.concatenate(frames, axis=1)
        ax_frames.imshow(tile)
        ax_frames.set_xticks([])
        ax_frames.set_yticks([])

    data, sr = load_audio_waveform(row["wav_path"])
    if data is not None:
        t = np.arange(len(data)) / sr
        ax_wave.plot(t, data, linewidth=0.5, color="#2a6f97")
        ax_wave.set_ylim(-1.05, 1.05)
        ax_wave.set_xlim(0, t[-1] if len(t) > 0 else 1.0)
        ax_wave.set_xlabel("time (s)", fontsize=8)
        ax_wave.tick_params(axis="both", labelsize=7)
    else:
        ax_wave.text(0.5, 0.5, "[no audio]", ha="center", va="center",
                     transform=ax_wave.transAxes)
        ax_wave.axis("off")

    # Show whichever models we have probabilities for; assume A6 always.
    model_specs = []
    if "prob_fake_a2" in row:
        model_specs.append(("Two-way", row["prob_fake_a2"], "#86b3d4"))
    if "prob_fake_a5" in row and pd.notna(row.get("prob_fake_a5")):
        model_specs.append(("Three-way", row["prob_fake_a5"], "#e9b562"))
    model_specs.append(("TriRoute (ours)", row["prob_fake_a6"], "#2a9d8f"))

    names = [m[0] for m in model_specs]
    probs = [m[1] for m in model_specs]
    colors = [m[2] for m in model_specs]
    y_pos = np.arange(len(model_specs))

    ax_score.barh(y_pos, probs, color=colors)
    ax_score.set_yticks(y_pos)
    ax_score.set_yticklabels(names, fontsize=8)
    for tick in ax_score.get_yticklabels():
        if "ours" in tick.get_text():
            tick.set_fontweight("bold")
    ax_score.invert_yaxis()  # TriRoute on top — it's the headline model.
    ax_score.axvline(0.5, color="grey", linestyle="--", linewidth=0.7)
    ax_score.set_xlim(0, 1)
    ax_score.set_xlabel("$P(\\mathrm{fake})$", fontsize=8)
    ax_score.tick_params(axis="both", labelsize=7)
    for i, p in enumerate(probs):
        ax_score.text(min(p + 0.02, 0.92), i, f"{p:.2f}",
                      va="center", fontsize=7)

    expected, _cat = expected_output(row)
    # Expected-Output sub-caption sits below the score panel — addresses the
    # reviewer ask that each example carries its ground-truth verdict.
    ax_score.set_title(
        f"Case {label_idx} — Expected output: {expected}",
        fontsize=8.5, loc="left", color="#222222")


def make_figure(rows, out_pdf, title="", row_height=1.9):
    n = len(rows)
    if n == 0:
        return
    fig = plt.figure(figsize=(11, row_height * n + 0.6))
    gs = fig.add_gridspec(n, 3, width_ratios=[3.2, 2.0, 4.0],
                          hspace=0.65, wspace=0.50)
    for i, (_, row) in enumerate(rows.iterrows()):
        ax_f = fig.add_subplot(gs[i, 0])
        ax_w = fig.add_subplot(gs[i, 1])
        ax_s = fig.add_subplot(gs[i, 2])
        render_clip(ax_f, ax_w, ax_s, row, i + 1)
    if False and title:
        fig.suptitle(title, fontsize=11, y=0.995)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {out_pdf}", flush=True)


def write_latex_table(rows, out_tex, caption, label):
    """Booktabs LaTeX table mirroring the figure — supervisor asked for a
    compact tabular form alongside the visual panels."""
    has_a5 = "prob_fake_a5" in rows.columns and rows["prob_fake_a5"].notna().any()
    cols = "lll" + ("rrr" if has_a5 else "rr") + "l"
    header = ["#", "Quadrant", "Expected"]
    header += ["Two-way", "Three-way", "TriRoute"] if has_a5 \
        else ["Two-way", "TriRoute"]
    header.append("Verdict (TriRoute)")

    body_lines = []
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        expected, cat = expected_output(r)
        gt_fake = r["label"] == 1
        a6_fake_pred = r["prob_fake_a6"] >= 0.5
        verdict = (r"\textcolor{ForestGreen}{correct}" if a6_fake_pred == gt_fake
                   else r"\textcolor{BrickRed}{wrong}")
        cells = [str(i), cat.replace("Video-", "V/").replace("Audio", "A"),
                 expected, f"{r['prob_fake_a2']:.3f}"]
        if has_a5:
            cells.append(f"{r['prob_fake_a5']:.3f}"
                         if pd.notna(r.get("prob_fake_a5")) else "—")
        cells.append(f"{r['prob_fake_a6']:.3f}")
        cells.append(verdict)
        body_lines.append(" & ".join(cells) + r" \\")

    tex = (
        "% Auto-generated by make_qualitative_figure.py — do not hand-edit.\n"
        "\\begin{table}[t]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\small\n"
        f"  \\begin{{tabular}}{{{cols}}}\n"
        "    \\toprule\n"
        "    " + " & ".join(header) + r" \\" + "\n"
        "    \\midrule\n"
        "    " + "\n    ".join(body_lines) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(out_tex, "w") as f:
        f.write(tex)
    print(f"Saved {out_tex}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="success_cases.csv / failure_cases.csv / "
                         "qualitative_failures.csv")
    ap.add_argument("--n_main", type=int, default=1,
                    help="how many cases on main figure")
    ap.add_argument("--n_total", type=int, default=6,
                    help="total cases (main + appendix)")
    ap.add_argument("--out_dir", required=True,
                    help="where to save figs (paper/figures/)")
    ap.add_argument("--title_main", default="TriRoute corrects baseline failure on FV-RA")
    ap.add_argument("--title_appendix", default="Additional FV-RA cases (TriRoute correct, baselines wrong)")
    ap.add_argument("--label_prefix", default="qualitative_success",
                    help="LaTeX label prefix for tables")
    ap.add_argument("--out_suffix", default=None,
                    help="suffix for output filenames, e.g. 'success_appendix' → "
                         "fig_qualitative_success_appendix.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "stratum_role" in df.columns:
        df = df[df["stratum_role"] == "primary"].copy()
    if "wav_exists" in df.columns and "mp4_exists" in df.columns:
        df = df[df["wav_exists"] & df["mp4_exists"]].copy()
    df = df.head(args.n_total)
    if len(df) == 0:
        print("ERROR: no rows after filtering", flush=True)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = args.out_suffix  # e.g. "success_appendix" or None
    def _fname(base):
        if suffix:
            return os.path.join(args.out_dir, f"fig_qualitative_{suffix}.pdf")
        return os.path.join(args.out_dir, base)

    main_rows = df.head(args.n_main)
    rest = df.iloc[args.n_main:args.n_total]

    if args.n_main > 0:
        make_figure(main_rows,
                    os.path.join(args.out_dir, "fig_qualitative_main.pdf"),
                    title=args.title_main)
    if len(rest) > 0:
        make_figure(rest,
                    _fname("fig_qualitative_appendix.pdf"),
                    title=args.title_appendix)

    write_latex_table(
        main_rows,
        os.path.join(args.out_dir, "tab_qualitative_main.tex"),
        caption=args.title_main + " (per-clip probabilities; bold = correct).",
        label=f"tab:{args.label_prefix}_main")
    write_latex_table(
        df,
        os.path.join(args.out_dir, "tab_qualitative_all.tex"),
        caption=args.title_main + " — all selected cases.",
        label=f"tab:{args.label_prefix}_all")


if __name__ == "__main__":
    main()
