"""
Compose the qualitative failure figure from clips selected by
mine_qualitative_failures.py.

For each clip, extract 3 frames evenly spaced through the .mp4 plus an audio
waveform, and assemble a paneled figure suitable for the paper.

Outputs:
  - paper/figures/fig_qualitative_main.pdf  (one representative case)
  - paper/figures/fig_qualitative_appendix.pdf  (the rest)
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
        # tile frames horizontally inside ax_frames
        h, w = frames[0].shape[:2]
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

    p_a2 = row["prob_fake_a2"]
    p_a6 = row["prob_fake_a6"]
    ax_score.barh([0, 1], [p_a2, p_a6],
                  color=["#bd5734", "#2a9d8f"])
    ax_score.set_yticks([0, 1])
    ax_score.set_yticklabels(["Two-way", "TriRoute"], fontsize=8)
    ax_score.axvline(0.5, color="grey", linestyle="--", linewidth=0.7)
    ax_score.set_xlim(0, 1)
    ax_score.set_xlabel("$P(\\mathrm{fake})$", fontsize=8)
    ax_score.tick_params(axis="both", labelsize=7)
    for i, p in enumerate([p_a2, p_a6]):
        ax_score.text(min(p + 0.02, 0.95), i, f"{p:.2f}",
                      va="center", fontsize=7)
    ax_score.set_title(f"Case {label_idx}: gap={row['gap']:.2f}",
                       fontsize=8)


def make_figure(rows, out_pdf, n_per_row=1, title=""):
    n = len(rows)
    fig = plt.figure(figsize=(10, 2.4 * n))
    gs = fig.add_gridspec(n, 3, width_ratios=[3, 2, 1.4],
                          hspace=0.6, wspace=0.25)
    for i, (_, row) in enumerate(rows.iterrows()):
        ax_f = fig.add_subplot(gs[i, 0])
        ax_w = fig.add_subplot(gs[i, 1])
        ax_s = fig.add_subplot(gs[i, 2])
        render_clip(ax_f, ax_w, ax_s, row, i + 1)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved {out_pdf}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="qualitative_failures.csv from mine_qualitative_failures.py")
    ap.add_argument("--n_main", type=int, default=1,
                    help="how many cases on main figure")
    ap.add_argument("--n_total", type=int, default=6,
                    help="total cases (main + appendix)")
    ap.add_argument("--out_dir", required=True,
                    help="where to save figs (paper/figures/)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df.head(args.n_total)
    if len(df) == 0:
        print("ERROR: no rows in csv", flush=True)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    main_rows = df.head(args.n_main)
    rest = df.iloc[args.n_main:args.n_total]

    make_figure(main_rows, os.path.join(args.out_dir, "fig_qualitative_main.pdf"),
                title="Two-way follows audio shortcut, TriRoute corrects via visual")
    if len(rest) > 0:
        make_figure(rest, os.path.join(args.out_dir, "fig_qualitative_appendix.pdf"),
                    title="Additional FV-RA cases where two-way fails / TriRoute corrects")


if __name__ == "__main__":
    main()
