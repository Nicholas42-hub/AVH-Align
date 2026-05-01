"""Progressive masking sweep figure for FV-RA AUC.

Reproduces the curves from Tables 7a/7b of the updated NeurIPS eval list:
visual masking should monotonically lower FV-RA AUC for routing-aware models,
while audio masking should be neutral or beneficial.

Run from this directory: python make_masking_sweep.py
Outputs: masking_sweep.pdf, masking_sweep.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

P = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])

# Visual masking (Table 7a): FV-RA AUC vs visual mask probability p
visual = {
    "Two-way baseline":        [0.4826, 0.4851, 0.4887, 0.4863, 0.4744, 0.4453],
    "Three-way, no push-pull": [0.6003, 0.5938, 0.5835, 0.5712, 0.5261, 0.4683],
    "TriRoute (ours)":         [0.6593, 0.6505, 0.6378, 0.6061, 0.5590, 0.4518],
}

# Audio masking (Table 7b): FV-RA AUC vs audio mask probability p
audio = {
    "Two-way baseline":        [0.4826, 0.5127, 0.5292, 0.5413, 0.5688, 0.5776],
    "Three-way, no push-pull": [0.6003, 0.5993, 0.6023, 0.6175, 0.6354, 0.6531],
    "TriRoute (ours)":         [0.6593, 0.6661, 0.6729, 0.6830, 0.6946, 0.7068],
}

COLORS = {
    "Two-way baseline":        "#888888",
    "Three-way, no push-pull": "#3B82F6",
    "TriRoute (ours)":         "#DC2626",
}
MARKERS = {"Two-way baseline": "o", "Three-way, no push-pull": "s", "TriRoute (ours)": "D"}

# Caren-amended: the two subplots share their lines, so a single centered
# legend at the bottom of the figure replaces the per-axis legend (which
# previously overlapped the curves on the left subplot).
fig, (ax_v, ax_a) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

for name, ys in visual.items():
    ax_v.plot(P, ys, color=COLORS[name], marker=MARKERS[name], lw=1.4,
              ms=4.5, label=name)
for name, ys in audio.items():
    ax_a.plot(P, ys, color=COLORS[name], marker=MARKERS[name], lw=1.4,
              ms=4.5, label=name)

for ax, title in [(ax_v, "Visual masking (FV-RA $\\downarrow$ if routing through vision)"),
                  (ax_a, "Audio masking (FV-RA $\\uparrow$ if audio was a distractor)")]:
    ax.set_xlabel("Mask probability $p$")
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xticks(P)

ax_v.set_ylabel("FV-RA AUC")

# Single shared legend, centered between the two subplots
handles, labels = ax_v.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=3, frameon=False, fontsize=8)

fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
fig.savefig("masking_sweep.pdf", bbox_inches="tight", dpi=300)
fig.savefig("masking_sweep.png", bbox_inches="tight", dpi=300)
print("Wrote masking_sweep.pdf / .png")
