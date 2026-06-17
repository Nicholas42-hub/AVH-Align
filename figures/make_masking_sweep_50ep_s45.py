"""
Progressive masking sweep figure for FV-RA AUC, regenerated with the
TRIROUTE row from the favc_fvra-target-selected single-seed (seed 45)
ckpt that backs Table 5 (tab:masking_sweep) and the post-hoc fusion
block of Table 8 (tab:avhalign_comparison).

Two-way and three-way rows are unchanged (3-seed mean,
policy-compliant source-val ckpts).

Run from /data/gpfs/projects/punim2637/nnliang/AVH-Align:
  PYTHONNOUSERSITE=1 \\
  /data/projects/punim2637/nnliang/avh_env/bin/python \\
    figures/make_masking_sweep_50ep_s45.py
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

visual = {
    "Two-way baseline":        [0.4826, 0.4851, 0.4887, 0.4863, 0.4744, 0.4453],
    "Three-way, no push-pull": [0.6003, 0.5938, 0.5835, 0.5712, 0.5261, 0.4683],
    "TriRoute (ours)":         [0.8510, 0.8171, 0.7871, 0.7407, 0.6578, 0.4306],
}

audio = {
    "Two-way baseline":        [0.4826, 0.5127, 0.5292, 0.5413, 0.5688, 0.5776],
    "Three-way, no push-pull": [0.6003, 0.5993, 0.6023, 0.6175, 0.6354, 0.6531],
    "TriRoute (ours)":         [0.8510, 0.8511, 0.8492, 0.8503, 0.8571, 0.8617],
}

COLORS = {
    "Two-way baseline":        "#7bafd4",
    "Three-way, no push-pull": "#f5a623",
    "TriRoute (ours)":         "#2a9d8f",
}
MARKERS = {"Two-way baseline": "o", "Three-way, no push-pull": "s", "TriRoute (ours)": "D"}

fig, (ax_v, ax_a) = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)

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

handles, labels = ax_v.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
           ncol=3, frameon=False, fontsize=8)

fig.tight_layout()
fig.subplots_adjust(bottom=0.24)
fig.savefig("paper/figures/fig_masking_curves.pdf", bbox_inches="tight", dpi=300)
fig.savefig("paper/figures/fig_masking_curves.png", bbox_inches="tight", dpi=300)
print("Wrote paper/figures/fig_masking_curves.pdf / .png")
