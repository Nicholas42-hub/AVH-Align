"""
Regenerate fig_main_fvra and fig_appendix_table1_main with correct 3-seed means.

Canonical values (non-50ep patience-20 runs, seeds 43/44/45):
  Two-way: FV-RA=0.483, Overall=0.760
  DANN (50ep): FV-RA=0.506, Overall=0.770
  CORAL (50ep): FV-RA=0.506, Overall=0.759
  Three-way: FV-RA=0.600, Overall=0.814
  TriRoute: FV-RA=0.838, Overall=0.922
  TriRoute+UDA: FV-RA=0.821, Overall=0.915
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/data/projects/punim2637/nnliang/AVH-Align/paper/figures"

COLORS = {
    "two_way":   "#5778a4",
    "three_way": "#e49444",
    "triroute":  "#2ca084",
}

# ── Fig: Main FV-RA bar chart ──────────────────────────────────────────────────
ABL_FVRA = [0.8633, 0.7945, 0.8045]   # TriRoute + real-target UDA seeds

methods = [
    "AVH-Align/sup\n[Smeu et al.]",
    "Two-way\nbaseline",
    "Two-way\n+DANN/GRL",
    "Two-way\n+CORAL",
    "Three-way\nno push-pull",
    "TriRoute\n+ real-target\ndomain (abl.)",
    "TriRoute\n(ours)",
]
fvra_aucs = [0.519, 0.483, 0.496, 0.521, 0.600,
             float(np.mean(ABL_FVRA)), 0.838]
stds = [0.0, 0.041, 0.069, 0.054, 0.023,
        float(np.std(ABL_FVRA, ddof=1)), 0.015]

bar_colors = [
    "#adb5bd",
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["three_way"],
    "#88c9c0",
    COLORS["triroute"],
]

fig, ax = plt.subplots(figsize=(4.5, 3.5))
fig.patch.set_facecolor("white")
x = np.arange(len(methods))
bw = 0.40
bars = ax.bar(x, fvra_aucs, color=bar_colors, edgecolor="white", width=bw,
              yerr=stds, capsize=3,
              error_kw={"elinewidth": 1.0, "ecolor": "black"})
ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2, alpha=0.85, zorder=1)
ax.text(len(methods) - 0.45, 0.505, "chance (AUC = 0.5)",
        fontsize=11, color="#444444", ha="right", va="bottom", style="italic")

for bar, v, s in zip(bars, fvra_aucs, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (s if s > 0 else 0) + 0.014,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222222")

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=7.5, rotation=40, ha="right")
ax.tick_params(axis="y", labelsize=8)
ax.set_ylabel(r"FV-RA AUC (AV1M $\rightarrow$ FAVC)", fontsize=8)
ax.set_ylim(0.40, 0.97)
ax.set_xlim(-0.55, len(methods) - 0.45)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor("white")

legend_handles = [
    mpatches.Patch(color="#adb5bd",           label="External baseline"),
    mpatches.Patch(color=COLORS["two_way"],   label="Two-way models"),
    mpatches.Patch(color=COLORS["three_way"], label="Three-way, no push-pull"),
    mpatches.Patch(color="#88c9c0",           label="TriRoute, real-target domain (abl.)"),
    mpatches.Patch(color=COLORS["triroute"],  label="TriRoute (ours)"),
]
ax.legend(handles=legend_handles, fontsize=7.5,
          loc="upper left", bbox_to_anchor=(0.01, 0.99),
          ncol=1, frameon=False, handlelength=1.2, labelspacing=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_main_fvra.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_main_fvra.pdf", bbox_inches="tight", facecolor="white")
print("Saved: fig_main_fvra.png/.pdf")
plt.close()


# ── Fig: Appendix Table 1 (Overall + FV-RA panels) ────────────────────────────
T1_METHODS = [
    "AVH-Align/sup",
    "Two-way baseline",
    "Two-way + DANN/GRL",
    "Two-way + CORAL",
    "Three-way, no push-pull",
    "TriRoute (no target)",
    "TriRoute + UDA",
]
T1_COLORS = [
    "#adb5bd",
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["three_way"],
    COLORS["triroute"],
    "#88c9c0",
]

T1_OVERALL     = [0.777, 0.760, 0.770, 0.759, 0.814, 0.922, 0.915]
T1_FVRA        = [0.519, 0.483, 0.506, 0.506, 0.600, 0.838, 0.821]
T1_OVERALL_STD = [0,     0.019, 0.031, 0.030, 0.011, 0.007, 0.020]
T1_FVRA_STD    = [0,     0.041, 0.069, 0.054, 0.023, 0.015, 0.037]

T1_LEGEND_HANDLES = [
    mpatches.Patch(color="#adb5bd",            label="External baseline"),
    mpatches.Patch(color=COLORS["two_way"],    label="Two-way models"),
    mpatches.Patch(color=COLORS["three_way"],  label="Three-way, no push-pull"),
    mpatches.Patch(color=COLORS["triroute"],   label="TriRoute (no target)"),
    mpatches.Patch(color="#88c9c0",            label="TriRoute + UDA (real-target)"),
]
SUBTITLE = r"Error bars: sample std across available seeds; single-seed entries shown without error bars"


def _draw_t1_panel(ax, title, vals, errs, ylim, draw_chance):
    ax.set_facecolor("white")
    xi = np.arange(len(T1_METHODS))
    bw = 0.6
    bars = ax.bar(xi, vals, color=T1_COLORS, edgecolor="white", width=bw,
                  yerr=errs, capsize=4,
                  error_kw={"elinewidth": 1.1, "ecolor": "black"})
    ax.set_title(title, fontsize=13.5, pad=8)
    ax.set_ylim(ylim)
    ax.set_xticks(xi)
    ax.set_xticklabels(T1_METHODS, fontsize=11, rotation=25, ha="right")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylabel("AUC", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    label_pad = (ylim[1] - ylim[0]) * 0.025
    for bar, v, s in zip(bars, vals, errs):
        top = bar.get_height() + (s if s > 0 else 0) + label_pad
        ax.text(bar.get_x() + bar.get_width()/2, top,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#222222")
    if draw_chance:
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0,
                   alpha=0.7, zorder=0)
        ax.text(len(T1_METHODS) - 0.6, 0.508, "chance (AUC = 0.5)",
                fontsize=10, color="#444444", ha="right", va="bottom",
                style="italic")


fig2, axes2 = plt.subplots(1, 2, figsize=(13.0, 5.6))
fig2.patch.set_facecolor("white")
fig2.subplots_adjust(wspace=0.22, bottom=0.34, top=0.84)
for ax2, (title, vals, errs, ylim) in zip(axes2, [
    ("Overall AUC",                 T1_OVERALL, T1_OVERALL_STD, (0.65, 1.02)),
    ("FakeVideo-RealAudio (FV-RA)", T1_FVRA,    T1_FVRA_STD,    (0.30, 1.02)),
]):
    _draw_t1_panel(ax2, title, vals, errs, ylim, draw_chance=("FV-RA" in title))

fig2.legend(handles=T1_LEGEND_HANDLES, fontsize=11,
            loc="lower center", bbox_to_anchor=(0.5, -0.02),
            ncol=5, frameon=False, handlelength=1.8, columnspacing=2.0)
fig2.suptitle(
    f"Cross-Domain Detection AUC Across Methods — discriminating panels\n({SUBTITLE})",
    fontsize=13.5, y=0.99)
plt.savefig(f"{OUT}/fig_appendix_table1_main.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(f"{OUT}/fig_appendix_table1_main.pdf", bbox_inches="tight", facecolor="white")
print("Saved: fig_appendix_table1_main.png/.pdf")
plt.close()

print("\nAll done.")
