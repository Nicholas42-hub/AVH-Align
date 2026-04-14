"""
Figure 2: Subspace OOD Probe
Grouped bar chart comparing OOD probe AUC across representation subspaces.

Data from Section 6 / Step 6 of PAPER_STORY_OVERVIEW.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, dim, in_domain, ood, domain_probe, color, hatch)
subspace_data = [
    # A2
    ("A2\n$Z_c$",   1024, 0.9420, 0.7052, 0.9975, "#4E79A7", ""),
    ("A2\n$v_c$",    512, 0.5704, 0.8906, 0.9944, "#6B9EC7", "//"),
    # A5
    ("A5\n$s_v$",    256, 0.5672, 0.8138, 0.9921, "#76B7B2", ""),
    ("A5\n$u_v$",    256, 0.7879, 0.8248, 0.9668, "#59A14F", ""),
    # A6
    ("A6\n$s_v$",    256, 0.5297, 0.7258, 0.9890, "#86C37A", ""),
    ("A6\n$u_v$",    256, 0.7818, 0.9238, 0.9763, "#2D8047", ""),
]

labels      = [d[0] for d in subspace_data]
in_domain   = [d[2] for d in subspace_data]
ood         = [d[3] for d in subspace_data]
domain_pr   = [d[4] for d in subspace_data]
colors      = [d[5] for d in subspace_data]
hatches     = [d[6] for d in subspace_data]

x = np.arange(len(labels))
bar_w = 0.24

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Three grouped bars per subspace
bars_in  = ax.bar(x - bar_w, in_domain, bar_w, label="In-domain probe AUC",
                  color=[c + "CC" for c in colors], edgecolor="#444", linewidth=0.8)
bars_ood = ax.bar(x,         ood,       bar_w, label="OOD probe AUC  ← key",
                  color=colors,          edgecolor="#222", linewidth=1.2)
bars_dom = ax.bar(x + bar_w, domain_pr, bar_w, label="Domain probe AUC",
                  color="#BBBBBB",       edgecolor="#666", linewidth=0.8,
                  hatch="xx")

# Colour each bar individually (in-domain / domain bars keep tint)
for i, (bar_i, bar_o, color) in enumerate(zip(bars_in, bars_ood, colors)):
    bar_i.set_facecolor(color + "88")
    bar_o.set_facecolor(color)

# Annotate OOD bars
for i, (bar, val) in enumerate(zip(bars_ood, ood)):
    ypos = val + 0.007
    weight = "bold" if labels[i].startswith("A6\n$u") else "normal"
    fs = 9.5 if weight == "bold" else 8.5
    ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.4f}",
            ha="center", va="bottom", fontsize=fs, fontweight=weight,
            color="#111111" if weight == "bold" else "#444444")

# Highlight A6 u_v OOD bar with annotation
star_x = bars_ood[5].get_x() + bars_ood[5].get_width() / 2
ax.annotate(
    "★  A6 $u_v$ OOD = 0.9238\n(strongest visual-unique carrier)",
    xy=(star_x, 0.9238 + 0.015),
    xytext=(star_x - 1.3, 0.97),
    fontsize=8.5, color="#2D8047", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#2D8047", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
              edgecolor="#2D8047", alpha=0.9),
)

# Reference line: A2 Z_c OOD baseline
ax.axhline(0.7052, color="#4E79A7", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(5.65, 0.714, "A2 $Z_c$ OOD baseline (0.7052)",
        fontsize=8, color="#4E79A7", va="bottom")

# Separator lines between model groups
for xsep in [1.5, 3.5]:
    ax.axvline(xsep, color="#DDDDDD", linewidth=1.0, linestyle=":")

# Group labels
ax.text(0.5,  0.47, "A2", ha="center", fontsize=11, fontweight="bold",
        color="#4E79A7", transform=ax.get_xaxis_transform())
ax.text(2.5,  0.47, "A5", ha="center", fontsize=11, fontweight="bold",
        color="#59A14F", transform=ax.get_xaxis_transform())
ax.text(4.5,  0.47, "A6", ha="center", fontsize=11, fontweight="bold",
        color="#2D8047", transform=ax.get_xaxis_transform())

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("AUC", fontsize=11)
ax.set_ylim(0.45, 1.07)
ax.set_yticks(np.arange(0.5, 1.05, 0.05))
ax.yaxis.set_tick_params(labelsize=9)
ax.set_title(
    "Figure 2 — Three-way factorization exposes a stronger visual-unique OOD subspace\n"
    "($u_v$: modality-unique visual component;  $s_v$: synergistic visual component)",
    fontsize=11, fontweight="bold", pad=10,
)
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

# Legend
ax.legend(
    handles=[
        mpatches.Patch(facecolor="#AAAAAA88", edgecolor="#444", label="In-domain probe AUC"),
        mpatches.Patch(facecolor="#2D8047",   edgecolor="#222", label="OOD probe AUC  ← key"),
        mpatches.Patch(facecolor="#BBBBBB",   edgecolor="#666", hatch="xx",
                       label="Domain probe AUC  (high = not invariant)"),
    ],
    loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
)

plt.tight_layout()
plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig2_subspace_ood_probe.pdf",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig2_subspace_ood_probe.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
print("Fig 2 saved.")
