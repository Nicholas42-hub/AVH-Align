"""
Figure 3: Routing Evidence — Head Ablation
Grouped bar chart showing ΔFV-RA and ΔRV-FA for diagnostic ablations.

The key visual claim:
  Only A6 exhibits a large routing drop (−0.35) when the visual-unique
  channel u_v is masked, demonstrating dedicated u_v routing for FV-RA.
  A2 exhibits a symmetric large drop for audio masking (ΔAV1M = −0.47),
  confirming audio-dominant routing in the binary model.

Data from Section 6 / Step 7 of PAPER_STORY_OVERVIEW.md
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_A2  = "#4E79A7"
C_A5  = "#59A14F"
C_A6  = "#E15759"
C_AUD = "#F28E2B"
C_VIS = "#76B7B2"

# ══════════════════════════════════════════════════════════════════════════════
#  Main figure: two-panel
#  Left  — FV-RA routing drops for each model (primary mechanism figure)
#  Right — Symmetric audio-routing proof for A2 (supporting annotation)
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax_main, ax_side) = plt.subplots(
    1, 2, figsize=(13, 5.5),
    gridspec_kw={"width_ratios": [2.6, 1]},
)
fig.patch.set_facecolor("white")

# ── LEFT PANEL: ΔFV-RA (most diagnostic ablation per model) ──────────────────

# Data: (model, ablation label, baseline FV-RA, ablated FV-RA, delta, model colour)
fvra_data = [
    ("A2",  "mask $v_c$\n(visual causal)", 0.4949, 0.4613, -0.0336, C_A2),
    ("A5",  "mask visual\n($s_v$+$u_v$)",   0.5860, 0.4729, -0.1131, C_A5),
    ("A6",  "mask $u_v$\n(visual-unique)",  0.7206, 0.3661, -0.3544, C_A6),
]

models    = [d[0] for d in fvra_data]
ablations = [d[1] for d in fvra_data]
baselines = [d[2] for d in fvra_data]
ablated   = [d[3] for d in fvra_data]
deltas    = [d[4] for d in fvra_data]
mcolors   = [d[5] for d in fvra_data]

x = np.arange(len(models))
bar_w = 0.32

ax = ax_main
ax.set_facecolor("white")

# Baseline bars (lighter shade)
bars_base = ax.bar(x - bar_w / 2, baselines, bar_w,
                   color=[c + "55" for c in mcolors],
                   edgecolor=[c for c in mcolors], linewidth=1.4,
                   label="Baseline FV-RA AUC")

# Ablated bars (solid)
bars_abl = ax.bar(x + bar_w / 2, ablated, bar_w,
                  color=mcolors,
                  edgecolor=[c for c in mcolors], linewidth=1.4,
                  label="Ablated FV-RA AUC")

# Delta annotation with double-headed arrow
for i, (base, abl, delta, color) in enumerate(zip(baselines, ablated, deltas, mcolors)):
    xi_base = x[i] - bar_w / 2
    xi_abl  = x[i] + bar_w / 2
    top = max(base, abl) + 0.02
    ax.annotate(
        "", xy=(xi_abl + bar_w / 2 - 0.03, abl),
           xytext=(xi_abl + bar_w / 2 - 0.03, base),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.8),
    )
    ax.text(xi_abl + bar_w / 2 + 0.02, (base + abl) / 2,
            f"Δ = {delta:+.4f}",
            ha="left", va="center", fontsize=9.5,
            fontweight="bold" if abs(delta) > 0.2 else "normal",
            color=color)

# Ablation label below each group
for i, abl_label in enumerate(ablations):
    ax.text(x[i], -0.03, abl_label,
            ha="center", va="top", fontsize=8.5, color=mcolors[i],
            transform=ax.get_xaxis_transform())

# Value labels on bars
for bars in [bars_base, bars_abl]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

# Highlight A6 delta
ax.annotate(
    "★  A6 drops − 0.35 when $u_v$ masked\n→ FV-RA decision routed via $u_v$",
    xy=(x[2] + bar_w / 2 + 0.05, (baselines[2] + ablated[2]) / 2),
    xytext=(2.35, 0.6),
    fontsize=9, color=C_A6, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=C_A6, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA",
              edgecolor=C_A6, alpha=0.9),
)

ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n({abl})" for m, abl in zip(models, ablations)],
                   fontsize=9.5)
ax.set_ylabel("FV-RA AUC", fontsize=11)
ax.set_ylim(0.0, 1.05)
ax.set_yticks(np.arange(0.0, 1.05, 0.1))
ax.yaxis.set_tick_params(labelsize=9)
ax.set_title(
    "Figure 3 (left) — A6 routes FV-RA decisions through the visual-unique pathway ($u_v$)\n"
    "Baseline vs. ablated FV-RA AUC for the most diagnostic ablation per model",
    fontsize=10, fontweight="bold", pad=8,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")
ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)


# ── RIGHT PANEL: A2 audio symmetry diagnostic ─────────────────────────────────
# Shows A2 is equally sensitive to audio ablation on AV1M (confirming audio routing)

ax2 = ax_side
ax2.set_facecolor("white")

side_data = [
    # (label, baseline, ablated, delta, color)
    ("A2\nFV-RA\nmask $v_c$",    0.4949, 0.4613, -0.0336, C_VIS),
    ("A2\nAV1M\nmask audio",     0.9975, 0.5280, -0.4695, C_AUD),
]
sx    = np.arange(2)
sbars_base = ax2.bar(sx - 0.2, [d[1] for d in side_data], 0.38,
                     color=[d[4] + "55" for d in side_data],
                     edgecolor=[d[4] for d in side_data], linewidth=1.4,
                     label="Baseline")
sbars_abl  = ax2.bar(sx + 0.2, [d[2] for d in side_data], 0.38,
                     color=[d[4] for d in side_data],
                     edgecolor=[d[4] for d in side_data], linewidth=1.4,
                     label="Ablated")

for i, d in enumerate(side_data):
    ax2.annotate(
        "", xy=(sx[i] + 0.38, d[2]),
           xytext=(sx[i] + 0.38, d[1]),
        arrowprops=dict(arrowstyle="<->", color=d[4], lw=1.8),
    )
    ax2.text(sx[i] + 0.42, (d[1] + d[2]) / 2,
             f"Δ = {d[3]:+.3f}",
             ha="left", va="center", fontsize=9,
             fontweight="bold" if abs(d[3]) > 0.2 else "normal",
             color=d[4])

for sbars in [sbars_base, sbars_abl]:
    for bar in sbars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax2.set_xticks(sx)
ax2.set_xticklabels([d[0] for d in side_data], fontsize=8.5)
ax2.set_ylabel("AUC", fontsize=10)
ax2.set_ylim(0.0, 1.15)
ax2.set_yticks(np.arange(0.0, 1.1, 0.2))
ax2.yaxis.set_tick_params(labelsize=8)
ax2.set_title(
    "Figure 3 (right)\nA2 audio-dominance\ncontext diagnostic",
    fontsize=9.5, fontweight="bold", pad=6,
)
ax2.text(0.5, 1.08,
         "A2 is equally\nsensitive —\nbut to audio",
         ha="center", va="top", fontsize=8, color=C_AUD,
         transform=ax2.get_xaxis_transform())
ax2.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")
ax2.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.7)
ax2.spines[["top", "right"]].set_visible(False)


plt.tight_layout(w_pad=2.5)
plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig3_routing_ablation.pdf",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig3_routing_ablation.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
print("Fig 3 saved.")
