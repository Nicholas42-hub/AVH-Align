"""
Figure 1: Model Schematic
Binary decomposition (A2) vs Three-way factorization with domain push-pull (A5/A6)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_VIDEO   = "#4E79A7"   # blue  — visual modality
C_AUDIO   = "#F28E2B"   # orange — audio modality
C_SHARED  = "#76B7B2"   # teal   — synergistic/shared
C_UNIQUE  = "#59A14F"   # green  — modality-unique
C_REDUND  = "#E15759"   # red    — redundant/spurious
C_DOMAIN  = "#B07AA1"   # purple — domain push-pull
C_HEAD    = "#9C755F"   # brown  — classifier head
C_ARROW   = "#555555"
C_BG      = "#F9F9F9"
C_BORDER  = "#CCCCCC"

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("white")

# Left panel = A2 (binary), Right panel = A5/A6 (three-way)
ax_left  = fig.add_axes([0.03, 0.05, 0.42, 0.88])
ax_right = fig.add_axes([0.55, 0.05, 0.44, 0.88])

for ax in (ax_left, ax_right):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")


# ── Helper functions ──────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, color, label, fontsize=9, text_color="white",
                alpha=1.0, linestyle="-", edgecolor=None):
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=ec, linewidth=1.4,
        alpha=alpha, linestyle=linestyle,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color,
            wrap=True)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
    )


def dashed_arrow(ax, x1, y1, x2, y2, color=C_DOMAIN, lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.0",
        ),
    )


def grl_box(ax, x, y):
    """Small GRL indicator."""
    box = FancyBboxPatch(
        (x - 0.45, y - 0.25), 0.9, 0.5,
        boxstyle="round,pad=0.04",
        facecolor=C_DOMAIN, edgecolor=C_DOMAIN, linewidth=1.2, alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(x, y, "GRL", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white")


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — A2: Binary decomposition
# ══════════════════════════════════════════════════════════════════════════════

ax = ax_left
ax.set_title("A2 — Binary Decomposition", fontsize=13, fontweight="bold",
             pad=6, color="#222222")

# Input features
rounded_box(ax, 3.5, 9.0, 2.4, 0.7, C_VIDEO,  "z_v  (video feat. 1024-d)", fontsize=8.5)
rounded_box(ax, 7.0, 9.0, 2.4, 0.7, C_AUDIO,  "z_a  (audio feat. 1024-d)", fontsize=8.5)

# Causal encoder
arrow(ax, 3.5, 8.65, 3.5, 7.85)
arrow(ax, 7.0, 8.65, 7.0, 7.85)
rounded_box(ax, 3.5, 7.5, 2.4, 0.65, C_VIDEO, "F_c  (video causal)", fontsize=8.5)
rounded_box(ax, 7.0, 7.5, 2.4, 0.65, C_AUDIO, "F_c  (audio causal)", fontsize=8.5)

# Spurious encoder
arrow(ax, 3.5, 7.175, 3.5, 6.35)
arrow(ax, 7.0, 7.175, 7.0, 6.35)
rounded_box(ax, 3.5, 6.0, 2.4, 0.65, C_REDUND, "F_s  (video spur.)",  fontsize=8.5)
rounded_box(ax, 7.0, 6.0, 2.4, 0.65, C_REDUND, "F_s  (audio spur.)",  fontsize=8.5)

# Z_c concatenation
arrow(ax, 3.5, 5.675, 4.0, 4.95)
arrow(ax, 7.0, 5.675, 6.5, 4.95)
rounded_box(ax, 5.25, 4.65, 3.8, 0.65, C_HEAD,
            "Z_c = cat(v_c, a_c)   [1024-d]", fontsize=8.5)

# Z_s concat label
arrow(ax, 3.5, 5.675, 3.5, 5.1)
ax.text(3.5, 4.8, "Z_s = cat(v_s, a_s)", ha="center", va="center",
        fontsize=8, color=C_REDUND, style="italic")

# Classifier head
arrow(ax, 5.25, 4.325, 5.25, 3.55)
rounded_box(ax, 5.25, 3.2, 3.0, 0.65, C_HEAD, "Classifier head  (MLP)", fontsize=8.5)

# Output
arrow(ax, 5.25, 2.875, 5.25, 2.25)
rounded_box(ax, 5.25, 1.9, 2.4, 0.65, "#333333", "P(fake)", fontsize=9)

# Annotation: audio-dominant cursor
ax.annotate(
    "Audio-dominant\nrouting (shortcut)",
    xy=(7.0, 4.65), xytext=(8.6, 4.65),
    fontsize=7.5, color=C_REDUND, ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C_REDUND, lw=1.0),
)

# Legend hint
ax.text(5.25, 0.9, "Z_c entangles modality signal\nand domain shortcut",
        ha="center", va="center", fontsize=8.5, color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                  edgecolor="#FFBF00", alpha=0.85))


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — A5 / A6: Three-way factorization
# ══════════════════════════════════════════════════════════════════════════════

ax = ax_right
ax.set_title("A5 / A6 — Three-way Factorization  (+domain push-pull in A6)",
             fontsize=13, fontweight="bold", pad=6, color="#222222")

# Input features
rounded_box(ax, 3.0, 9.1, 2.2, 0.65, C_VIDEO,  "z_v  [1024-d]", fontsize=8.5)
rounded_box(ax, 7.5, 9.1, 2.2, 0.65, C_AUDIO,  "z_a  [1024-d]", fontsize=8.5)

# CCD splits (each into s and h)
arrow(ax, 3.0, 8.775, 3.0, 8.05)
arrow(ax, 7.5, 8.775, 7.5, 8.05)
rounded_box(ax, 3.0, 7.7, 2.4, 0.65, C_SHARED,  "CCD_v\n→ s_v [256] + h_v [256]",
            fontsize=7.5)
rounded_box(ax, 7.5, 7.7, 2.4, 0.65, C_SHARED,  "CCD_a\n→ s_a [256] + h_a [256]",
            fontsize=7.5)

# Synergistic alignment (L_SDA)
ax.annotate("", xy=(5.5, 7.55), xytext=(4.2, 7.55),
            arrowprops=dict(arrowstyle="<->", color=C_SHARED, lw=1.6))
ax.text(5.25, 7.2, "$\\mathcal{L}_{SDA}$ (Sinkhorn align $s_v \\leftrightarrow s_a$)",
        ha="center", va="center", fontsize=7.5, color=C_SHARED)

# URD splits
arrow(ax, 3.0, 7.375, 3.0, 6.55)
arrow(ax, 7.5, 7.375, 7.5, 6.55)
rounded_box(ax, 3.0, 6.2, 2.4, 0.65, C_UNIQUE,
            "URD_v\n→ u_v [256] + r_v = h_v − u_v", fontsize=7.5)
rounded_box(ax, 7.5, 6.2, 2.4, 0.65, C_UNIQUE,
            "URD_a\n→ u_a [256] + r_a = h_a − u_a", fontsize=7.5)

# L_MI annotation
ax.text(3.0, 5.65, "$\\mathcal{L}_{MI}$: CE loss on $u_v$",
        ha="center", va="center", fontsize=7.5, color=C_UNIQUE)
ax.text(7.5, 5.65, "$\\mathcal{L}_{MI}$: CE loss on $u_a$",
        ha="center", va="center", fontsize=7.5, color=C_UNIQUE)

# Redundant branch
ax.text(5.25, 5.3, "$r_v, r_a$  — redundant (spurious / shortcut)",
        ha="center", va="center", fontsize=7.5, color=C_REDUND, style="italic")
ax.text(5.25, 4.95, "$\\mathcal{L}_{Dis}$: modality disc. on $r$    "
        "$\\mathcal{L}_{orth}$: $u \\perp r$",
        ha="center", va="center", fontsize=7.5, color=C_REDUND)

# Causal head (task branch)
arrow(ax, 3.0, 5.875, 3.8, 4.4)
arrow(ax, 7.5, 5.875, 6.7, 4.4)
rounded_box(ax, 5.25, 4.1, 4.2, 0.65, C_HEAD,
            "Z_c = cat(s_v, u_v, s_a, u_a)   [1024-d]", fontsize=8.5)

# A6: GRL + domain push-pull
grl_box(ax, 5.25, 3.38)
arrow(ax, 5.25, 3.78, 5.25, 3.63)
rounded_box(ax, 5.25, 3.0, 3.4, 0.55, C_DOMAIN,
            "A6: GRL → domain classifier  ($\\mathcal{L}_{dadv}$, expel domain from $Z_c$)",
            fontsize=7.2, text_color="white")
ax.text(9.0, 3.0, "$Z_s$ concentrates\ndomain info\n($\\mathcal{L}_{ddis}$)",
        ha="center", va="center", fontsize=7.5, color=C_DOMAIN,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3E5F5",
                  edgecolor=C_DOMAIN, alpha=0.85))
dashed_arrow(ax, 8.2, 3.0, 7.5, 3.0, color=C_DOMAIN)

# Classifier head
arrow(ax, 5.25, 2.67, 5.25, 2.1)
rounded_box(ax, 5.25, 1.75, 3.0, 0.65, C_HEAD, "Classifier head  (MLP)", fontsize=8.5)

# Output
arrow(ax, 5.25, 1.425, 5.25, 0.85)
rounded_box(ax, 5.25, 0.5, 2.4, 0.65, "#333333", "P(fake)", fontsize=9)

# Highlight u_v arrow in green
ax.annotate(
    "$\\mathbf{u_v}$ — visual-unique\nOOD carrier  (AUC 0.92)",
    xy=(3.6, 4.35), xytext=(0.6, 3.5),
    fontsize=7.5, color=C_UNIQUE, ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C_UNIQUE, lw=1.2),
)


# ── Divider line between panels ───────────────────────────────────────────────
fig.add_artist(
    matplotlib.lines.Line2D([0.505, 0.505], [0.05, 0.93],
                             transform=fig.transFigure,
                             color=C_BORDER, linewidth=1.2)
)

# ── Panel labels ──────────────────────────────────────────────────────────────
fig.text(0.03,  0.97, "(a)", fontsize=13, fontweight="bold", color="#333333",
         va="top")
fig.text(0.55,  0.97, "(b)", fontsize=13, fontweight="bold", color="#333333",
         va="top")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_VIDEO,  label="Visual modality branch"),
    mpatches.Patch(color=C_AUDIO,  label="Audio modality branch"),
    mpatches.Patch(color=C_SHARED, label="Synergistic / shared ($s$)"),
    mpatches.Patch(color=C_UNIQUE, label="Modality-unique ($u$)  ← key OOD carrier"),
    mpatches.Patch(color=C_REDUND, label="Redundant / spurious ($r$)"),
    mpatches.Patch(color=C_DOMAIN, label="A6 domain push-pull (GRL)"),
    mpatches.Patch(color=C_HEAD,   label="Classifier head"),
]
fig.legend(
    handles=legend_items, loc="lower center",
    ncol=4, fontsize=8.5, framealpha=0.9,
    bbox_to_anchor=(0.5, -0.02),
    edgecolor=C_BORDER,
)

plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig1_model_schematic.pdf",
    dpi=300, bbox_inches="tight", facecolor="white",
)
plt.savefig(
    "/data/projects/punim2637/nnliang/AVH-Align/submission_prep/figures/fig1_model_schematic.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
print("Fig 1 saved.")
