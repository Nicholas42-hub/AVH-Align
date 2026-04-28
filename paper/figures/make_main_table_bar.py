"""Bar plot of FV-RA AUC across model variants (seed 44 only).

Companion appendix figure to Table 1 of the main paper. Per Caren's review
note, this uses a single seed (44) so no error bars are drawn.

Run from this directory: python make_main_table_bar.py
Outputs: main_table_bar.pdf, main_table_bar.png
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# FV-RA AUC at seed 44 on FAVC 7030 test split (N=3158 in FV-RA)
# Sources:
#   AVH-Align/sup     -> external (Smeu et al. 2025), single published checkpoint
#   Two-way baseline  -> outputs_A2_50ep_seed44/results_favc7030_full
#   DANN/GRL          -> outputs_DANN_seed44/results_favc_7030_full (alpha=1.0)
#   CORAL             -> outputs_CORAL_50ep_seed44/results_favc7030_full
#   Three-way no push -> outputs_A5_seed44/results_favc_7030_full
#   TriRoute (ours)   -> outputs_A6_av1m_pseudodomain_v2_50ep_seed44 (p=0 baseline)
labels = [
    "AVH-Align/sup\n[external]",
    "Two-way\nbaseline",
    "+ DANN/GRL\n($\\alpha{=}1$)",
    "+ CORAL",
    "Three-way,\nno push-pull",
    "TriRoute\n(ours)",
]
values = [0.519, 0.4371, 0.5319, 0.5208, 0.5889, 0.8272]
is_method = [False, False, False, False, False, True]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(7.0, 3.4))
bars = ax.bar(
    x,
    values,
    color=["#9CA3AF", "#9CA3AF", "#60A5FA", "#60A5FA", "#FBBF24", "#DC2626"],
    edgecolor="#1F2937",
    linewidth=0.8,
)
ax.axhline(0.5, ls="--", lw=0.9, color="#6B7280", alpha=0.7)
ax.text(len(labels) - 0.5, 0.51, "chance", ha="right", va="bottom",
        color="#6B7280", fontsize=8)

for bar, v, m in zip(bars, values, is_method):
    label = f"$\\mathbf{{{v:.3f}}}$" if m else f"{v:.3f}"
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, label,
            ha="center", va="bottom",
            fontsize=8.5,
            fontweight="bold" if m else "normal")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("FV-RA AUC (single seed = 44)")
ax.set_ylim(0.3, 0.95)
ax.set_title("Cross-domain FV-RA AUC on FakeAVCeleb (AV1M $\\to$ FAVC, $N{=}3{,}158$, seed 44)")
ax.grid(True, axis="y", ls=":", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("main_table_bar.pdf", bbox_inches="tight", dpi=300)
fig.savefig("main_table_bar.png", bbox_inches="tight", dpi=300)
print("Wrote main_table_bar.pdf / .png")
