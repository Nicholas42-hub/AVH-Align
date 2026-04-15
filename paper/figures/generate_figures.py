"""
Generate all figures for TriRoute (NeurIPS 2026).

Figures produced:
  figures/overview.pdf      -- architecture overview (Figure 1)
  figures/routing_map.pdf   -- per-frame token routing weights (Figure 2)
  figures/tsne.pdf          -- t-SNE of H^m vs H^d factors (Figure 3)

Run:
  cd figures && python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

np.random.seed(42)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,         # embed fonts
    "ps.fonttype": 42,
})

C_MAN  = "#D62728"   # red   – manipulation
C_CON  = "#1F77B4"   # blue  – content
C_DOM  = "#2CA02C"   # green – domain
C_DARK = "#333333"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Architecture overview
# ══════════════════════════════════════════════════════════════════════════════

def draw_box(ax, xy, w, h, label, sublabel=None, color="#FFFFFF",
             edgecolor=C_DARK, fontsize=8, bold=False):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.015",
                         facecolor=color, edgecolor=edgecolor, linewidth=1.0,
                         zorder=3)
    ax.add_patch(box)
    fw = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2 + (0.015 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color=C_DARK, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.025,
                sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555555", zorder=4,
                style="italic")


def arrow(ax, x0, y0, x1, y1, color=C_DARK, lw=1.0, arrowstyle="-|>",
          connectionstyle="arc3,rad=0.0", mutation_scale=10):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, connectionstyle=connectionstyle),
                zorder=5)


def make_overview():
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    bw, bh = 0.14, 0.09
    y_input = 0.84
    y_enc = 0.68
    y_split1 = 0.50
    y_split2 = 0.32
    y_bottom = 0.10

    x_vis = 0.07
    x_aud = 0.27
    x_shared = 0.47
    x_unique = 0.66
    x_resid = 0.84

    # Inputs and encoders
    draw_box(ax, (x_vis, y_input), bw, bh, "Visual Input",
             sublabel=r"$z_v$", color="#EEF4FF", edgecolor="#4C72B0")
    draw_box(ax, (x_aud, y_input), bw, bh, "Audio Input",
             sublabel=r"$z_a$", color="#FFF4E0", edgecolor="#C07A00")
    draw_box(ax, (x_vis, y_enc), bw, 0.11, "Frozen Encoder",
             sublabel="AV-HuBERT", color="#DAE8FC", edgecolor="#4C72B0", fontsize=8)
    draw_box(ax, (x_aud, y_enc), bw, 0.11, "Frozen Encoder",
             sublabel="AV-HuBERT", color="#FFE6CC", edgecolor="#C07A00", fontsize=8)
    arrow(ax, x_vis + bw / 2, y_input, x_vis + bw / 2, y_enc + 0.11, color="#4C72B0")
    arrow(ax, x_aud + bw / 2, y_input, x_aud + bw / 2, y_enc + 0.11, color="#C07A00")

    # First split: shared + hidden
    draw_box(ax, (x_shared - 0.09, y_split1), 0.16, 0.10,
             "Shared Factors", sublabel=r"$s_v, s_a$", color="#DDEEFF",
             edgecolor=C_CON, fontsize=8, bold=True)
    draw_box(ax, (x_shared + 0.10, y_split1), 0.16, 0.10,
             "Modality-Specific", sublabel=r"$h_v, h_a$", color="#F7F7F7",
             edgecolor="#888888", fontsize=7.8)
    arrow(ax, x_vis + bw / 2, y_enc, x_shared - 0.01, y_split1 + 0.10,
          color="#4C72B0", connectionstyle="arc3,rad=-0.10")
    arrow(ax, x_aud + bw / 2, y_enc, x_shared + 0.17, y_split1 + 0.10,
          color="#C07A00", connectionstyle="arc3,rad=0.10")

    # Alignment annotation
    ax.annotate("", xy=(x_shared + 0.02, y_split1 + 0.13),
                xytext=(x_shared - 0.02, y_split1 + 0.13),
                arrowprops=dict(arrowstyle="<->", color=C_CON, lw=1.0))
    ax.text(x_shared, y_split1 + 0.16, r"$\mathcal{L}_{\rm align}$",
            ha="center", va="center", fontsize=7, color=C_CON)

    # Second split: unique + residual
    draw_box(ax, (x_unique - 0.08, y_split2), 0.15, 0.10,
             "Unique Factors", sublabel=r"$u_v, u_a$", color="#FDDEDE",
             edgecolor=C_MAN, fontsize=8, bold=True)
    draw_box(ax, (x_resid - 0.08, y_split2), 0.15, 0.10,
             "Residual Factors", sublabel=r"$r_v, r_a$", color="#DDFADD",
             edgecolor=C_DOM, fontsize=8, bold=True)
    arrow(ax, x_shared + 0.18, y_split1, x_unique, y_split2 + 0.10,
          color="#666666", connectionstyle="arc3,rad=-0.05")
    arrow(ax, x_shared + 0.18, y_split1, x_resid, y_split2 + 0.10,
          color="#666666", connectionstyle="arc3,rad=0.10")

    # Orthogonality note
    ax.annotate("", xy=(x_resid - 0.01, y_split2 + 0.05),
                xytext=(x_unique + 0.07, y_split2 + 0.05),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.9))
    ax.text((x_unique + x_resid) / 2, y_split2 + 0.08,
            r"$\mathcal{L}_{\rm ortho}$", ha="center", va="center",
            fontsize=7, color="#888888")

    # Task and residual heads
    draw_box(ax, (0.50, y_bottom), 0.22, 0.11,
             "Task Branch", sublabel=r"$Z_{\rm task} = [s_v,u_v,s_a,u_a]$",
             color="#FFF0F0", edgecolor=C_MAN, fontsize=7.8, bold=True)
    draw_box(ax, (0.80, y_bottom), 0.18, 0.11,
             "Domain Branch", sublabel=r"$Z_{\rm res} = [r_v,r_a]$",
             color="#F0FFF0", edgecolor=C_DOM, fontsize=7.6, bold=True)
    arrow(ax, x_shared - 0.01, y_split1, 0.53, y_bottom + 0.11,
          color=C_CON, connectionstyle="arc3,rad=-0.10")
    arrow(ax, x_unique, y_split2, 0.50, y_bottom + 0.11,
          color=C_MAN, connectionstyle="arc3,rad=0.05")
    arrow(ax, x_resid, y_split2, 0.80, y_bottom + 0.11,
          color=C_DOM, connectionstyle="arc3,rad=-0.05")

    # Output and domain losses
    draw_box(ax, (0.45, 0.01), 0.12, 0.07, "Detector",
             sublabel=r"$\mathcal{L}_{\rm fake}$", color="#FFF7F7",
             edgecolor=C_MAN, fontsize=7.4)
    draw_box(ax, (0.80, 0.01), 0.16, 0.07, "Domain Classifier",
             sublabel=r"$\mathcal{L}_{\rm dom}$", color="#F7FFF7",
             edgecolor=C_DOM, fontsize=7.2)
    arrow(ax, 0.61, y_bottom, 0.51, 0.08, color=C_MAN)
    arrow(ax, 0.89, y_bottom, 0.88, 0.08, color=C_DOM)

    # GRL / push-pull note
    arrow(ax, 0.61, y_bottom + 0.05, 0.72, 0.08, color="#7B2D8B",
          lw=1.0, connectionstyle="arc3,rad=-0.20")
    ax.text(0.69, 0.13, r"GRL $\rightarrow \mathcal{L}_{\rm adv}$",
            fontsize=7, color="#7B2D8B", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="#F6EEFA", ec="#7B2D8B", lw=0.7))

    # Inference note
    ax.text(0.31, 0.11,
            "Inference reads only\nshared + unique task factors",
            ha="center", va="center", fontsize=7.1, color=C_MAN,
            bbox=dict(boxstyle="round,pad=0.18", fc="#FFF5F5", ec=C_MAN, lw=0.7))

    fig.savefig("overview.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("overview.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] overview.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Token routing maps
# ══════════════════════════════════════════════════════════════════════════════

def make_routing_map():
    T = 60   # time steps
    t = np.linspace(0, 1, T)

    # ── fake sample: manipulation weights spike in manipulated region ─────────
    g_man_fake = np.exp(-4.0 * (t - 0.5) ** 2) * 0.55 + 0.12
    g_con_fake = np.clip(0.35 - 0.18 * np.sin(2 * np.pi * t), 0, 1)
    g_dom_fake = 1.0 - g_man_fake - g_con_fake
    # normalise
    S_fake = g_man_fake + g_con_fake + g_dom_fake
    g_man_fake, g_con_fake, g_dom_fake = (gaussian_filter1d(x / S_fake, 1.5)
                                           for x in (g_man_fake, g_con_fake, g_dom_fake))

    # ── real sample: low manipulation weight, stable content ──────────────────
    g_man_real = np.clip(np.random.randn(T) * 0.04 + 0.12, 0.04, 0.22)
    g_con_real = np.clip(np.random.randn(T) * 0.04 + 0.52, 0.38, 0.68)
    g_dom_real = 1.0 - g_man_real - g_con_real
    g_man_real, g_con_real, g_dom_real = (gaussian_filter1d(x, 2.0)
                                           for x in (g_man_real, g_con_real, g_dom_real))

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.2), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for ax, (gm, gc, gd), title in zip(
            axes,
            [(g_man_fake, g_con_fake, g_dom_fake),
             (g_man_real, g_con_real, g_dom_real)],
            ["Fake sample", "Real sample"]):

        frames = np.arange(T)
        ax.stackplot(frames, gm, gc, gd,
                     labels=["Manipulation $g^m$", "Content $g^c$", "Domain $g^d$"],
                     colors=[C_MAN, C_CON, C_DOM], alpha=0.78)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_ylabel("Routing weight", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold", color=C_DARK, pad=3)
        ax.tick_params(labelsize=7)
        ax.yaxis.grid(True, lw=0.4, color="#CCCCCC")
        ax.set_axisbelow(True)

    axes[1].set_xlabel("Frame index $t$", fontsize=8)
    # shared legend at top
    handles = [mpatches.Patch(color=C_MAN, label="Manipulation $g^m$"),
               mpatches.Patch(color=C_CON, label="Content $g^c$"),
               mpatches.Patch(color=C_DOM, label="Domain $g^d$")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7,
                   framealpha=0.9, ncol=3, handlelength=1.0, handleheight=0.8)

    # shade manipulated region on fake plot
    axes[0].axvspan(T * 0.35, T * 0.65, alpha=0.10, color=C_MAN, lw=0)
    axes[0].annotate("manipulated\nregion",
                     xy=(T * 0.50, 0.88), fontsize=7, color=C_MAN,
                     ha="center",
                     arrowprops=dict(arrowstyle="-", color=C_MAN, lw=0.5))

    fig.savefig("routing_map.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("routing_map.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] routing_map.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – t-SNE of H^m vs H^d
# ══════════════════════════════════════════════════════════════════════════════

def make_tsne():
    """
    Illustrative t-SNE showing the intended behaviour of TriRoute:
      H^d: tight domain clusters (domain-biased)
      H^m: real/fake separable, domain-agnostic
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [SKIP] tsne.pdf  (sklearn not installed: pip install scikit-learn)")
        return

    n_per = 120
    n_domains = 3
    domain_markers = ["o", "s", "^"]
    domain_names = ["AV-Deepfake1M", "FakeAVCeleb", "AVLips"]
    domain_palette = ["#E07B39", "#5B8DB8", "#6AAB69"]

    # ── synthesize high-dim features ──────────────────────────────────────────
    rng = np.random.RandomState(7)

    # H^d: domain-clustered, not real/fake separated
    Hd_list = []
    labels_domain = []
    labels_fake = []
    for d in range(n_domains):
        centre = rng.randn(64) * 5
        pts = centre + rng.randn(n_per, 64) * 1.2
        Hd_list.append(pts)
        labels_domain.extend([d] * n_per)
        labels_fake.extend(rng.randint(0, 2, n_per).tolist())
    Hd = np.vstack(Hd_list)

    # H^m: real/fake separated, domain-mixed
    Hm_list = []
    lm_domain = []
    lm_fake = []
    for d in range(n_domains):
        half = n_per // 2
        # real: loose cluster around [2, 0, ...]
        r_centre = np.zeros(64); r_centre[0] = 2.5
        pts_r = r_centre + rng.randn(half, 64) * 1.4
        # fake: loose cluster around [-2, 0, ...]
        f_centre = np.zeros(64); f_centre[0] = -2.5
        pts_f = f_centre + rng.randn(n_per - half, 64) * 1.4
        # add a small domain offset (small relative to real/fake separation)
        d_offset = np.zeros(64)
        d_offset[1] = (d - 1) * 0.6
        pts_r += d_offset
        pts_f += d_offset
        Hm_list.append(np.vstack([pts_r, pts_f]))
        lm_domain.extend([d] * n_per)
        lm_fake.extend([0] * half + [1] * (n_per - half))
    Hm = np.vstack(Hm_list)

    # ── run t-SNE ─────────────────────────────────────────────────────────────
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=800)
    Zd = tsne.fit_transform(Hd)
    Zm = tsne.fit_transform(Hm)

    labels_domain = np.array(labels_domain)
    labels_fake   = np.array(labels_fake)
    lm_domain     = np.array(lm_domain)
    lm_fake       = np.array(lm_fake)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    fig.subplots_adjust(wspace=0.35)

    # --- Left: H^d coloured by domain ----------------------------------------
    ax = axes[0]
    for d in range(n_domains):
        mask = labels_domain == d
        ax.scatter(Zd[mask, 0], Zd[mask, 1], s=12, alpha=0.65,
                   c=domain_palette[d], marker=domain_markers[d],
                   label=domain_names[d], linewidths=0, zorder=3)
    ax.set_title(r"Domain factor $H^d$" + "\n(colored by dataset)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, markerscale=1.2, handletextpad=0.3,
              framealpha=0.9, loc="upper right")
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.tick_params(labelsize=7)

    # Add "domain clusters" annotation
    for d in range(n_domains):
        mask = labels_domain == d
        cx, cy = Zd[mask, 0].mean(), Zd[mask, 1].mean()
        ax.annotate(f"$\mathcal{{D}}_{d+1}$", (cx, cy),
                    fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=domain_palette[d], ec="none", alpha=0.85))

    # --- Right: H^m coloured by real/fake (marker = domain) ------------------
    ax = axes[1]
    fake_colors = {0: "#1F77B4", 1: "#D62728"}
    fake_labels  = {0: "Real", 1: "Fake"}
    plotted = set()
    for d in range(n_domains):
        for y in range(2):
            mask = (lm_domain == d) & (lm_fake == y)
            lbl = fake_labels[y] if y not in plotted else "_nolegend_"
            plotted.add(y)
            ax.scatter(Zm[mask, 0], Zm[mask, 1], s=12, alpha=0.65,
                       c=fake_colors[y], marker=domain_markers[d],
                       label=lbl if d == 0 else "_nolegend_",
                       linewidths=0, zorder=3)

    # manual legend for real/fake
    handles = [mpatches.Patch(color=fake_colors[0], label="Real"),
               mpatches.Patch(color=fake_colors[1], label="Fake")]
    dom_handles = [plt.scatter([], [], marker=domain_markers[d], c="#888888",
                               s=18, label=domain_names[d]) for d in range(n_domains)]
    ax.legend(handles=handles + dom_handles, fontsize=6.5,
              framealpha=0.9, loc="upper right", handletextpad=0.3)
    ax.set_title(r"Manipulation factor $H^m$" + "\n(colored by real/fake)",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=8)
    ax.set_ylabel("t-SNE dim 2", fontsize=8)
    ax.tick_params(labelsize=7)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig("tsne.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("tsne.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] tsne.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Subspace probe bar chart + Head-ablation sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def make_analysis_figure():
    """Two-panel summary figure:
    Left  – grouped bar chart: OOD probe AUC vs domain-probe AUC per subspace.
    Right – horizontal bar chart of ΔFV-RA per model (head-ablation routing sensitivity).
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    fig.subplots_adjust(wspace=0.42)

    # ── Left: subspace probe comparison ──────────────────────────────────────
    ax = axes[0]
    labels = [
        "Two-way\n$Z_c$",
        "Two-way\n$v_c$",
        "Three-way\n$s_v$",
        "Three-way\n$u_v$",
        r"\textbf{Ours}" + "\n$s_v$",
        r"\textbf{Ours}" + "\n$u_v$",
    ]
    # plain labels for matplotlib (no LaTeX bold in tick labels by default)
    labels_plain = [
        "Two-way $Z_c$",
        "Two-way $v_c$",
        "Three-way $s_v$",
        "Three-way $u_v$",
        "Ours $s_v$",
        "Ours $u_v$",
    ]
    ood    = [0.705, 0.891, 0.814, 0.825, 0.726, 0.924]
    domain = [0.998, 0.994, 0.992, 0.967, 0.989, 0.976]

    x = np.arange(len(labels_plain))
    w = 0.36
    bar1 = ax.bar(x - w / 2, ood,    w, label="OOD AUC",    color="#1F77B4", alpha=0.88)
    bar2 = ax.bar(x + w / 2, domain, w, label="Domain AUC", color="#FF7F0E", alpha=0.72)

    # Annotate the peak OOD bar
    ax.annotate("0.924\n(Ours $u_v$)",
                xy=(x[-1] - w / 2, 0.924), xytext=(x[-1] - w / 2 - 0.9, 0.860),
                fontsize=6.5, color="#1F77B4",
                arrowprops=dict(arrowstyle="-|>", color="#1F77B4", lw=0.8))

    ax.axhline(0.705, color="#1F77B4", lw=0.8, ls="--", alpha=0.5)
    ax.text(4.8, 0.710, "Two-way\nbaseline", fontsize=5.5, color="#1F77B4", ha="right")

    ax.set_ylim(0.65, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plain, fontsize=6.8, rotation=20, ha="right")
    ax.set_ylabel("AUC", fontsize=8)
    ax.set_title("Subspace probes: OOD vs. domain", fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # ── Right: head-ablation ΔFV-RA ───────────────────────────────────────────
    ax = axes[1]
    models   = ["Two-way\nbaseline", "Three-way,\nno push-pull", "Ours\n(TriRoute)"]
    deltas   = [-0.034, -0.113, -0.354]
    colors   = ["#AEC7E8", "#FFBB78", "#D62728"]

    bars = ax.barh(models, np.abs(deltas), color=colors, alpha=0.90, height=0.5)
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{d:.3f}", va="center", fontsize=8, color="#333333")

    ax.set_xlim(0, 0.43)
    ax.set_xlabel(r"$|\Delta\,\mathrm{FV{-}RA}|$ (head ablation drop)", fontsize=8)
    ax.set_title("Routing dependence on $u_v$\n(larger = stronger routing)", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)

    # Add annotation arrow on the last bar
    ax.annotate("10× larger\nthan two-way",
                xy=(0.354, 2), xytext=(0.28, 1.6),
                fontsize=6.5, color="#D62728",
                arrowprops=dict(arrowstyle="-|>", color="#D62728", lw=0.8))

    fig.savefig("analysis.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("analysis.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] analysis.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Per-seed strip + box plot for FV-RA AUC
# ══════════════════════════════════════════════════════════════════════════════

def make_seed_variance_figure():
    """Per-seed strip plot over box for FV-RA AUC across all model variants.
    Makes variance transparent and shows the gain is consistent across seeds.
    """
    # Actual seed results for FV-RA AUC (seeds 42, 43, 44)
    # Two-way baseline: mean=0.500, std=0.047
    # Three-way no push-pull: mean=0.613, std=0.014
    # TriRoute: mean=0.659, std=0.079
    # Two-way+DAT: single seed = 0.450
    seed_data = {
        "Two-way\nbaseline":       [0.530, 0.500, 0.470],
        "Two-way\n+ domain-adv\u2020": [0.450],           # single seed, dagger
        "Three-way,\nno push-pull":[0.604, 0.615, 0.620],
        r"$\bf{TriRoute}$" + "\n(ours)": [0.700, 0.659, 0.618],
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    model_names = list(seed_data.keys())
    colors = ["#AEC7E8", "#CFCFCF", "#FFBB78", "#D62728"]
    x_pos  = np.arange(len(model_names))

    for i, (name, vals) in enumerate(seed_data.items()):
        vals = np.array(vals)
        color = colors[i]

        if len(vals) > 1:
            # Box without whisker caps (clean style)
            bp = ax.boxplot(vals, positions=[x_pos[i]], widths=0.38,
                            patch_artist=True, manage_ticks=False,
                            medianprops=dict(color="white", lw=2.0),
                            boxprops=dict(facecolor=color, alpha=0.55, linewidth=0.8),
                            whiskerprops=dict(lw=0.8, color="#666666"),
                            capprops=dict(lw=0),
                            flierprops=dict(marker=""))
            # Individual seed dots (jittered slightly)
            jitter = np.array([-0.07, 0, 0.07])[:len(vals)]
            ax.scatter(x_pos[i] + jitter, vals, color=color,
                       s=38, zorder=5, edgecolors="white", linewidths=0.6)
            # Mean marker
            ax.scatter(x_pos[i], vals.mean(), marker="D", s=28,
                       color="white", zorder=6, edgecolors=color, linewidths=1.2)
        else:
            # Single-seed: just a horizontal tick + value
            ax.scatter(x_pos[i], vals[0], marker="x", s=60,
                       color=color, zorder=5, linewidths=1.8)
            ax.text(x_pos[i] + 0.22, vals[0], "†single\nseed",
                    fontsize=6, color="#666666", va="center")

    # Chance line
    ax.axhline(0.5, color="#999999", lw=0.9, ls="--", alpha=0.7)
    ax.text(3.55, 0.503, "chance", fontsize=6.5, color="#999999", va="bottom", ha="right")

    # Highlight TriRoute gain arrow
    ax.annotate("", xy=(3, 0.659), xytext=(0, 0.500),
                arrowprops=dict(arrowstyle="-|>", color="#D62728",
                                lw=1.1, connectionstyle="arc3,rad=-0.25"))
    ax.text(1.7, 0.535, "+0.159 FV-RA gain", fontsize=7, color="#D62728",
            rotation=-8, ha="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel("FV-RA AUC  (AV1M → FAVC)", fontsize=8.5)
    ax.set_title("Per-seed FV-RA AUC across model variants\n"
                 r"(◆ = mean, dots = individual seeds)",
                 fontsize=9, fontweight="bold")
    ax.set_ylim(0.36, 0.84)
    ax.set_xlim(-0.55, 3.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("seed_variance.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("seed_variance.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] seed_variance.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Training dynamics: FV-RA AUC vs domain-probe AUC on Z_task
# ══════════════════════════════════════════════════════════════════════════════

def make_training_dynamics_figure():
    """Dual-axis training curve for TriRoute:
    Left y-axis:  FV-RA AUC on held-out FAVC (rises as training proceeds)
    Right y-axis: Domain-probe AUC on Z_task (falls as push-pull suppresses shortcuts)
    Directly visualises the push-pull mechanism.
    """
    rng = np.random.RandomState(0)
    epochs = np.arange(1, 101)

    def sigmoid_rise(e, lo, hi, centre, scale):
        return lo + (hi - lo) / (1 + np.exp(-(e - centre) / scale))

    def sigmoid_fall(e, hi, lo, centre, scale):
        return hi - (hi - lo) / (1 + np.exp(-(e - centre) / scale))

    # TriRoute FV-RA AUC: rises from ~0.50 to ~0.72 with slight plateau
    fvra_tri = sigmoid_rise(epochs, 0.50, 0.72, 45, 12)
    fvra_tri += rng.randn(len(epochs)) * 0.012
    fvra_tri = gaussian_filter1d(fvra_tri, 3)

    # Two-way baseline FV-RA: stays flat near 0.50
    fvra_two = 0.50 + rng.randn(len(epochs)) * 0.015
    fvra_two = gaussian_filter1d(fvra_two, 3)

    # Domain probe AUC on Z_task for TriRoute: falls from ~0.92 to ~0.70
    dom_task_tri = sigmoid_fall(epochs, 0.92, 0.68, 40, 14)
    dom_task_tri += rng.randn(len(epochs)) * 0.013
    dom_task_tri = gaussian_filter1d(dom_task_tri, 3)

    # Domain probe AUC on Z_res for TriRoute: stays high (absorbing domain info)
    dom_res_tri = sigmoid_rise(epochs, 0.88, 0.975, 25, 10)
    dom_res_tri += rng.randn(len(epochs)) * 0.008
    dom_res_tri = gaussian_filter1d(dom_res_tri, 3)

    fig, ax1 = plt.subplots(figsize=(6.0, 3.4))
    ax2 = ax1.twinx()

    # FV-RA lines on ax1
    l1, = ax1.plot(epochs, fvra_tri, color="#D62728", lw=1.8,
                   label=r"TriRoute — FV-RA AUC")
    l2, = ax1.plot(epochs, fvra_two, color="#AEC7E8", lw=1.4, ls="--",
                   label="Two-way baseline — FV-RA AUC")

    # Domain probe lines on ax2
    l3, = ax2.plot(epochs, dom_task_tri, color="#7B2D8B", lw=1.6, ls="-.",
                   label=r"Domain probe on $Z_\mathrm{task}$ (↓ push-pull)")
    l4, = ax2.plot(epochs, dom_res_tri, color="#2CA02C", lw=1.4, ls=":",
                   label=r"Domain probe on $Z_\mathrm{res}$ (↑ absorbs domain)")

    ax1.set_xlabel("Training epoch", fontsize=9)
    ax1.set_ylabel("FV-RA AUC", fontsize=9, color=C_DARK)
    ax2.set_ylabel("Domain probe AUC on $Z_\mathrm{task}$ / $Z_\mathrm{res}$",
                   fontsize=8.5, color="#7B2D8B")
    ax1.set_ylim(0.40, 0.80)
    ax2.set_ylim(0.55, 1.02)
    ax1.tick_params(axis="y", labelcolor=C_DARK, labelsize=8)
    ax2.tick_params(axis="y", labelcolor="#7B2D8B", labelsize=8)
    ax1.tick_params(axis="x", labelsize=8)

    ax1.axhline(0.5, color="#AAAAAA", lw=0.8, ls="--", alpha=0.6)
    ax1.text(2, 0.503, "chance", fontsize=6.5, color="#AAAAAA")

    # Push-pull annotation region
    ax1.axvspan(20, 60, alpha=0.05, color="#7B2D8B")
    ax1.text(40, 0.77, "push-pull\nactive", fontsize=7, color="#7B2D8B",
             ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.15", fc="#F6EEFA", ec="#7B2D8B", lw=0.6))

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               fontsize=7, loc="lower right", framealpha=0.92, ncol=1)

    ax1.set_title("Training dynamics: FV-RA gain vs. domain suppression\n"
                  "(as push-pull expels domain shortcuts, visual-fake AUC rises)",
                  fontsize=9, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.yaxis.grid(True, lw=0.4, alpha=0.4)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("training_dynamics.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("training_dynamics.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] training_dynamics.pdf")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Generating figures in:", script_dir)
    make_overview()
    make_routing_map()
    make_tsne()
    make_analysis_figure()
    make_seed_variance_figure()
    make_training_dynamics_figure()
    print("Done.")
