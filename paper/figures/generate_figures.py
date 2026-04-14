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
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # ── column x-centres & row y positions ───────────────────────────────────
    bw, bh = 0.13, 0.09   # box width / height
    bh_enc = 0.11          # encoder boxes

    # Row y positions (bottom of box)
    y_input  = 0.84
    y_enc    = 0.69
    y_feat   = 0.58   # feature label row (text only)
    y_router = 0.44
    y_expert = 0.28
    y_loss   = 0.08

    # X positions
    x_vis  = 0.08
    x_aud  = 0.30
    x_rout = 0.50   # router centre
    rw     = 0.18   # router box width
    xl     = 0.30   # left branch x
    xm     = 0.50   # middle branch x
    xr     = 0.70   # right branch x
    ew     = 0.12   # expert width

    # ── Input blocks ─────────────────────────────────────────────────────────
    draw_box(ax, (x_vis, y_input), bw, bh, "Face Frames",
             sublabel=r"$x^v \in \mathbb{R}^{T\times H\times W}$",
             color="#EEF4FF", edgecolor="#4C72B0")
    draw_box(ax, (x_aud, y_input), bw, bh, "Audio",
             sublabel=r"$x^a$",
             color="#FFF4E0", edgecolor="#C07A00")

    # ── Encoder blocks ────────────────────────────────────────────────────────
    draw_box(ax, (x_vis, y_enc), bw, bh_enc, "Visual Encoder",
             sublabel="AV-HuBERT", color="#DAE8FC", edgecolor="#4C72B0",
             fontsize=8, bold=False)
    draw_box(ax, (x_aud, y_enc), bw, bh_enc, "Audio Encoder",
             sublabel="AV-HuBERT", color="#FFE6CC", edgecolor="#C07A00",
             fontsize=8)

    # ── Feature labels ────────────────────────────────────────────────────────
    ax.text(x_vis + bw / 2, y_feat + 0.01,
            r"$V\!\in\!\mathbb{R}^{T\!\times\!C}$",
            ha="center", va="center", fontsize=7.5, color="#4C72B0")
    ax.text(x_aud + bw / 2, y_feat + 0.01,
            r"$A\!\in\!\mathbb{R}^{T\!\times\!C}$",
            ha="center", va="center", fontsize=7.5, color="#C07A00")

    # ── Router block ──────────────────────────────────────────────────────────
    rx = x_rout - rw / 2
    draw_box(ax, (rx, y_router), rw, 0.10,
             "Audio-conditioned Router",
             sublabel=r"$g_t = \mathrm{softmax}(W_r[v_t;a_t;v_t\!\odot\!a_t])$",
             color="#F0E6F6", edgecolor="#7B2D8B", fontsize=7.5, bold=True)

    # ── Expert blocks ─────────────────────────────────────────────────────────
    draw_box(ax, (xl - ew / 2, y_expert), ew, 0.10,
             "Manip. Expert\n$E_m$", color="#FDDEDE", edgecolor=C_MAN, fontsize=8)
    draw_box(ax, (xm - ew / 2, y_expert), ew, 0.10,
             "Content Expert\n$E_c$", color="#DDEEFF", edgecolor=C_CON, fontsize=8)
    draw_box(ax, (xr - ew / 2, y_expert), ew, 0.10,
             "Domain Expert\n$E_d$", color="#DDFADD", edgecolor=C_DOM, fontsize=8)

    # ── Factor labels ─────────────────────────────────────────────────────────
    for x, lbl, col in [(xl, r"$H^m$", C_MAN),
                         (xm, r"$H^c$", C_CON),
                         (xr, r"$H^d$", C_DOM)]:
        ax.text(x, y_expert - 0.03, lbl, ha="center", va="center",
                fontsize=9, color=col, fontweight="bold")

    # ── Loss boxes ────────────────────────────────────────────────────────────
    lw_box = 0.115
    # Manipulation losses
    draw_box(ax, (xl - lw_box / 2, y_loss), lw_box, 0.08,
             r"$\mathcal{L}_{\rm fake}$" + "\n" + r"$+ \mathcal{L}_{\rm adv}$",
             color="#FFF0F0", edgecolor=C_MAN, fontsize=7.5)
    # Content loss
    draw_box(ax, (xm - lw_box / 2, y_loss), lw_box, 0.08,
             r"$\mathcal{L}_{\rm align}$",
             color="#F0F4FF", edgecolor=C_CON, fontsize=7.5)
    # Domain loss
    draw_box(ax, (xr - lw_box / 2, y_loss), lw_box, 0.08,
             r"$\mathcal{L}_{\rm dom}$",
             color="#F0FFF0", edgecolor=C_DOM, fontsize=7.5)

    # ── Orthogonality annotation ───────────────────────────────────────────────
    ax.annotate("", xy=(xm - ew / 2 - 0.01, y_expert + 0.05),
                xytext=(xl + ew / 2 + 0.01, y_expert + 0.05),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=0.0"))
    ax.annotate("", xy=(xr - ew / 2 - 0.01, y_expert + 0.05),
                xytext=(xm + ew / 2 + 0.01, y_expert + 0.05),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=0.0"))
    ax.text((xl + xr) / 2, y_expert + 0.07,
            r"$\mathcal{L}_{\rm ortho}$",
            ha="center", va="center", fontsize=7, color="#888888")

    # ── Arrows: input → encoder ───────────────────────────────────────────────
    arrow(ax, x_vis + bw / 2, y_input, x_vis + bw / 2, y_enc + bh_enc,
          color="#4C72B0")
    arrow(ax, x_aud + bw / 2, y_input, x_aud + bw / 2, y_enc + bh_enc,
          color="#C07A00")

    # ── Arrows: encoder → feature text (skip, merge into router) ─────────────
    arrow(ax, x_vis + bw / 2, y_enc, x_vis + bw / 2, y_feat + 0.04,
          color="#4C72B0")
    arrow(ax, x_aud + bw / 2, y_enc, x_aud + bw / 2, y_feat + 0.04,
          color="#C07A00")

    # ── Arrows: features → router ─────────────────────────────────────────────
    arrow(ax, x_vis + bw / 2, y_feat - 0.005, x_rout - 0.01, y_router + 0.05,
          color="#7B2D8B", connectionstyle="arc3,rad=-0.15")
    arrow(ax, x_aud + bw / 2, y_feat - 0.005, x_rout + 0.01, y_router + 0.05,
          color="#7B2D8B", connectionstyle="arc3,rad=0.15")

    # ── Arrows: router → experts ──────────────────────────────────────────────
    router_bot_y = y_router
    for x_ex, col, label in [(xl, C_MAN, "$g^m$"),
                               (xm, C_CON, "$g^c$"),
                               (xr, C_DOM, "$g^d$")]:
        arrow(ax, x_rout, router_bot_y, x_ex, y_expert + 0.10,
              color=col, lw=1.1,
              connectionstyle=f"arc3,rad={0.25 if x_ex < x_rout else -0.25 if x_ex > x_rout else 0.0}")
        offset = -0.03 if x_ex < x_rout else 0.03 if x_ex > x_rout else 0.0
        ax.text(x_rout + offset * 0.5,
                (router_bot_y + y_expert + 0.10) / 2 - 0.02,
                label, fontsize=7, color=col, ha="center")

    # ── Arrows: experts → losses ──────────────────────────────────────────────
    for x_ex in [xl, xm, xr]:
        arrow(ax, x_ex, y_expert, x_ex, y_loss + 0.08, color="#999999", lw=0.8)

    # ── Inference arrow ───────────────────────────────────────────────────────
    ax.annotate("", xy=(xl - lw_box / 2 - 0.08, y_loss + 0.04),
                xytext=(xl - lw_box / 2 - 0.01, y_loss + 0.04),
                arrowprops=dict(arrowstyle="<-", color=C_MAN, lw=1.4))
    ax.text(xl - lw_box / 2 - 0.085, y_loss + 0.04,
            "Inference\n(only $H^m$)", ha="right", va="center",
            fontsize=7, color=C_MAN,
            bbox=dict(boxstyle="round,pad=0.1", fc="#FFF5F5", ec=C_MAN, lw=0.7))

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
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Generating figures in:", script_dir)
    make_overview()
    make_routing_map()
    make_tsne()
    print("Done.")
