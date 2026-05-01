"""
Aggregate baseline 50ep eval_results.txt across seeds and plot error bars.

Auto-detects available seeds; once seed 45 finishes it will be picked up.
"""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/data/gpfs/projects/punim2637/nnliang/AVH-Align/avh_sup")
OUT  = Path("/data/gpfs/projects/punim2637/nnliang/AVH-Align/figures")

# (display_name, dir_template, results_subdir)
MODEL_SPECS = [
    ("A2",            "outputs_A2_50ep_seed{seed}",                       "results_favc7030_full"),
    ("A5",            "outputs_A5_50ep_seed{seed}",                       "results_favc7030_full"),
    ("DANN",          "outputs_DANN_50ep_seed{seed}",                     "results_favc7030_full"),
    ("CORAL",         "outputs_CORAL_50ep_seed{seed}",                    "results_favc7030_full"),
    ("TriRoute",      "outputs_A6_av1m_pseudodomain_v2_seed{seed}",       "results_favc_7030_full"),
    ("TriRoute+UDA",  "outputs_A6_seed{seed}",                            "results_favc_7030_full"),
]
MODELS = [m[0] for m in MODEL_SPECS]
SEEDS  = [43, 44, 45]
SUBSETS = ["overall", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]

def parse_full_head(path: Path) -> dict[str, float]:
    """Return {subset: auc} from the FULL HEAD section."""
    text = path.read_text()
    m = re.search(r"── FULL HEAD ──(.+?)──", text, flags=re.S)
    block = m.group(1) if m else text
    out = {}
    for line in block.splitlines():
        for s in SUBSETS:
            mm = re.search(rf"\b{re.escape(s)}\b\s+AUC=([0-9.]+)", line)
            if mm:
                out[s] = float(mm.group(1))
    return out

# Collect: results[model][subset] = list of AUCs across seeds
results = {m: {s: [] for s in SUBSETS} for m in MODELS}
seeds_used = {m: [] for m in MODELS}

for name, dir_tpl, results_sub in MODEL_SPECS:
    for sd in SEEDS:
        f = ROOT / dir_tpl.format(seed=sd) / results_sub / "eval_results.txt"
        if not f.exists():
            continue
        d = parse_full_head(f)
        for s in SUBSETS:
            if s in d:
                results[name][s].append(d[s])
        seeds_used[name].append(sd)

# Print summary
print(f"{'Model':<6} {'seeds':<14} " + " ".join(f"{s[:14]:<16}" for s in SUBSETS))
for m in MODELS:
    seeds_str = ",".join(str(x) for x in seeds_used[m])
    cells = []
    for s in SUBSETS:
        v = np.array(results[m][s])
        if len(v) == 0:
            cells.append("--")
        else:
            cells.append(f"{v.mean():.4f}±{v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}±n/a")
    print(f"{m:<6} {seeds_str:<14} " + " ".join(f"{c:<16}" for c in cells))

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
colors = {
    "A2": "#7f7f7f", "A5": "#1f77b4", "DANN": "#2ca02c", "CORAL": "#d62728",
    "TriRoute": "#ff7f0e", "TriRoute+UDA": "#9467bd",
}

panels = [
    ("overall",             "FAVC overall (N=6667)"),
    ("FakeVideo-RealAudio", "FV-RA — hardest split (N=3158)"),
]

for ax, (subset, title) in zip(axes, panels):
    means, stds, labels = [], [], []
    for m in MODELS:
        v = np.array(results[m][subset])
        if len(v) == 0:
            continue
        means.append(v.mean())
        stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        labels.append(m)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[colors[l] for l in labels],
                  edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 1.2, "ecolor": "black"})
    for xi, m, sd in zip(x, means, stds):
        ax.text(xi, m + sd + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0.30, 1.0 if subset == "overall" else 0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

n_seeds_min = min(len(seeds_used[m]) for m in MODELS)
n_seeds_max = max(len(seeds_used[m]) for m in MODELS)
seed_note = (f"n={n_seeds_min} seeds" if n_seeds_min == n_seeds_max
             else f"n={n_seeds_min}–{n_seeds_max} seeds")
fig.suptitle(f"Baseline 50ep — FAVC cross-dataset AUC ({seed_note}, mean ± std)",
             fontsize=12, y=1.02)
fig.tight_layout()

png = OUT / "fig_baseline_errorbars.png"
pdf = OUT / "fig_baseline_errorbars.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"\nSaved: {png}")
print(f"Saved: {pdf}")
