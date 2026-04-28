"""
Update NIc_NIPS_eval_list.docx with latest pdv2_50ep numbers and create all Caren-requested plots.
"""

import re
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

# ─────────────────────────────────────────────────────────────
# 1. UPDATED NUMBERS (pdv2_50ep seeds 44/45 mean)
# ─────────────────────────────────────────────────────────────

# Table 1 – main benchmark (AUC/AP)
# TriRoute no target data row
NEW_T1_TRIROUTE = {
    "Overall":  "0.922/0.998",
    "RV-FA":    "0.999/0.999",
    "FV-RA":    "0.838/0.989",
    "FV-FA":    "0.994/1.000",
}

# Table 1 – baseline rows updated to 50ep seed44
# Row-label keyword → {old_cell_text: new_cell_text}
NEW_T1_BASELINES = {
    # Two-way baseline (A2 50ep s44)
    "Two-way baseline": {
        "0.779/0.994": "0.739/0.993",
        "0.779/0.993": "0.739/0.993",
        "0.523/0.957": "0.437/0.949",
        "0.999/1.000": "0.998/1.000",  # FV-FA only; RV-FA stays 1.000/1.000
    },
    # DANN / GRL
    "DANN": {
        "0.762/0.993": "0.767/0.994",
        "0.478/0.952": "0.496/0.955",
    },
    # CORAL
    "CORAL": {
        "0.753/0.993": "0.769/0.994",
        "0.491/0.947": "0.521/0.956",
        "0.971/0.999": "0.980/0.999",
    },
    # Three-way, no push-pull (A5 50ep s44)
    "Three-way": {
        "0.827/0.996": "0.766/0.994",
        "0.626/0.973": "0.496/0.955",
        "0.999/1.000": "0.998/1.000",  # FV-FA only
    },
}

# Table 5 – leading silence (per-category AUC)
NEW_T5 = {
    "Untrimmed": {"Overall": "0.9427", "RV-FA": "1.0000", "FV-RA": "0.8779", "FV-FA": "0.9969"},
    "Trimmed":   {"Overall": "0.9045", "RV-FA": "0.7759", "FV-RA": "0.8767", "FV-FA": "0.9348"},
    "Delta":     {"Overall": "-0.0382","RV-FA": "-0.2241","FV-RA": "-0.0012","FV-FA": "-0.0622"},
}

# Table 6 – AVLips
NEW_T6_TRIROUTE = "0.789"

# ─────────────────────────────────────────────────────────────
# 2. UPDATE DOCX
# ─────────────────────────────────────────────────────────────

doc = Document("NIc_NIPS_eval_list.docx")

def set_cell_text(cell, text, bold=False, color=None):
    """Replace cell text preserving paragraph formatting."""
    para = cell.paragraphs[0]
    for run in para.runs:
        run.text = ""
    if para.runs:
        run = para.runs[0]
    else:
        run = para.add_run()
    run.text = text
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def find_tables_with_keyword(doc, keyword):
    results = []
    for i, table in enumerate(doc.tables):
        for row in table.rows:
            for cell in row.cells:
                if keyword.lower() in cell.text.lower():
                    results.append(i)
                    break
    return list(set(results))

# --- Table 1: fix double-slash and update AP for FV-RA ---
for table in doc.tables:
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # Identify TriRoute no-target row by checking FV-RA content with old AP
        if any("TRIROUTE" in c.upper() or "TriRoute" in c for c in cells):
            if any("0.838" in c for c in cells):
                for j, cell in enumerate(row.cells):
                    txt = cell.text.strip()
                    # Fix double slash in Overall
                    if "0.922//0.998" in txt:
                        set_cell_text(cell, "0.922/0.998")
                    # Fix outdated FV-RA AP (0.979 → 0.989)
                    if "0.838/0.979" in txt:
                        set_cell_text(cell, "0.838/0.989")

# --- Table 1: update baseline rows to 50ep seed44 ---
for table in doc.tables:
    flat = " ".join(c.text for row in table.rows for c in row.cells)
    # Only process tables that look like Table 1 (main benchmark)
    if not any(k in flat for k in ["Two-way", "DANN", "CORAL", "Three-way"]):
        continue
    if "AVLips" in flat or "Trimmed" in flat:  # skip T5/T6
        continue
    for row in table.rows:
        row_text = " ".join(c.text.strip() for c in row.cells)
        matched_key = None
        for key in NEW_T1_BASELINES:
            if key.lower() in row_text.lower():
                matched_key = key
                break
        if matched_key is None:
            continue
        mapping = NEW_T1_BASELINES[matched_key]
        for cell in row.cells:
            txt = cell.text.strip()
            if txt in mapping:
                new_val = mapping[txt]
                set_cell_text(cell, new_val)
                print(f"  Updated T1 [{matched_key}]: '{txt}' → '{new_val}'")

# --- Table 5: update TriRoute silence rows ---
silence_update_map = {
    # old text fragment → new text
    "0.87":    ("Overall Untrimmed", "0.9427"),
    "0.731":   ("FV-RA Untrimmed",   "0.8779"),
    "0.9973":  ("FV-FA Untrimmed",   "0.9969"),
    "0.79":    ("Overall Trimmed",   "0.9045"),
    "0.9034":  ("RV-FA Trimmed",     "0.7759"),
    "0.73":    ("FV-RA Trimmed",     "0.8767"),
    "0.8341":  ("FV-FA Trimmed",     "0.9348"),
    "-0.08":   ("Overall Delta",     "-0.0382"),
    "-0.0966": ("RV-FA Delta",       "-0.2241"),
    # FV-RA delta was 0.00 → -0.0012
    "0.00":    ("FV-RA Delta",       "-0.0012"),
    "-0.1632": ("FV-FA Delta",       "-0.0622"),
}

for table in doc.tables:
    # Check if this is Table 5 by looking for "TriRoute" and "Trimmed"
    flat = " ".join(c.text for row in table.rows for c in row.cells)
    if "Trimmed" not in flat or "TriRoute" not in flat:
        continue
    for row in table.rows:
        cells_text = [c.text.strip() for c in row.cells]
        row_text = " ".join(cells_text)
        # Only process TriRoute rows
        if "TriRoute" not in row_text and "TRIROUTE" not in row_text.upper():
            continue
        for j, cell in enumerate(row.cells):
            txt = cell.text.strip()
            for old_frag, (desc, new_val) in silence_update_map.items():
                if txt == old_frag or (old_frag in txt and len(txt) < len(old_frag) + 5):
                    set_cell_text(cell, new_val)
                    print(f"  Updated T5 cell: '{txt}' → '{new_val}' ({desc})")
                    break

# --- Table 6: AVLips TriRoute 0.801 → 0.789 ---
for table in doc.tables:
    flat = " ".join(c.text for row in table.rows for c in row.cells)
    if "AVLips" not in flat and "avlips" not in flat.lower() and "0.801" not in flat:
        continue
    for row in table.rows:
        cells_text = [c.text.strip() for c in row.cells]
        row_text = " ".join(cells_text)
        if "TriRoute" not in row_text and "TRIROUTE" not in row_text.upper():
            continue
        for j, cell in enumerate(row.cells):
            if cell.text.strip() == "0.801":
                set_cell_text(cell, "0.789")
                print(f"  Updated T6 AVLips: 0.801 → 0.789")

doc.save("NIc_NIPS_eval_list_updated.docx")
print("Saved: NIc_NIPS_eval_list_updated.docx")

# ─────────────────────────────────────────────────────────────
# 3. PLOTS
# ─────────────────────────────────────────────────────────────

COLORS = {
    "two_way":    "#7bafd4",  # blue
    "three_way":  "#f5a623",  # orange
    "triroute":   "#2a9d8f",  # teal/green
}
FONTSIZE = 12
plt.rcParams.update({"font.size": FONTSIZE, "font.family": "sans-serif"})

# ── Plot A: ΔFV-RA bar chart (head ablation) ─────────────────
fig, ax = plt.subplots(figsize=(6, 4))
models   = ["Two-way\nbaseline", "Three-way\nno push-pull", "TriRoute\n(ours)"]
deltas   = [-0.034, -0.113, -0.345]
colors   = [COLORS["two_way"], COLORS["three_way"], COLORS["triroute"]]
bars = ax.bar(models, [-d for d in deltas], color=colors, edgecolor="white", width=0.5)
for bar, d in zip(bars, deltas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{d:+.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("FV-RA AUC drop when visual branches masked", fontsize=11)
ax.set_title("Head Ablation: Routing Dependence on Visual Branches\n"
             "(Larger drop = model actively uses visual pathway)", fontsize=11)
ax.set_ylim(0, 0.42)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_head_ablation_delta.png", dpi=200, bbox_inches="tight")
print("Saved: fig_head_ablation_delta.png")
plt.close()

# ── Plot B: Progressive masking curves ───────────────────────
p_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# Visual masking data (causal_auc)
vis_a2       = [0.5462, 0.5447, 0.5275, 0.5141, 0.5219, 0.4694]
vis_a5       = [0.6156, 0.6066, 0.5887, 0.5686, 0.5700, 0.4790]
vis_triroute = [0.8378, 0.8245, 0.7972, 0.7495, 0.6730, 0.4612]

# Audio masking data
aud_a2       = [0.5462, 0.5618, 0.6314, 0.5978, 0.5685, 0.5253]
aud_a5       = [0.6156, 0.6084, 0.5931, 0.6005, 0.6580, 0.6559]
aud_triroute = [0.8378, 0.8376, 0.8366, 0.8359, 0.8388, 0.8430]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

# Visual masking
ax = axes[0]
ax.plot(p_levels, vis_a2,       "o-", color=COLORS["two_way"],   lw=2, label="Two-way baseline",       markersize=6)
ax.plot(p_levels, vis_a5,       "s-", color=COLORS["three_way"], lw=2, label="Three-way, no push-pull", markersize=6)
ax.plot(p_levels, vis_triroute, "D-", color=COLORS["triroute"],  lw=2.5, label="TriRoute (ours)",       markersize=7)
ax.set_xlabel("Visual masking fraction $p$", fontsize=11)
ax.set_ylabel("FV-RA AUC", fontsize=11)
ax.set_title("Visual Masking Sweep\n(drop ↓ = visual-dependent)", fontsize=11)
ax.set_ylim(0.40, 0.90)
ax.set_xticks(p_levels)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Audio masking
ax = axes[1]
ax.plot(p_levels, aud_a2,       "o-", color=COLORS["two_way"],   lw=2, label="Two-way baseline",       markersize=6)
ax.plot(p_levels, aud_a5,       "s-", color=COLORS["three_way"], lw=2, label="Three-way, no push-pull", markersize=6)
ax.plot(p_levels, aud_triroute, "D-", color=COLORS["triroute"],  lw=2.5, label="TriRoute (ours)",       markersize=7)
ax.set_xlabel("Audio masking fraction $p$", fontsize=11)
ax.set_ylabel("FV-RA AUC", fontsize=11)
ax.set_title("Audio Masking Sweep\n(rise ↑ = audio was hurting, visual routing active)", fontsize=11)
ax.set_ylim(0.40, 0.90)
ax.set_xticks(p_levels)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.suptitle("Routing Sensitivity Analysis (FV-RA AUC under progressive input masking)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("fig_masking_curves.png", dpi=200, bbox_inches="tight")
print("Saved: fig_masking_curves.png")
plt.close()

# ── Plot C: Subspace probe bar plot ──────────────────────────
subspaces = [
    ("Two-way\ntask branch $Z_c$\n(1024d)", 0.705, 0.998),
    ("Two-way\nvisual factor $v_c$\n(512d)", 0.891, 0.994),
    ("Three-way\nshared visual $s_v$\n(256d)", 0.814, 0.992),
    ("Three-way\nunique visual $u_v$\n(256d)", 0.825, 0.967),
    ("TriRoute\nshared visual $s_v$\n(256d)", 0.726, 0.989),
    ("TriRoute\nunique visual $u_v$\n(256d)", 0.924, 0.976),
]
labels   = [s[0] for s in subspaces]
ood_aucs = [s[1] for s in subspaces]
dom_aucs = [s[2] for s in subspaces]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width/2, ood_aucs, width, label="OOD probe AUC (FV-RA)", color="#2a9d8f", edgecolor="white")
b2 = ax.bar(x + width/2, dom_aucs, width, label="Domain probe AUC", color="#e76f51", edgecolor="white", alpha=0.85)

for bar, v in zip(b1, ood_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for bar, v in zip(b2, dom_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("AUC", fontsize=11)
ax.set_ylim(0.65, 1.01)
ax.set_title("Subspace Probe: OOD vs Domain AUC\n"
             "(High domain AUC confirms routing, not invariance)", fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Highlight TriRoute u_v as key result
ax.annotate("Highest OOD\ndespite 256d",
            xy=(x[-1] - width/2, 0.924), xytext=(x[-1] - width/2 - 0.6, 0.95),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, color="black")
plt.tight_layout()
plt.savefig("fig_subspace_probe.png", dpi=200, bbox_inches="tight")
print("Saved: fig_subspace_probe.png")
plt.close()

# ── Plot D: Main table FV-RA grouped bar (with error bars) ───
methods = [
    "AVH-Align/sup\n[Smeu et al.]",
    "Two-way\nbaseline",
    "Two-way\n+DANN/GRL",
    "Two-way\n+CORAL",
    "Three-way\nno push-pull",
    "TriRoute\n(ours)",
]
fvra_aucs = [0.519, 0.437, 0.496, 0.521, 0.496, 0.838]
# Seed-level std for TriRoute (seeds 44/45): (0.8272, 0.8484)
stds = [0.0, 0.0, 0.0, 0.0, 0.0, np.std([0.8272, 0.8484])]

bar_colors = [
    "#adb5bd",  # grey for external baseline
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["two_way"],
    COLORS["three_way"],
    COLORS["triroute"],
]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(methods, fvra_aucs, color=bar_colors, edgecolor="white", width=0.6,
              yerr=stds, capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "black"})
ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance (AUC=0.5)")
for bar, v in zip(bars, fvra_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("FV-RA AUC (cross-domain: AV1M → FAVC)", fontsize=11)
ax.set_title("Cross-Domain FakeVideo-RealAudio Detection\n"
             "(FV-RA requires visual evidence; higher = more visual reliance)", fontsize=11)
ax.set_ylim(0.40, 0.92)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Legend patches
legend_handles = [
    mpatches.Patch(color="#adb5bd", label="External baseline"),
    mpatches.Patch(color=COLORS["two_way"], label="Two-way models"),
    mpatches.Patch(color=COLORS["three_way"], label="Three-way, no push-pull"),
    mpatches.Patch(color=COLORS["triroute"], label="TriRoute (ours)"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("fig_main_fvra.png", dpi=200, bbox_inches="tight")
print("Saved: fig_main_fvra.png")
plt.close()

print("\nAll done.")
