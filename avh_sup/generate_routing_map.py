"""
Generate routing_map.pdf from real model outputs.

Loads the seed-43 TriRoute checkpoint, picks one FV-RA fake clip and one
RealVideo-RealAudio real clip from the FAVC test split, runs a forward pass,
and plots per-frame ||u_v^t|| (visual-unique) and ||r_v^t|| (residual) norms.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── paths ──────────────────────────────────────────────────────────────────
BASE         = "/data/projects/punim2637/nnliang/AVH-Align"
CKPT         = f"{BASE}/avh_sup/outputs_A6_trainvalreal_seed43/ckpts/model-epoch=32.ckpt"
FEAT_ROOT    = f"{BASE}/data/favc_features"
CSV_PATH     = f"{BASE}/avh_sup/csv_metadata/favc_npz_matched/test_split.csv"
OUT_PDF      = f"{BASE}/paper/figures/routing_map.pdf"
OUT_PNG      = f"{BASE}/paper/figures/routing_map.png"
OUT_META     = f"{BASE}/paper/figures/routing_map_metadata.json"

sys.path.insert(0, f"{BASE}/avh_sup")
from mlp_fcd_a6_lite import AVH_FCD_A6_Lite

# ── load model ─────────────────────────────────────────────────────────────
print("Loading checkpoint …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AVH_FCD_A6_Lite.load_from_checkpoint(CKPT, map_location=device)
model.eval()
print(f"  device: {device}")

# ── pick clips from test split ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

def full_path_to_npz(full_path: str) -> str:
    rel = full_path.replace("FakeAVCeleb/", "").replace(".mp4", ".npz")
    return os.path.join(FEAT_ROOT, rel)

def load_npz(full_path: str, apply_l2: bool = True):
    path = full_path_to_npz(full_path)
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    v = d["visual"].astype(np.float32)
    a = d["audio"].astype(np.float32)
    if apply_l2:
        v = v / (np.linalg.norm(v, ord=2, axis=-1, keepdims=True) + 1e-8)
        a = a / (np.linalg.norm(a, ord=2, axis=-1, keepdims=True) + 1e-8)
    return v, a

# ── forward pass ──────────────────────────────────────────────────────────
def get_model_outputs(v: np.ndarray, a: np.ndarray):
    """Return clip score plus per-frame ||u_v^t|| and ||r_v^t||."""
    vt = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,1024]
    at = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        score = model.predict_scores(vt, at, mode="causal").item()
        _, _, comp = model._encode(vt, at)
    u_v = comp["u_v"].squeeze(0).cpu().numpy()   # [T, spec_dim]
    r_v = comp["r_v"].squeeze(0).cpu().numpy()   # [T, spec_dim]
    uv_norm = np.linalg.norm(u_v, ord=2, axis=-1)  # [T]
    rv_norm = np.linalg.norm(r_v, ord=2, axis=-1)  # [T]
    return score, uv_norm, rv_norm

def peakiness(x: np.ndarray) -> float:
    return float((x.max() - np.median(x)) / (x.std() + 1e-8))

def collect_candidates(category: str, min_frames: int = 40):
    subset = df[df["full_path"].str.contains(category)].copy().reset_index(drop=True)
    rows = []
    for i, row in subset.iterrows():
        v, a = load_npz(row["full_path"])
        if v is None or v.shape[0] < min_frames:
            continue
        score, uv, rv = get_model_outputs(v, a)
        rows.append({
            "path": row["full_path"],
            "v": v,
            "a": a,
            "score": score,
            "uv": uv,
            "rv": rv,
            "uv_mean": float(uv.mean()),
            "uv_max": float(uv.max()),
            "uv_peakiness": peakiness(uv),
            "rv_mean": float(rv.mean()),
            "rv_max": float(rv.max()),
            "rv_peakiness": peakiness(rv),
        })
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(subset)} {category} clips", flush=True)
    if not rows:
        raise RuntimeError(f"No valid clip found for category {category}")
    return rows

print("Scanning held-out FV-RA fake clips …")
fake_candidates = collect_candidates("FakeVideo-RealAudio")
fake_tp = [r for r in fake_candidates if r["score"] > 0]
if not fake_tp:
    raise RuntimeError("No correctly classified FV-RA fake clips found")
fake_sel = max(fake_tp, key=lambda r: r["score"] + 0.25 * r["uv_peakiness"])

print("Scanning held-out real clips …")
real_candidates = collect_candidates("RealVideo-RealAudio")
real_tn = [r for r in real_candidates if r["score"] < 0]
if not real_tn:
    raise RuntimeError("No correctly classified RealVideo-RealAudio clips found")
real_sel = min(real_tn, key=lambda r: r["score"] + 0.25 * r["uv_peakiness"])

v_fake, a_fake, path_fake = fake_sel["v"], fake_sel["a"], fake_sel["path"]
v_real, a_real, path_real = real_sel["v"], real_sel["a"], real_sel["path"]
uv_fake, rv_fake = fake_sel["uv"], fake_sel["rv"]
uv_real, rv_real = real_sel["uv"], real_sel["rv"]

print("Selected correctly classified clips:")
print(f"  fake: score={fake_sel['score']:.3f} peak={fake_sel['uv_peakiness']:.3f} "
      f"T={v_fake.shape[0]}  {path_fake}")
print(f"  real: score={real_sel['score']:.3f} peak={real_sel['uv_peakiness']:.3f} "
      f"T={v_real.shape[0]}  {path_real}")

metadata = {
    "checkpoint": CKPT,
    "csv_path": CSV_PATH,
    "feature_root": FEAT_ROOT,
    "selection_rule": (
        "fake: correctly classified FV-RA clip maximizing score + 0.25*u_v_peakiness; "
        "real: correctly classified RealVideo-RealAudio clip minimizing score + 0.25*u_v_peakiness"
    ),
    "fake": {k: v for k, v in fake_sel.items() if k not in {"v", "a", "uv", "rv"}},
    "real": {k: v for k, v in real_sel.items() if k not in {"v", "a", "uv", "rv"}},
}

# Smooth lightly for readability (sigma=1 frame, ~40ms)
uv_fake_s = gaussian_filter1d(uv_fake, sigma=1.0)
rv_fake_s = gaussian_filter1d(rv_fake, sigma=1.0)
uv_real_s = gaussian_filter1d(uv_real, sigma=1.0)
rv_real_s = gaussian_filter1d(rv_real, sigma=1.0)

# ── plot ──────────────────────────────────────────────────────────────────
C_UV = "#1f77b4"   # blue  – visual-unique
C_RV = "#ff7f0e"   # orange – residual

def time_axis(n: int) -> np.ndarray:
    return np.arange(n) / 25.0   # 25 fps → seconds

fig, axes = plt.subplots(2, 1, figsize=(8, 3.6), sharex=False)
fig.subplots_adjust(hspace=0.55)

# ── top: fake clip ──
ax = axes[0]
t = time_axis(len(uv_fake_s))
ax.plot(t, uv_fake_s, color=C_UV, lw=1.5, label=r"$\|u_v^t\|$ (visual-unique)")
ax.plot(t, rv_fake_s, color=C_RV, lw=1.5, linestyle="--", label=r"$\|r_v^t\|$ (residual)")
ax.axvline(t[int(np.argmax(uv_fake_s))], color=C_UV, lw=0.8, alpha=0.35)
ax.set_title(f"Correctly classified fake clip (FV-RA), score={fake_sel['score']:.2f}", fontsize=9, pad=3)
ax.set_ylabel("Feature norm", fontsize=8)
ax.set_xlabel("Time (s)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
ax.set_xlim(0, t[-1])

# ── bottom: real clip ──
ax = axes[1]
t = time_axis(len(uv_real_s))
ax.plot(t, uv_real_s, color=C_UV, lw=1.5, label=r"$\|u_v^t\|$ (visual-unique)")
ax.plot(t, rv_real_s, color=C_RV, lw=1.5, linestyle="--", label=r"$\|r_v^t\|$ (residual)")
ax.axvline(t[int(np.argmax(uv_real_s))], color=C_UV, lw=0.8, alpha=0.25)
ax.set_title(f"Correctly classified real clip, score={real_sel['score']:.2f}", fontsize=9, pad=3)
ax.set_ylabel("Feature norm", fontsize=8)
ax.set_xlabel("Time (s)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
ax.set_xlim(0, t[-1])

fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_META}")

# Print basic stats for sanity check
print(f"\nFake clip: u_v mean={uv_fake.mean():.3f} max={uv_fake.max():.3f}  "
      f"r_v mean={rv_fake.mean():.3f} max={rv_fake.max():.3f}")
print(f"Real clip: u_v mean={uv_real.mean():.3f} max={uv_real.max():.3f}  "
      f"r_v mean={rv_real.mean():.3f} max={rv_real.max():.3f}")
