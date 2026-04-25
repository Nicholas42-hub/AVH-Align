"""
Generate routing_map.pdf from real model outputs.

Loads the seed-43 TriRoute checkpoint, picks one FV-RA fake clip and one
RV-RA real clip from the FAVC test split, runs a forward pass, and plots
per-frame ||u_v^t|| (visual-unique) and ||r_v^t|| (residual) norms.
"""

import os
import sys
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

sys.path.insert(0, f"{BASE}/avh_sup")
from mlp_fcd_a6 import AVH_FCD_A6

# ── load model ─────────────────────────────────────────────────────────────
print("Loading checkpoint …")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AVH_FCD_A6.load_from_checkpoint(CKPT, map_location=device)
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

# Select clips: prefer clips with >= 40 frames for a legible plot
def pick_clip(category: str, min_frames: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    subset = df[df["full_path"].str.contains(category)].copy()
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    for _, row in subset.iterrows():
        v, a = load_npz(row["full_path"])
        if v is not None and v.shape[0] >= min_frames:
            return v, a, row["full_path"]
    raise RuntimeError(f"No valid clip found for category {category}")

print("Picking FV-RA fake clip …")
v_fake, a_fake, path_fake = pick_clip("FakeVideo-RealAudio")
print(f"  {path_fake}  T={v_fake.shape[0]}")

print("Picking RV-RA real clip …")
v_real, a_real, path_real = pick_clip("RealVideo-RealAudio")
print(f"  {path_real}  T={v_real.shape[0]}")

# ── forward pass ──────────────────────────────────────────────────────────
def get_frame_norms(v: np.ndarray, a: np.ndarray):
    """Return per-frame ||u_v^t|| and ||r_v^t||."""
    vt = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,1024]
    at = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, comp = model._encode(vt, at)
    u_v = comp["u_v"].squeeze(0).cpu().numpy()   # [T, spec_dim]
    r_v = comp["r_v"].squeeze(0).cpu().numpy()   # [T, spec_dim]
    uv_norm = np.linalg.norm(u_v, ord=2, axis=-1)  # [T]
    rv_norm = np.linalg.norm(r_v, ord=2, axis=-1)  # [T]
    return uv_norm, rv_norm

print("Running forward pass …")
uv_fake, rv_fake = get_frame_norms(v_fake, a_fake)
uv_real, rv_real = get_frame_norms(v_real, a_real)

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
ax.set_title("Fake clip (FV-RA)", fontsize=9, pad=3)
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
ax.set_title("Real clip (RV-RA)", fontsize=9, pad=3)
ax.set_ylabel("Feature norm", fontsize=8)
ax.set_xlabel("Time (s)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
ax.set_xlim(0, t[-1])

fig.savefig(OUT_PDF, bbox_inches="tight", dpi=150)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")

# Print basic stats for sanity check
print(f"\nFake clip: u_v mean={uv_fake.mean():.3f} max={uv_fake.max():.3f}  "
      f"r_v mean={rv_fake.mean():.3f} max={rv_fake.max():.3f}")
print(f"Real clip: u_v mean={uv_real.mean():.3f} max={uv_real.max():.3f}  "
      f"r_v mean={rv_real.mean():.3f} max={rv_real.max():.3f}")
