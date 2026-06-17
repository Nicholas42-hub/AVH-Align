"""
SWA (Stochastic Weight Averaging) for seed 44 — post-saturation pre-drift window.

INPUT (read-only):
  outputs_A6_av1m_pseudodomain_v2_50ep_seed44/ckpts/favc_fvra/best-favc-fvra-epoch={40,41,42,46,50}-auc=*.ckpt

OUTPUT (new directory, never touches existing files):
  outputs_A6_av1m_pseudodomain_v2_50ep_seed44_SWA/ckpts/swa_e40_50.ckpt

These five ckpts are all from the same in-trajectory window of the pdv2_s44_to100 run,
where source val_auc has already saturated (~0.99) and target FV-RA has not yet drifted.
"""

from __future__ import annotations

import argparse
import glob
from collections import OrderedDict
from pathlib import Path

import torch


SRC_CKPT_DIR = Path(
    "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/"
    "outputs_A6_av1m_pseudodomain_v2_50ep_seed44/ckpts/favc_fvra"
)
OUT_DIR = Path(
    "/data/projects/punim2637/nnliang/AVH-Align/avh_sup/"
    "outputs_A6_av1m_pseudodomain_v2_50ep_seed44_SWA/ckpts"
)
EPOCHS = [40, 41, 42, 46, 50]


def find_ckpt(epoch: int) -> Path:
    matches = sorted(SRC_CKPT_DIR.glob(f"best-favc-fvra-epoch={epoch:02d}-auc=*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"epoch {epoch}: expected 1 match, got {len(matches)}: {matches}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_name", default="swa_e40_50.ckpt")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [find_ckpt(e) for e in EPOCHS]
    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p.name}")

    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    ref = ckpts[0]["state_dict"]
    keys = list(ref.keys())
    for c in ckpts[1:]:
        if set(c["state_dict"].keys()) != set(keys):
            extra_a = set(keys) - set(c["state_dict"].keys())
            extra_b = set(c["state_dict"].keys()) - set(keys)
            raise RuntimeError(f"state_dict keys mismatch. Only in ref: {extra_a}; only in other: {extra_b}")

    avg_state = OrderedDict()
    n = len(ckpts)
    for k in keys:
        ref_t = ref[k]
        if isinstance(ref_t, torch.Tensor) and ref_t.is_floating_point():
            acc = torch.zeros_like(ref_t, dtype=torch.float64)
            for c in ckpts:
                acc += c["state_dict"][k].to(torch.float64)
            avg_state[k] = (acc / n).to(ref_t.dtype)
        else:
            avg_state[k] = ref_t

    out_ckpt = dict(ckpts[-1])
    out_ckpt["state_dict"] = avg_state

    out_path = OUT_DIR / args.out_name
    torch.save(out_ckpt, out_path)
    print(f"\nWrote {out_path}")
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  averaged epochs: {EPOCHS}")


if __name__ == "__main__":
    main()
