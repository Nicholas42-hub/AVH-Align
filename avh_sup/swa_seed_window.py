"""
Generic SWA over a list of favc_fvra/best-* ckpts for any seed.

Usage:
  python swa_seed_window.py --seed 43 --epochs 59 61 62 77 --out_name swa_e59_77.ckpt

INPUT (read-only):
  outputs_A6_av1m_pseudodomain_v2_50ep_seed{SEED}/ckpts/favc_fvra/
    best-favc-fvra-epoch={E}-auc=*.ckpt for each E in --epochs

OUTPUT (NEW directory):
  outputs_A6_av1m_pseudodomain_v2_50ep_seed{SEED}_SWA/ckpts/{out_name}
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


BASE = Path(
    "/data/projects/punim2637/nnliang/AVH-Align/avh_sup"
)


def find_ckpt(seed: int, epoch: int) -> Path:
    src_dir = BASE / f"outputs_A6_av1m_pseudodomain_v2_50ep_seed{seed}/ckpts/favc_fvra"
    matches = sorted(src_dir.glob(f"best-favc-fvra-epoch={epoch:02d}-auc=*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"seed={seed} epoch={epoch}: expected 1 match, got {len(matches)}: {matches}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--out_name", required=True)
    args = parser.parse_args()

    out_dir = BASE / f"outputs_A6_av1m_pseudodomain_v2_50ep_seed{args.seed}_SWA/ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [find_ckpt(args.seed, e) for e in args.epochs]
    print(f"[seed {args.seed}] averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p.name}")

    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]

    ref = ckpts[0]["state_dict"]
    keys = list(ref.keys())
    for c, p in zip(ckpts[1:], paths[1:]):
        if set(c["state_dict"].keys()) != set(keys):
            extra_a = set(keys) - set(c["state_dict"].keys())
            extra_b = set(c["state_dict"].keys()) - set(keys)
            raise RuntimeError(
                f"state_dict keys mismatch with {p.name}.\n  only in ref: {extra_a}\n  only in other: {extra_b}"
            )

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

    out_path = out_dir / args.out_name
    torch.save(out_ckpt, out_path)
    print(f"\nWrote {out_path}")
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  averaged epochs: {args.epochs}")


if __name__ == "__main__":
    main()
