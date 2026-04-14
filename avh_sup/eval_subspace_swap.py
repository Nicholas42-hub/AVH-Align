"""
Intervention-style Subspace Swap Experiment
============================================
Answers: "does replacing a SINGLE subspace in recipient's Z_c with the
corresponding subspace from a donor clip causally control the fake score?"

Method (representation-level counterfactual / hard swap):
  For each (recipient_category, donor_category) pair:
    1. Encode every clip in both categories → get full Z_c per clip.
    2. For each recipient clip r, randomly sample a donor clip d.
    3. Construct modified Z_c:
         z'_c = copy of z_c^r, with dims [lo:hi] replaced by z_c^d[lo:hi]
    4. Run modified z'_c through the TRAINED causal head → get fake score.
    5. Compare to score from unmodified z_c^r (baseline).
    6. Report delta = mean(score_after − score_before).

Also runs INTERPOLATION at alpha ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:
    z' = (1−alpha)*z_c^r[subspace] + alpha*z_c^d[subspace]

Subspaces:
  A2: v_c=[0:512], a_c=[512:1024]          (causal model — control comparison)
  A5/A6 FCD: s_v=[0:256], u_v=[256:512],   (shared / unique visual)
             s_a=[512:768], u_a=[768:1024]  (shared / unique audio)

Output: subspace_swap.json
  {
    "ckpt_path": "...",
    "model_class": "...",
    "n_donors_per_recipient": ...,
    "seed": 42,
    "categories": ["RealVideo-RealAudio", "FakeVideo-RealAudio", ...],
    "subspaces": {
      "u_v": {
        "dim_range": [256, 512],
        "swap": {
          "RV-RA_to_FV-RA": {
            "recipient_cat": "RealVideo-RealAudio",
            "donor_cat":     "FakeVideo-RealAudio",
            "n_recipient":   100,
            "mean_score_baseline": 1.23,
            "mean_score_swapped":  4.56,
            "delta_score":         3.33,
            "std_delta":           0.5,
          },
          ...
        },
        "interpolation": {               # averaged over all donor/recipient pairs
          "RV-RA_to_FV-RA": {
            "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
            "mean_score": [1.23, 2.10, 3.00, 3.70, 4.56],
          },
          ...
        }
      },
      ...
    }
  }

Usage:
  python eval_subspace_swap.py \\
      --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
      --config     configs/A6.yaml \\
      --favc_root  ../data/favc_features \\
      --output_dir outputs_A6/results \\
      --n_donors   30 \\
      --seed       42 \\
      --device     cuda
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mlp_causal_ablation import AVH_Causal_Ablation
from mlp_causal_mi import AVH_Causal_MI
from mlp_fcd import AVH_FCD
from mlp_fcd_a6 import AVH_FCD_A6
from mlp_fcd_a7 import AVH_FCD_A7
from mlp_sad_a8 import AVH_SAD_A8


# ── Category aliases ───────────────────────────────────────────────────────────

_CAT_SHORT = {
    "RealVideo-RealAudio": "RV-RA",
    "FakeVideo-RealAudio": "FV-RA",
    "RealVideo-FakeAudio": "RV-FA",
    "FakeVideo-FakeAudio": "FV-FA",
}
_ALL_CATEGORIES = list(_CAT_SHORT.keys())

# Alpha levels for interpolation sweep
_INTERP_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["hyper_parameters"]["config"]
    hp   = cfg.get("model_hparams", {})
    if "sync_dim" in hp:
        model = AVH_SAD_A8(config=cfg)
    elif "lambda_ladv" in hp:
        model = AVH_FCD_A7(config=cfg)
    elif "lambda_dadv" in hp:
        model = AVH_FCD_A6(config=cfg)
    elif "syn_dim" in hp:
        model = AVH_FCD(config=cfg)
    elif "lambda_mi_spu" in hp or "lambda_club" in hp:
        model = AVH_Causal_MI(config=cfg)
    else:
        model = AVH_Causal_Ablation(config=cfg)
    ms = model.state_dict()
    filtered = {k: v for k, v in ckpt["state_dict"].items()
                if k in ms and v.shape == ms[k].shape}
    model.load_state_dict(filtered, strict=False)
    return model, cfg


def _is_fcd(model):
    return isinstance(model, (AVH_FCD, AVH_FCD_A6))


# ── Subspace definitions ───────────────────────────────────────────────────────

def _build_subspace_defs(model, causal_dim):
    """
    Returns ordered dict: subspace_name -> (lo, hi)
    """
    if not _is_fcd(model):
        # Binary causal model (A2): v_c / a_c halves
        half = causal_dim // 2
        return {
            "v_c": (0,    half),
            "a_c": (half, causal_dim),
        }
    else:
        # FCD (A5/A6): [s_v | u_v | s_a | u_a]
        q = causal_dim // 4   # = 256
        return {
            "s_v":  (0,     q),
            "u_v":  (q,     2*q),
            "s_a":  (2*q,   3*q),
            "u_a":  (3*q,   4*q),
        }


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_favc_by_category(root, apply_l2=True):
    """
    Returns dict: category_name -> list of (path, label)
    Only categories that exist under `root` and match _ALL_CATEGORIES.
    """
    cats = {}
    for cat in _ALL_CATEGORIES:
        cat_dir = os.path.join(root, cat)
        if not os.path.isdir(cat_dir):
            continue
        items = []
        for dp, _, fnames in os.walk(cat_dir):
            for fn in sorted(fnames):
                if fn.endswith(".npz"):
                    items.append(os.path.join(dp, fn))
        if items:
            cats[cat] = items
    return cats


def _load_npz(path, apply_l2=True):
    data = np.load(path, allow_pickle=True)
    v = data["visual"].astype(np.float32)
    a = data["audio"].astype(np.float32)
    if apply_l2:
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    return torch.tensor(v), torch.tensor(a)


# ── Feature cache: encode all clips once ──────────────────────────────────────

@torch.no_grad()
def _encode_all(model, paths, apply_l2, device):
    """
    Mean-pools Z_c over the time dimension → returns list of [causal_dim] tensors.
    Using mean-pooled clip-level representations avoids variable-T shape mismatches
    during subspace swap (clips have different numbers of frames).
    """
    model.eval()
    model.to(device)
    encoded = []
    for path in tqdm(paths, desc="  Encoding", leave=False):
        v, a = _load_npz(path, apply_l2)
        v = v.unsqueeze(0).to(device)   # [1, T, Dv]
        a = a.unsqueeze(0).to(device)   # [1, T, Da]
        zc, _, _ = model._encode(v, a)  # [1, T, causal_dim]
        # Mean-pool over T → [causal_dim]
        encoded.append(zc.squeeze(0).mean(0).cpu())  # [causal_dim]
    return encoded   # list of [causal_dim] tensors


# ── Score from Z_c ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_from_zc(model, zc, device):
    """
    zc: [causal_dim] CPU tensor (mean-pooled clip representation).
    Returns scalar fake score.
    """
    # [causal_dim] -> [1, 1, causal_dim] (batch=1, T=1)
    zc = zc.unsqueeze(0).unsqueeze(0).to(device)
    per_frame = model.causal_head(zc)[..., 0]   # [1, 1]
    score = torch.logsumexp(per_frame, dim=-1)  # [1]
    return score.cpu().item()


# ── Core experiment ────────────────────────────────────────────────────────────

def run_swap_experiment(
    model, device,
    cat_encodings,     # dict: category_name -> list[Tensor [T, D]]
    subspace_defs,     # dict: name -> (lo, hi)
    n_donors,          # int: number of donor clips per recipient
    rng,               # random.Random instance
):
    """
    For each subspace, for each (recipient_cat, donor_cat) pair:
      hard swap + interpolation sweep.

    Returns nested dict of results.
    """
    results = {}
    categories = sorted(cat_encodings.keys())

    for sp_name, (lo, hi) in subspace_defs.items():
        print(f"\n  Subspace: {sp_name} [{lo}:{hi}]")
        results[sp_name] = {
            "dim_range": [lo, hi],
            "swap": {},
            "interpolation": {},
        }

        for rec_cat in categories:
            rec_short = _CAT_SHORT.get(rec_cat, rec_cat)
            rec_zcs   = cat_encodings[rec_cat]

            for don_cat in categories:
                don_short = _CAT_SHORT.get(don_cat, don_cat)
                don_zcs   = cat_encodings[don_cat]

                pair_key = f"{rec_short}_to_{don_short}"
                print(f"    {rec_short} <- {don_short} ...", end=" ", flush=True)

                # Collect per-recipient delta scores
                delta_list   = []
                interp_scores = {alpha: [] for alpha in _INTERP_ALPHAS}
                baseline_list = []
                swapped_list  = []

                for rec_zc in rec_zcs:
                    # Sample donors (with replacement if needed)
                    donors = rng.choices(don_zcs, k=n_donors)

                    # Baseline score for this recipient
                    b_score = _score_from_zc(model, rec_zc, device)
                    baseline_list.append(b_score)

                    donor_scores  = []
                    interp_acc    = {alpha: [] for alpha in _INTERP_ALPHAS}

                    for don_zc in donors:
                        # ── Hard swap (1-D: both zc are [causal_dim]) ────
                        z_swapped = rec_zc.clone()
                        z_swapped[lo:hi] = don_zc[lo:hi]
                        s_score = _score_from_zc(model, z_swapped, device)
                        donor_scores.append(s_score)

                        # ── Interpolation sweep ───────────────────────────
                        for alpha in _INTERP_ALPHAS:
                            z_interp = rec_zc.clone()
                            z_interp[lo:hi] = (
                                (1.0 - alpha) * rec_zc[lo:hi]
                                + alpha       * don_zc[lo:hi]
                            )
                            i_score = _score_from_zc(model, z_interp, device)
                            interp_acc[alpha].append(i_score)

                    # Mean over donors for this recipient
                    mean_swapped = float(np.mean(donor_scores))
                    swapped_list.append(mean_swapped)
                    delta_list.append(mean_swapped - b_score)
                    for alpha in _INTERP_ALPHAS:
                        interp_scores[alpha].append(float(np.mean(interp_acc[alpha])))

                mean_delta = float(np.mean(delta_list))
                std_delta  = float(np.std(delta_list))
                print(f"Δ={mean_delta:+.4f} ± {std_delta:.4f}")

                results[sp_name]["swap"][pair_key] = {
                    "recipient_cat":      rec_cat,
                    "donor_cat":          don_cat,
                    "n_recipient":        len(rec_zcs),
                    "n_donors_each":      n_donors,
                    "mean_score_baseline": float(np.mean(baseline_list)),
                    "mean_score_swapped":  float(np.mean(swapped_list)),
                    "delta_score":         mean_delta,
                    "std_delta":           std_delta,
                    "per_recipient_delta": [round(d, 6) for d in delta_list],
                }
                results[sp_name]["interpolation"][pair_key] = {
                    "recipient_cat": rec_cat,
                    "donor_cat":     don_cat,
                    "alphas":        _INTERP_ALPHAS,
                    "mean_score_per_alpha": [
                        float(np.mean(interp_scores[a])) for a in _INTERP_ALPHAS
                    ],
                    "std_score_per_alpha": [
                        float(np.std(interp_scores[a])) for a in _INTERP_ALPHAS
                    ],
                }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--favc_root",  required=True,
                        help="Root of FakeAVCeleb feature directory")
    parser.add_argument("--n_donors",   type=int, default=30,
                        help="Number of donor clips sampled per recipient")
    parser.add_argument("--max_per_cat", type=int, default=None,
                        help="Cap clips per category for speed (default: all)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    model, _ = _load_model(args.ckpt_path)
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    print(f"Model : {model.__class__.__name__}  ({args.ckpt_path})")
    print(f"Device: {device}")

    # causal_dim
    if _is_fcd(model):
        hp = config.get("model_hparams", {})
        syn_dim  = int(hp.get("syn_dim",  256))
        spec_dim = int(hp.get("spec_dim", 256))
        causal_dim = (syn_dim + spec_dim) * 2
    else:
        hp = config.get("model_hparams", {})
        proj_dim   = int(hp.get("proj_dim", 512))
        causal_dim = proj_dim * 2

    subspace_defs = _build_subspace_defs(model, causal_dim)
    print(f"Causal dim : {causal_dim}")
    print(f"Subspaces  : { {k: v for k, v in subspace_defs.items()} }")

    # ── Load clip paths by category ────────────────────────────────────────
    cat_paths = _load_favc_by_category(args.favc_root, apply_l2=apply_l2)
    print(f"\nFAVC categories found: {list(cat_paths.keys())}")
    for cat, paths in cat_paths.items():
        print(f"  {_CAT_SHORT.get(cat,cat):6s}: {len(paths)} clips")

    # Optional cap
    if args.max_per_cat:
        for cat in list(cat_paths.keys()):
            paths = cat_paths[cat]
            if len(paths) > args.max_per_cat:
                rng_list = random.Random(args.seed)
                cat_paths[cat] = rng_list.sample(paths, args.max_per_cat)
        print(f"\nCapped to max {args.max_per_cat} clips per category.")
        for cat, paths in cat_paths.items():
            print(f"  {_CAT_SHORT.get(cat,cat):6s}: {len(paths)}")

    # ── Encode all clips ────────────────────────────────────────────────────
    print("\nEncoding clips...")
    cat_encodings = {}
    for cat, paths in cat_paths.items():
        print(f"  {_CAT_SHORT.get(cat,cat)} ({len(paths)} clips)...")
        cat_encodings[cat] = _encode_all(model, paths, apply_l2, device)

    # ── Run swap experiment ─────────────────────────────────────────────────
    print("\nRunning swap experiment...")
    swap_results = run_swap_experiment(
        model=model,
        device=device,
        cat_encodings=cat_encodings,
        subspace_defs=subspace_defs,
        n_donors=args.n_donors,
        rng=rng,
    )

    # ── Save ────────────────────────────────────────────────────────────────
    output = {
        "ckpt_path":          args.ckpt_path,
        "model_class":        model.__class__.__name__,
        "causal_dim":         causal_dim,
        "n_donors_per_recipient": args.n_donors,
        "seed":               args.seed,
        "categories":         {_CAT_SHORT.get(c,c): c for c in sorted(cat_paths.keys())},
        "n_clips_per_category": {
            _CAT_SHORT.get(c,c): len(cat_encodings[c]) for c in cat_paths
        },
        "subspaces": swap_results,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "subspace_swap.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved → {out_path}")

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SUMMARY: Δ fake score (swapped − baseline) by subspace + pair")
    print("="*72)
    categories_short = [_CAT_SHORT.get(c,c) for c in sorted(cat_paths.keys())]
    for sp_name in swap_results:
        print(f"\n  Subspace: {sp_name}")
        rec_don_lbl = "rec\\don"
        header = f"  {rec_don_lbl:>10s}" + "".join(f"  {d:>8s}" for d in categories_short)
        print(header)
        for rec_cat in sorted(cat_paths.keys()):
            rec_short = _CAT_SHORT.get(rec_cat, rec_cat)
            row = f"  {rec_short:>10s}"
            for don_cat in sorted(cat_paths.keys()):
                don_short = _CAT_SHORT.get(don_cat, don_cat)
                key = f"{rec_short}_to_{don_short}"
                d = swap_results[sp_name]["swap"].get(key, {})
                delta = d.get("delta_score", float("nan"))
                row += f"  {delta:+8.4f}"
            print(row)


if __name__ == "__main__":
    main()
