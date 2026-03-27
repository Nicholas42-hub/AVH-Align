"""
Perturbation Robustness Experiment
====================================
Experiment 3: Measure how well Z_c and Z_s maintain detection AUC on
FakeAVCeleb as Gaussian noise is added to the input AV-HuBERT features.

Hypothesis (if disentanglement works):
  Z_c (causal branch) should degrade LESS than Z_s (spurious branch)
  under audio noise, because:
    - Z_c captures A/V temporal sync — robust to mild acoustic noise
    - Z_s captures acoustic fingerprints (TTS artefacts etc.) — directly
      destroyed by audio noise → detection AUC collapses toward 0.5

Metric: FakeAVCeleb cross-dataset detection AUC for each head, at each
  noise level σ:
    AUC_c(σ)  — from causal_head(Z_c)
    AUC_s(σ)  — from spurious_head(Z_s)
    AUC_full(σ) — from full_head(Z_c + Z_s)

  Expected: AUC_c(σ) degrades more slowly than AUC_s(σ) as σ increases.
  AUC_s at clean is already ~0.3 (inverted) and should move toward 0.5
  (random) as noise destroys acoustic artefacts.

Perturbation types:
  audio_noise      — Gaussian noise on audio features    (σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0})
  visual_noise     — Gaussian noise on visual features   (σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0})
  audio_masking    — Zero-out random audio frames        (rate ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0})
  visual_masking   — Zero-out random visual frames       (rate ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0})
  temporal_shuffle — Shuffle audio frames in time        (frac ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0})
                     Destroys A/V temporal sync — the causal signal.
                     Expected: hurts all models; disentangled models degrade
                     on Z_c (sync lost) but Z_s remains stable.

Noise/masking is applied BEFORE L2 normalisation.

Usage:
  python eval_perturbation_robustness.py \\
      --ckpt_path      outputs_A1/ckpts/model-epoch=27.ckpt \\
      --config         configs/A1.yaml \\
      --favc_root      ../data/favc_features \\
      --csv_root_path  csv_metadata/favc \\
      --output_dir     outputs_A1/results
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets import FakeAVCeleb_NPZ_Dataset


# ── Perturbation parameters ───────────────────────────────────────────────────
# Each entry maps perturbation type → list of intensity levels.
#   audio/visual_noise:    σ (Gaussian std)
#   audio/visual_masking:  frame dropout rate in [0, 1]
#   temporal_shuffle:      fraction of audio frames randomly permuted

PERTURB_CONFIG = {
    "audio_noise":      [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "visual_noise":     [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "audio_masking":    [0.0, 0.1,  0.25, 0.5, 0.75, 1.0],
    "visual_masking":   [0.0, 0.1,  0.25, 0.5, 0.75, 1.0],
    "temporal_shuffle": [0.0, 0.1,  0.25, 0.5, 0.75, 1.0],
}


# ── Collate helper ────────────────────────────────────────────────────────────

def _identity_collate(batch):
    """Pass single clips through unchanged — requires batch_size=1."""
    video, audio, label, path = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label]), [path]


# ── Noise helper ─────────────────────────────────────────────────────────────

def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def apply_perturbation(video: torch.Tensor, audio: torch.Tensor,
                       perturb_type: str, level: float) -> tuple:
    """
    Apply a perturbation of `level` intensity to features and return
    (video, audio) ready for model._encode (L2-normalised per frame).
    level=0.0 always returns clean normalised features.

    video/audio shape: [B, T, D]
    """
    if level == 0.0:
        return _l2_norm(video), _l2_norm(audio)

    if perturb_type == "audio_noise":
        audio = audio + torch.randn_like(audio) * level
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "visual_noise":
        video = video + torch.randn_like(video) * level
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "audio_masking":
        # Zero-out each audio frame independently with prob=level
        B, T, D = audio.shape
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=audio.device)
        )
        audio = audio * mask
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "visual_masking":
        B, T, D = video.shape
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - level, device=video.device)
        )
        video = video * mask
        return _l2_norm(video), _l2_norm(audio)

    elif perturb_type == "temporal_shuffle":
        # Randomly permute `level` fraction of audio frames, breaking AV sync.
        # Video frames are kept in original order.
        B, T, _ = audio.shape
        n_shuffle = max(1, int(T * level))
        audio = audio.clone()
        for b in range(B):
            idx = torch.randperm(T, device=audio.device)[:n_shuffle]
            perm = idx[torch.randperm(n_shuffle, device=audio.device)]
            audio[b, idx] = audio[b, perm]
        return _l2_norm(video), _l2_norm(audio)

    else:
        raise ValueError(f"Unknown perturb_type: {perturb_type}")


# ── Inference over FAVC under a single noise setting ─────────────────────────

@torch.no_grad()
def eval_auc(model,
             dataloader: DataLoader,
             perturb_type: str,
             level: float,
             device: torch.device,
             seed: int = 42) -> dict:
    """
    Run inference on the full FakeAVCeleb dataset with perturbed input and
    return AUC for causal_head, spurious_head, and full_head.
    """
    model.eval()
    model.to(device)

    # Seed global RNG for reproducibility across perturbation levels
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    all_causal, all_spurious, all_full, all_labels = [], [], [], []

    for batch in tqdm(dataloader, desc=f"  level={level:.2f}", leave=False):
        video_raw, audio_raw, labels, _ = batch
        video_raw = video_raw.to(device)
        audio_raw = audio_raw.to(device)

        video_raw, audio_raw = apply_perturbation(video_raw, audio_raw,
                                                  perturb_type, level)

        if hasattr(model, 'predict_scores') and 'mode' in model.predict_scores.__code__.co_varnames:
            causal_s   = model.predict_scores(video_raw, audio_raw, mode="causal")
            spurious_s = model.predict_scores(video_raw, audio_raw, mode="spurious")
            full_s     = model.predict_scores(video_raw, audio_raw, mode="full")
        else:
            # baseline AVH_Sup: single head, no causal/spurious split
            full_s     = model.predict_scores(video_raw, audio_raw)
            causal_s   = full_s
            spurious_s = torch.zeros_like(full_s)

        all_causal.extend(causal_s.cpu().numpy().tolist())
        all_spurious.extend(spurious_s.cpu().numpy().tolist())
        all_full.extend(full_s.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    def safe_auc(scores):
        if len(set(all_labels)) < 2:
            return None
        return float(roc_auc_score(all_labels, scores))

    return {
        "causal_auc":   safe_auc(all_causal),
        "spurious_auc": safe_auc(all_spurious),
        "full_auc":     safe_auc(all_full),
        "n_clips":      len(all_labels),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Perturbation robustness: detection AUC under input noise.")
    parser.add_argument("--ckpt_path",      required=True)
    parser.add_argument("--model",          choices=["causal", "ablation", "modal", "baseline", "fcd", "fcd_a6", "fcd_a7", "sad_a8"], default="ablation",
                        help="causal=AVH_Causal, ablation=AVH_Causal_Ablation (A0-A4), fcd=AVH_FCD (A5), fcd_a6=AVH_FCD_A6 (A6), fcd_a7=AVH_FCD_A7 (A7), sad_a8=AVH_SAD_A8 (A8), modal=AVH_Causal_Modal (B1/B2/B3), baseline=AVH_Sup (paper model)")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--favc_root",      required=True,
                        help="Root of FakeAVCeleb NPZ features")
    parser.add_argument("--csv_root_path",  default="csv_metadata/favc",
                        help="Dir with {split}_split.csv files")
    parser.add_argument("--split",          default="test")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output_dir",     default=None)
    parser.add_argument("--fake_category",  default=None,
                        help="If set, restrict evaluation to RealVideo-RealAudio + this fake "
                             "category only (e.g. 'RealVideo-FakeAudio' or 'FakeVideo-RealAudio')")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    print(f"Loading checkpoint: {args.ckpt_path}")
    if args.model == "causal":
        from mlp_causal import AVH_Causal
        model = AVH_Causal.load_from_checkpoint(args.ckpt_path)
    elif args.model == "mi":
        from mlp_causal_mi import AVH_Causal_MI
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_Causal_MI(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    elif args.model == "modal":
        from mlp_causal_modal import AVH_Causal_Modal
        model = AVH_Causal_Modal.load_from_checkpoint(args.ckpt_path)
    elif args.model == "baseline":
        from mlp import AVH_Sup
        model = AVH_Sup.load_from_checkpoint(args.ckpt_path)
    elif args.model == "fcd":
        from mlp_fcd import AVH_FCD
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_FCD(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    elif args.model == "fcd_a6":
        from mlp_fcd_a6 import AVH_FCD_A6
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_FCD_A6(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    elif args.model == "fcd_a7":
        from mlp_fcd_a7 import AVH_FCD_A7
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_FCD_A7(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    elif args.model == "sad_a8":
        from mlp_sad_a8 import AVH_SAD_A8
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_SAD_A8(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    else:
        from mlp_causal_ablation import AVH_Causal_Ablation
        ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
        model = AVH_Causal_Ablation(config=ckpt["hyper_parameters"]["config"])
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                          if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered_state, strict=False)
    device = torch.device(args.device)

    # ── Build dataset ─────────────────────────────────────────────────────────
    ds_config = {
        "root_path":     args.favc_root,
        "csv_root_path": args.csv_root_path,
        "apply_l2":      True,   # normalise per-frame; noise added on top in eval_auc
    }
    print(f"\nLoading FakeAVCeleb ({args.split}) from: {args.favc_root}")
    ds = FakeAVCeleb_NPZ_Dataset(ds_config, split=args.split)

    # Optional: restrict to RealVideo-RealAudio + one specific fake category
    if args.fake_category:
        keep = {"RealVideo-RealAudio", args.fake_category}
        ds.items = [(a, r, l, c) for a, r, l, c in ds.items if c in keep]
        n_real = sum(1 for _, _, l, _ in ds.items if l == 0)
        n_fake = sum(1 for _, _, l, _ in ds.items if l == 1)
        print(f"  → filtered to {args.fake_category}: {len(ds.items)} clips "
              f"(real={n_real}, fake={n_fake})")

    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=4, collate_fn=_identity_collate)

    all_results = {
        "ckpt_path":     args.ckpt_path,
        "split":         args.split,
        "seed":          args.seed,
        "fake_category": args.fake_category,
        "perturb_types": {},
    }

    for ptype, levels in PERTURB_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"  Perturbation: {ptype}")
        print(f"  {'level':>6}  {'Causal AUC':>12}  {'Spurious AUC':>13}  {'Full AUC':>10}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*13}  {'-'*10}")

        type_results = {}
        for level in levels:
            res = eval_auc(model, dl, ptype, level, device, seed=args.seed)
            cauc = res["causal_auc"]
            sauc = res["spurious_auc"]
            fauc = res["full_auc"]
            print(f"  {level:>6.2f}  {cauc:>12.4f}  {sauc:>13.4f}  {fauc:>10.4f}")
            type_results[str(level)] = res

        all_results["perturb_types"][ptype] = type_results

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        suffix = f"_{args.fake_category.replace('-', '_')}" if args.fake_category else ""
        out_path = os.path.join(args.output_dir, f"perturbation_robustness{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    main()
