"""
AVLips visual/audio masking sweep at seed 44.

Reuses model loading from eval_avlips.py and apply_perturbation from
eval_perturbation_robustness.py. Sweeps visual_masking and audio_masking at
levels {0.0, 0.5, 1.0} on the AVLips test set and reports AUC.
"""

import argparse
import json
import os
import sys

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets import AVLips_NPZ_Dataset
from eval_perturbation_robustness import apply_perturbation


PERTURB_LEVELS = {
    "visual_masking": [0.0, 0.5, 1.0],
    "audio_masking":  [0.0, 0.5, 1.0],
}


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true=y_true, y_score=y_score))


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return float(average_precision_score(y_true=y_true, y_score=y_score))


def load_model(model_arg, ckpt_path):
    if model_arg == "baseline":
        from mlp import AVH_Sup
        return AVH_Sup.load_from_checkpoint(ckpt_path)
    if model_arg == "fcd":
        from mlp_fcd import AVH_FCD
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = AVH_FCD(config=ckpt["hyper_parameters"]["config"])
        ts = m.state_dict()
        filt = {k: v for k, v in ckpt["state_dict"].items()
                if k in ts and v.shape == ts[k].shape}
        m.load_state_dict(filt, strict=False)
        return m
    if model_arg == "fcd_a6":
        from mlp_fcd_a6 import AVH_FCD_A6
        try:
            return AVH_FCD_A6.load_from_checkpoint(ckpt_path)
        except RuntimeError:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = dict(ckpt["hyper_parameters"]["config"])
            head0 = ckpt["state_dict"].get("causal_head.0.weight")
            if head0 is not None and head0.shape[1] == 1024:
                hp = dict(cfg.get("model_hparams", {}))
                hp["incon_dim"] = 0
                cfg["model_hparams"] = hp
            m = AVH_FCD_A6(config=cfg)
            m.load_state_dict(ckpt["state_dict"], strict=True)
            return m
    if model_arg == "ablation":
        from mlp_causal_ablation import AVH_Causal_Ablation
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = AVH_Causal_Ablation(config=ckpt["hyper_parameters"]["config"])
        ts = m.state_dict()
        filt = {k: v for k, v in ckpt["state_dict"].items()
                if k in ts and v.shape == ts[k].shape}
        m.load_state_dict(filt, strict=False)
        return m
    raise ValueError(f"Unknown model: {model_arg}")


CAUSAL_MODELS = {"fcd", "fcd_a6", "ablation"}


def _identity_collate(batch):
    video, audio, label, path = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label]), [path]


@torch.no_grad()
def run_sweep(model, loader, device, ptype, level, seed, is_causal):
    model.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    full_s, causal_s, spur_s, labels = [], [], [], []
    for batch in tqdm(loader, desc=f"  {ptype}@{level:.2f}", leave=False):
        v, a, lbl, _ = batch
        v, a = v.to(device), a.to(device)
        v, a = apply_perturbation(v, a, ptype, level)
        if is_causal:
            full_s.extend(model.predict_scores(v, a, mode="full").cpu().numpy().tolist())
            causal_s.extend(model.predict_scores(v, a, mode="causal").cpu().numpy().tolist())
            spur_s.extend(model.predict_scores(v, a, mode="spurious").cpu().numpy().tolist())
        else:
            s = model.predict_scores(v, a)
            full_s.extend(s.cpu().numpy().tolist())
            causal_s.extend(s.cpu().numpy().tolist())
            spur_s.extend([0.0] * len(s))
        labels.extend(lbl.numpy().tolist())
    return {
        "full_auc":     safe_auc(labels, full_s),
        "causal_auc":   safe_auc(labels, causal_s),
        "spurious_auc": safe_auc(labels, spur_s),
        "full_ap":      safe_ap(labels, full_s),
        "n":            len(labels),
        "n_real":       labels.count(0),
        "n_fake":       labels.count(1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model", required=True,
                   choices=["baseline", "fcd", "fcd_a6", "ablation"])
    p.add_argument("--features_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--apply_l2", action="store_true", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Loading {args.model} from {args.ckpt}", flush=True)
    model = load_model(args.model, args.ckpt).to(device)
    is_causal = args.model in CAUSAL_MODELS

    ds_cfg = {"name": "AVLIPS_NPZ", "root_path": args.features_path, "apply_l2": args.apply_l2}
    ds = AVLips_NPZ_Dataset(ds_cfg)
    print(f"AVLips test set: {len(ds)} clips", flush=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                    collate_fn=_identity_collate)

    out = {
        "ckpt": args.ckpt,
        "model": args.model,
        "seed": args.seed,
        "perturb_types": {},
    }
    for ptype, levels in PERTURB_LEVELS.items():
        print(f"\n=== {ptype} ===", flush=True)
        type_res = {}
        for lvl in levels:
            r = run_sweep(model, dl, device, ptype, lvl, args.seed, is_causal)
            print(f"  level={lvl:.2f}  full_AUC={r['full_auc']:.4f}  full_AP={r['full_ap']:.4f}  N={r['n']}", flush=True)
            type_res[str(lvl)] = r
        out["perturb_types"][ptype] = type_res

    out_path = os.path.join(args.output_dir, "avlips_masking.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
