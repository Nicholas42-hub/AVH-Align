"""
Trained Head Usage / Subspace Ablation
========================================
Answers: "which subspace dimensions does the TRAINED classification head
actually use when making predictions?"

Protocol:
  1. Run model._encode() to get Z_c [1, T, causal_dim].
  2. For each ablation mask: zero out specific dimension ranges in Z_c.
  3. Pass modified Z_c through the trained causal_head + _cls_score().
  4. Measure AUC on:
       (a) AV1M val  (in-domain)
       (b) FAVC per-category (FV-RA, RV-FA, FV-FA, overall)
  5. Report ΔAUC = ablated − baseline for each ablation.

Ablation masks defined:

  A2: (proj_dim=512 per sub; Z_c=[v_c(0:512) ; a_c(512:1024)])
    - mask_vc      zero out v_c (dims 0:512)   → head can only see a_c
    - mask_ac      zero out a_c (dims 512:1024) → head can only see v_c

  A5/A6: (dim=256 each; Z_c=[s_v|u_v|s_a|u_a])
    - mask_sv      zero out s_v (dims 0:256)
    - mask_uv      zero out u_v (dims 256:512)
    - mask_sa      zero out s_a (dims 512:768)
    - mask_ua      zero out u_a (dims 768:1024)
    - mask_vis     zero out s_v + u_v (dims 0:512)   → head sees audio only
    - mask_aud     zero out s_a + u_a (dims 512:1024) → head sees visual only
    - mask_syn     zero out s_v + s_a (dims 0:256, 512:768) → remove shared signal
    - mask_uniq    zero out u_v + u_a (dims 256:512, 768:1024) → remove unique signal

Output: head_ablation.json

Usage:
  python eval_head_ablation.py \\
      --ckpt_path  outputs_A6/ckpts/model-epoch=15.ckpt \\
      --config     configs/A6.yaml \\
      --av1m_root  ../data/avh_features/val \\
      --av1m_csv   csv_metadata/av1m/val_labels.csv \\
      --favc_root  ../data/favc_features \\
      --output_dir outputs_A6/results \\
      --device     cuda
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mlp_causal_ablation import AVH_Causal_Ablation
from mlp_causal_mi import AVH_Causal_MI
from mlp_fcd import AVH_FCD
from mlp_fcd_a6 import AVH_FCD_A6
from mlp_fcd_a7 import AVH_FCD_A7
from mlp_sad_a8 import AVH_SAD_A8

_FAVC_REAL_CATEGORY = "RealVideo-RealAudio"
_FAVC_FAKE_CATEGORIES = {
    "FakeVideo-RealAudio": "FV-RA",
    "FakeVideo-FakeAudio": "FV-FA",
    "RealVideo-FakeAudio": "RV-FA",
}


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


# ── Build ablation masks ────────────────────────────────────────────────────────

def _build_ablation_masks(model, causal_dim):
    """
    Returns dict: mask_name -> list of (start, end) dim ranges to ZERO OUT in Z_c.

    Convention: mask = list of (lo, hi) pairs; those dimensions are set to 0.
    Keeping remaining dims = head sees only the complement.
    """
    masks = {}
    if not _is_fcd(model):
        # A2: proj_dim default 512; causal_dim = 512*2 = 1024
        half = causal_dim // 2
        masks["mask_vc"]  = [(0, half)]          # zero v_c → only a_c survives
        masks["mask_ac"]  = [(half, causal_dim)]  # zero a_c → only v_c survives
    else:
        # FCD: [s_v|u_v|s_a|u_a] each 256
        q = causal_dim // 4      # = 256
        masks["mask_sv"]    = [(0,   q)]
        masks["mask_uv"]    = [(q,   2*q)]
        masks["mask_sa"]    = [(2*q, 3*q)]
        masks["mask_ua"]    = [(3*q, 4*q)]
        masks["mask_vis"]   = [(0,   2*q)]        # zero all visual (s_v + u_v)
        masks["mask_aud"]   = [(2*q, 4*q)]        # zero all audio  (s_a + u_a)
        masks["mask_syn"]   = [(0,   q), (2*q, 3*q)]   # zero s_v + s_a
        masks["mask_uniq"]  = [(q,   2*q), (3*q, 4*q)] # zero u_v + u_a
    return masks


def _apply_mask(zc, ranges):
    """Zero out dims in ranges; zc: [1, T, D] or [1, D]."""
    zc = zc.clone()
    for lo, hi in ranges:
        zc[..., lo:hi] = 0.0
    return zc


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, items, apply_l2=True):
        self.items    = items
        self.apply_l2 = apply_l2
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        data  = np.load(path, allow_pickle=True)
        v = data["visual"].astype(np.float32)
        a = data["audio"].astype(np.float32)
        if self.apply_l2:
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(v), torch.tensor(a), label

def _seq_collate(batch):
    v, a, l = batch[0]
    return v.unsqueeze(0), a.unsqueeze(0), torch.tensor([l])


# ── Item loaders ───────────────────────────────────────────────────────────────

def _load_av1m_items(root, csv_path):
    items = []
    with open(csv_path) as f:
        hdr  = f.readline().strip().split(",")
        pc   = hdr.index("path")
        lc   = hdr.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            npz = os.path.join(root, parts[pc].replace(".mp4", ".npz"))
            if os.path.exists(npz):
                items.append((npz, int(parts[lc])))
    return items

def _load_favc_items_by_category(root):
    """Returns dict: category_name -> list of (path, label)."""
    cats = {}
    for cat in os.listdir(root):
        cat_path = os.path.join(root, cat)
        if not os.path.isdir(cat_path):
            continue
        if cat == _FAVC_REAL_CATEGORY:
            label = 0
        elif cat in _FAVC_FAKE_CATEGORIES:
            label = 1
        else:
            continue
        items = []
        for dp, _, fnames in os.walk(cat_path):
            for fn in sorted(fnames):
                if fn.endswith(".npz"):
                    items.append((os.path.join(dp, fn), label))
        if items:
            cats[cat] = items
    return cats


# ── Per-clip inference with optional Z_c masking ──────────────────────────────

@torch.no_grad()
def run_inference(model, dataloader, device, ablation_ranges=None):
    """
    Run inference on a dataset.
    If ablation_ranges is not None, zero those dims of Z_c before the head.
    Returns (scores [N], labels [N]).
    """
    model.eval()
    model.to(device)
    scores_list = []
    labels_list = []

    for batch in tqdm(dataloader, desc="  Inferring", leave=False):
        video, audio, lbl = batch
        video = video.to(device)   # [1, T, D]
        audio = audio.to(device)

        causal_repr, _, _ = model._encode(video, audio)
        # causal_repr: [1, T, causal_dim]

        if ablation_ranges is not None:
            causal_repr = _apply_mask(causal_repr, ablation_ranges)

        # Apply the trained causal head: per-frame logit → logsumexp for clip score
        if _is_fcd(model):
            head = model.causal_head
        else:
            head = model.causal_head
        per_frame = head(causal_repr)[..., 0]          # [1, T]
        score     = torch.logsumexp(per_frame, dim=-1) # [1]

        scores_list.append(score.cpu().item())
        labels_list.append(lbl.item())

    return np.array(scores_list), np.array(labels_list, dtype=np.int32)


def _auc(scores, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--av1m_root",  required=True)
    parser.add_argument("--av1m_csv",   required=True)
    parser.add_argument("--favc_root",  required=True)
    parser.add_argument("--max_av1m",   type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    model, _ = _load_model(args.ckpt_path)
    model.eval()
    device = torch.device(args.device)
    print(f"Model : {model.__class__.__name__}  ({args.ckpt_path})")
    print(f"Device: {device}")
    model.to(device)

    # causal_dim
    if _is_fcd(model):
        hp = config.get("model_hparams", {})
        syn_dim  = int(hp.get("syn_dim",  256))
        spec_dim = int(hp.get("spec_dim", 256))
        causal_dim = (syn_dim + spec_dim) * 2   # 1024
    else:
        hp = config.get("model_hparams", {})
        proj_dim   = int(hp.get("proj_dim", 512))
        causal_dim = proj_dim * 2               # 1024

    ablation_masks = _build_ablation_masks(model, causal_dim)
    print(f"\nCausal dim : {causal_dim}")
    print(f"Ablations  : {list(ablation_masks.keys())}")

    # ── Load data ──────────────────────────────────────────────────────────
    av1m_items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"\nAV1M val : {len(av1m_items)} clips")
    if args.max_av1m and len(av1m_items) > args.max_av1m:
        import random
        random.seed(42)
        av1m_items = random.sample(av1m_items, args.max_av1m)
        print(f"  subsampled to {len(av1m_items)}")

    favc_cats = _load_favc_items_by_category(args.favc_root)
    real_cat  = _FAVC_REAL_CATEGORY
    real_items = favc_cats.get(real_cat, [])

    # Build per-category datasets: each fake cat + real items
    favc_per_cat = {}
    all_favc_items = list(real_items)
    for cat, items in favc_cats.items():
        if cat == real_cat:
            continue
        short = _FAVC_FAKE_CATEGORIES[cat]
        favc_per_cat[short] = real_items + items
        all_favc_items = list(set(all_favc_items) | set(items))
    favc_per_cat["overall"] = real_items + [
        item for cat, items in favc_cats.items() if cat != real_cat for item in items
    ]

    # Deduplicate overall
    seen = set()
    deduped = []
    for item in favc_per_cat["overall"]:
        if item[0] not in seen:
            deduped.append(item)
            seen.add(item[0])
    favc_per_cat["overall"] = deduped
    print(f"FAVC categories: "
          + ", ".join(f"{k}({len(v)})" for k, v in favc_per_cat.items()))

    # ── Build loaders ──────────────────────────────────────────────────────
    av1m_loader = DataLoader(SequenceDataset(av1m_items, apply_l2=apply_l2),
                             batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=_seq_collate)
    favc_loaders = {
        name: DataLoader(SequenceDataset(items, apply_l2=apply_l2),
                         batch_size=1, shuffle=False, num_workers=0,
                         collate_fn=_seq_collate)
        for name, items in favc_per_cat.items()
    }

    # ── Run ablation ───────────────────────────────────────────────────────
    all_ablations = {"baseline": None, **ablation_masks}
    results = {
        "ckpt_path":   args.ckpt_path,
        "model_class": model.__class__.__name__,
        "causal_dim":  causal_dim,
        "ablations":   {},
    }

    for abl_name, abl_ranges in all_ablations.items():
        print(f"\n── Ablation: {abl_name} "
              + (f"(zero dims {abl_ranges})" if abl_ranges else "(baseline)"))

        # AV1M
        sc, lb = run_inference(model, av1m_loader, device, abl_ranges)
        av1m_auc = _auc(sc, lb)

        # FAVC per-category
        cat_aucs = {}
        for cat_name, loader in favc_loaders.items():
            sc, lb = run_inference(model, loader, device, abl_ranges)
            cat_aucs[cat_name] = _auc(sc, lb)

        print(f"  AV1M: {av1m_auc:.4f}  |  FAVC: "
              + "  ".join(f"{k}={v:.4f}" for k, v in cat_aucs.items()))

        results["ablations"][abl_name] = {
            "av1m_auc":  av1m_auc,
            "favc":      cat_aucs,
        }

    # ── Compute ΔAUC vs baseline ───────────────────────────────────────────
    baseline = results["ablations"]["baseline"]
    print("\n── ΔAUC (ablated − baseline) ──")
    print(f"  {'Ablation':<14} {'AV1M':>7} "
          + "  ".join(f"{k:>9}" for k in favc_per_cat.keys()))
    for abl_name, abl_data in results["ablations"].items():
        if abl_name == "baseline":
            continue
        dav1m = abl_data["av1m_auc"] - baseline["av1m_auc"]
        row = [f"  {abl_name:<14} {dav1m:>+7.4f}"]
        for cat_name in favc_per_cat.keys():
            d = abl_data["favc"].get(cat_name, float("nan")) - \
                baseline["favc"].get(cat_name, float("nan"))
            row.append(f"{d:>+9.4f}")
        print("  ".join(row))

        # Store delta in results
        results["ablations"][abl_name]["delta_av1m"] = \
            abl_data["av1m_auc"] - baseline["av1m_auc"]
        results["ablations"][abl_name]["delta_favc"] = {
            k: abl_data["favc"].get(k, float("nan")) - baseline["favc"].get(k, float("nan"))
            for k in favc_per_cat.keys()
        }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "head_ablation.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out}")

    return results


if __name__ == "__main__":
    main()
