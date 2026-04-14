"""
Subspace-Level Linear Probe
============================
Answers: "what information does each internal subspace of A2/A5/A6 capture?"

For A2 (binary split):
  v_c  [0:512]   — visual causal
  a_c  [512:1024] — audio causal
  v_s  [0:512]   — visual spurious
  a_s  [512:1024] — audio spurious
  Z_c  = cat(v_c, a_c)   [1024]  (the full causal repr)
  Z_s  = cat(v_s, a_s)   [1024]

For A5/A6 (FCD three-way):
  s_v  [0:256]    — visual synergistic (cross-modal shared)
  u_v  [256:512]  — visual unique      (modality-specific task signal)
  s_a  [512:768]  — audio synergistic
  u_a  [768:1024] — audio unique
  r_v  [0:256]    — visual redundant   (spurious residual)
  r_a  [256:512]  — audio redundant
  Z_c  = cat(s_v, u_v, s_a, u_a) [1024]
  Z_s  = cat(r_v, r_a)           [512]
  + compound subspaces:
    vis_all    = cat(s_v, u_v)   [512]  — all visual signal in Z_c
    aud_all    = cat(s_a, u_a)   [512]  — all audio signal in Z_c
    syn_all    = cat(s_v, s_a)   [512]  — all synergistic (cross-modal shared)
    uniq_all   = cat(u_v, u_a)   [512]  — all unique (modality-specific task signal)

Probes run:
  (1) In-domain label AUC   — 5-fold CV on AV1M val (fake/real)
  (2) OOD label AUC         — train AV1M val → eval FAVC
  (3) Domain AUC            — AV1M=0 vs FAVC=1

Output: subspace_probe.json

Usage:
  python eval_subspace_probe.py \\
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
import random
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
_FAVC_FAKE_CATEGORIES = {"FakeVideo-RealAudio", "FakeVideo-FakeAudio", "RealVideo-FakeAudio"}


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str):
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


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, items: list, apply_l2: bool = True):
        self.items    = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data  = np.load(path, allow_pickle=True)
        video = data["visual"].astype(np.float32)
        audio = data["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(video), torch.tensor(audio), label

def _seq_collate(batch):
    video, audio, label = batch[0]
    return video.unsqueeze(0), audio.unsqueeze(0), torch.tensor([label])


# ── Item loaders ───────────────────────────────────────────────────────────────

def _load_av1m_items(root, csv_path):
    items = []
    with open(csv_path) as f:
        hdr   = f.readline().strip().split(",")
        pc    = hdr.index("path")
        lc    = hdr.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            npz = os.path.join(root, parts[pc].replace(".mp4", ".npz"))
            if os.path.exists(npz):
                items.append((npz, int(parts[lc])))
    return items

def _load_favc_items(root):
    items = []
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
        for dp, _, fnames in os.walk(cat_path):
            for fn in sorted(fnames):
                if fn.endswith(".npz"):
                    items.append((os.path.join(dp, fn), label))
    return items


# ── Subspace extraction ────────────────────────────────────────────────────────

def _is_fcd(model) -> bool:
    return isinstance(model, (AVH_FCD, AVH_FCD_A6))


@torch.no_grad()
def extract_subspaces(model, dataloader, device):
    """
    Returns a dict of subspace name → numpy array [N, D].
    Also returns labels [N].
    """
    model.eval()
    model.to(device)
    is_fcd = _is_fcd(model)

    buffers = {}   # name -> list of [1, D] arrays
    labels  = []

    for batch in tqdm(dataloader, desc="  Extracting", leave=False):
        video, audio, lbl = batch
        video = video.to(device)
        audio = audio.to(device)

        causal_repr, spurious_repr, extra = model._encode(video, audio)
        # causal_repr:  [1, T, causal_dim]
        # spurious_repr: [1, T, spurious_dim]
        # extra: comp dict (FCD) or None (A2)

        if is_fcd:
            comp = extra
            s_v = comp["s_v"].mean(1).cpu().numpy()   # [1, 256]
            u_v = comp["u_v"].mean(1).cpu().numpy()
            r_v = comp["r_v"].mean(1).cpu().numpy()
            s_a = comp["s_a"].mean(1).cpu().numpy()
            u_a = comp["u_a"].mean(1).cpu().numpy()
            r_a = comp["r_a"].mean(1).cpu().numpy()

            zc  = causal_repr.mean(1).cpu().numpy()    # [1, 1024]
            zs  = spurious_repr.mean(1).cpu().numpy()  # [1, 512]

            row = {
                "s_v":      s_v,
                "u_v":      u_v,
                "s_a":      s_a,
                "u_a":      u_a,
                "r_v":      r_v,
                "r_a":      r_a,
                "Z_c":      zc,
                "Z_s":      zs,
                # compound subspaces
                "vis_all":  np.concatenate([s_v, u_v], axis=-1),  # [1,512]
                "aud_all":  np.concatenate([s_a, u_a], axis=-1),
                "syn_all":  np.concatenate([s_v, s_a], axis=-1),
                "uniq_all": np.concatenate([u_v, u_a], axis=-1),
                "r_all":    np.concatenate([r_v, r_a], axis=-1),
            }
        else:
            # A2: Z_c = cat(v_c, a_c), Z_s = cat(v_s, a_s); each half = proj_dim=512
            zc  = causal_repr.mean(1).cpu().numpy()    # [1, 1024]
            zs  = spurious_repr.mean(1).cpu().numpy()  # [1, 1024]
            half_c = zc.shape[-1] // 2
            half_s = zs.shape[-1] // 2
            row = {
                "v_c": zc[:, :half_c],
                "a_c": zc[:, half_c:],
                "v_s": zs[:, :half_s],
                "a_s": zs[:, half_s:],
                "Z_c": zc,
                "Z_s": zs,
            }

        for k, v in row.items():
            if k not in buffers:
                buffers[k] = []
            buffers[k].append(v) 
        labels.extend(lbl.tolist())

    return {k: np.concatenate(v, axis=0) for k, v in buffers.items()}, \
           np.array(labels, dtype=np.int32)


# ── Probe helpers ──────────────────────────────────────────────────────────────

def _probe_cv(features, labels, n_splits=5, seed=42):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(features, labels):
        sc  = StandardScaler().fit(features[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(sc.transform(features[tr]), labels[tr])
        p = clf.predict_proba(sc.transform(features[te]))[:, 1]
        aucs.append(roc_auc_score(labels[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

def _probe_transfer(tr_f, tr_l, te_f, te_l, seed=42):
    sc  = StandardScaler().fit(tr_f)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(sc.transform(tr_f), tr_l)
    p = clf.predict_proba(sc.transform(te_f))[:, 1]
    return float(roc_auc_score(te_l, p))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--config",     required=True)
    parser.add_argument("--av1m_root",  required=True)
    parser.add_argument("--av1m_csv",   required=True)
    parser.add_argument("--favc_root",  required=True)
    parser.add_argument("--n_folds",    type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max_av1m",   type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)

    model, _ = _load_model(args.ckpt_path)
    device   = torch.device(args.device)
    print(f"Model : {model.__class__.__name__}  ({args.ckpt_path})")
    print(f"Device: {device}")

    # ── Items ──────────────────────────────────────────────────────────────
    av1m_items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"\nAV1M val : {len(av1m_items)} clips")
    if args.max_av1m and len(av1m_items) > args.max_av1m:
        av1m_items = rng.sample(av1m_items, args.max_av1m)
        print(f"  subsampled to {len(av1m_items)}")

    favc_items = _load_favc_items(args.favc_root)
    print(f"FAVC     : {len(favc_items)} clips")

    # ── DataLoaders ────────────────────────────────────────────────────────
    av1m_loader = DataLoader(SequenceDataset(av1m_items, apply_l2=apply_l2),
                             batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=_seq_collate)
    favc_loader = DataLoader(SequenceDataset(favc_items, apply_l2=apply_l2),
                             batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=_seq_collate)

    # ── Extract ────────────────────────────────────────────────────────────
    print("\n[1/2] Extracting AV1M representations...")
    av1m_sub, av1m_lbl = extract_subspaces(model, av1m_loader, device)
    print(f"  Subspaces: {list(av1m_sub.keys())}")

    print("\n[2/2] Extracting FAVC representations...")
    favc_sub, favc_lbl = extract_subspaces(model, favc_loader, device)

    # ── Probe each subspace ────────────────────────────────────────────────
    print("\n── Subspace probes ──")
    results = {
        "ckpt_path": args.ckpt_path,
        "model_class": model.__class__.__name__,
        "n_av1m": len(av1m_items),
        "n_favc": len(favc_items),
        "subspaces": {},
    }

    subspace_names = list(av1m_sub.keys())
    for name in subspace_names:
        av1m_feat = av1m_sub[name]
        favc_feat = favc_sub[name]

        # In-domain label probe (AV1M 5-fold CV)
        ind_auc, ind_std = _probe_cv(av1m_feat, av1m_lbl, args.n_folds, args.seed)

        # OOD label probe (train AV1M → eval FAVC)
        ood_auc = _probe_transfer(av1m_feat, av1m_lbl, favc_feat, favc_lbl, args.seed)

        # Domain probe (AV1M=0 vs FAVC=1, 5-fold CV)
        dom_feat = np.concatenate([av1m_feat, favc_feat], axis=0)
        dom_lbl  = np.array([0] * len(av1m_feat) + [1] * len(favc_feat), dtype=np.int32)
        dom_auc, dom_std = _probe_cv(dom_feat, dom_lbl, args.n_folds, args.seed)

        print(f"  {name:12s}  label_ind={ind_auc:.4f}±{ind_std:.4f}  "
              f"label_ood={ood_auc:.4f}  domain={dom_auc:.4f}±{dom_std:.4f}  "
              f"dim={av1m_feat.shape[-1]}")

        results["subspaces"][name] = {
            "dim": int(av1m_feat.shape[-1]),
            "label_ind_auc": ind_auc,
            "label_ind_std": ind_std,
            "label_ood_auc": ood_auc,
            "domain_auc":    dom_auc,
            "domain_std":    dom_std,
        }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "subspace_probe.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out}")

    return results


if __name__ == "__main__":
    main()
