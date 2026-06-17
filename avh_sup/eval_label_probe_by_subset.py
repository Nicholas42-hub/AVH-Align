"""
Label + Domain Probe — per FAVC subset
======================================
Same probe machinery as eval_label_probe.py, but breaks the OOD label probe
and the domain probe down by FAVC fake category:

  - RealVideo-FakeAudio  (RV-FA)
  - FakeVideo-RealAudio  (FV-RA)   <-- the hardest case
  - FakeVideo-FakeAudio  (FV-FA)   <-- "easy" (both modalities fake)

For each subset we report, on Z_c, Z_s and raw AV-HuBERT:

  Label AUC  : probe trained on AV1M val, evaluated on (FAVC-real + subset-fake)
  Domain AUC : 5-fold CV on AV1M (=0) vs (FAVC-real + subset-fake) (=1)

Plus the standard in-domain (AV1M val 5-fold CV) and overall-FAVC numbers
for sanity / context.

Output: JSON + a printed markdown table.
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
from mlp_fcd import AVH_FCD
from mlp_fcd_a6 import AVH_FCD_A6
from mlp_fcd_a7 import AVH_FCD_A7

# Lazy imports for ckpt families that may not exist in this checkout.
def _try_import(name, cls):
    try:
        mod = __import__(name)
        return getattr(mod, cls)
    except (ImportError, AttributeError):
        return None

AVH_Causal_MI = _try_import("mlp_causal_mi", "AVH_Causal_MI")
AVH_SAD_A8 = _try_import("mlp_sad_a8", "AVH_SAD_A8")


_FAVC_REAL_CATEGORY = "RealVideo-RealAudio"
_FAVC_FAKE_CATEGORIES = ("RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio")
_SUBSET_ALIAS = {
    "RealVideo-FakeAudio": "RV-FA",
    "FakeVideo-RealAudio": "FV-RA  (hard)",
    "FakeVideo-FakeAudio": "FV-FA  (easy)",
}


def _load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["config"]
    hp = cfg.get("model_hparams", {})
    if "sync_dim" in hp:
        if AVH_SAD_A8 is None:
            raise ImportError("ckpt requires AVH_SAD_A8 but mlp_sad_a8 is not available")
        model = AVH_SAD_A8(config=cfg)
    elif "lambda_ladv" in hp:
        model = AVH_FCD_A7(config=cfg)
    elif "lambda_dadv" in hp:
        model = AVH_FCD_A6(config=cfg)
    elif "syn_dim" in hp:
        model = AVH_FCD(config=cfg)
    elif "lambda_mi_spu" in hp or "lambda_club" in hp:
        if AVH_Causal_MI is None:
            raise ImportError("ckpt requires AVH_Causal_MI but mlp_causal_mi is not available")
        model = AVH_Causal_MI(config=cfg)
    else:
        model = AVH_Causal_Ablation(config=cfg)
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in ckpt["state_dict"].items()
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    return model, cfg


class CategoryDataset(Dataset):
    """(visual, audio, label, category_idx) — keeps which FAVC category each clip belongs to."""

    def __init__(self, items, apply_l2=True):
        self.items = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, cat_idx = self.items[idx]
        data = np.load(path, allow_pickle=True)
        video = data["visual"].astype(np.float32)
        audio = data["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        video = video.mean(axis=0)
        audio = audio.mean(axis=0)
        return torch.tensor(video), torch.tensor(audio), label, cat_idx


def _load_av1m_items(av1m_root, csv_path):
    items = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        path_col = header.index("path")
        label_col = header.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            rel_path = parts[path_col]
            label = int(parts[label_col])
            npz = os.path.join(av1m_root, rel_path.replace(".mp4", ".npz"))
            if os.path.exists(npz):
                items.append((npz, label, -1))  # cat_idx unused for AV1M
    return items


def _load_favc_items(favc_root):
    """Returns items list and a category-index map."""
    cat_to_idx = {_FAVC_REAL_CATEGORY: 0}
    for i, c in enumerate(_FAVC_FAKE_CATEGORIES):
        cat_to_idx[c] = i + 1
    items = []
    for category in os.listdir(favc_root):
        cat_path = os.path.join(favc_root, category)
        if not os.path.isdir(cat_path) or category not in cat_to_idx:
            continue
        cat_idx = cat_to_idx[category]
        label = 0 if category == _FAVC_REAL_CATEGORY else 1
        for dirpath, _, fnames in os.walk(cat_path):
            for fname in sorted(fnames):
                if fname.endswith(".npz"):
                    items.append((os.path.join(dirpath, fname), label, cat_idx))
    return items, cat_to_idx


@torch.no_grad()
def extract_repr(model, dataloader, device):
    model.eval()
    model.to(device)
    zc_all, zs_all, lab_all, cat_all = [], [], [], []
    for batch in tqdm(dataloader, desc="Extracting", leave=False):
        v, a, lab, cat = batch
        v = v.unsqueeze(1).to(device)
        a = a.unsqueeze(1).to(device)
        zc, zs, *_ = model._encode(v, a)
        zc_all.append(zc.squeeze(1).cpu().numpy())
        zs_all.append(zs.squeeze(1).cpu().numpy())
        lab_all.extend(lab.numpy().tolist() if hasattr(lab, "numpy") else lab.tolist())
        cat_all.extend(cat.numpy().tolist() if hasattr(cat, "numpy") else cat.tolist())
    return (np.concatenate(zc_all, axis=0),
            np.concatenate(zs_all, axis=0),
            np.array(lab_all, dtype=np.int32),
            np.array(cat_all, dtype=np.int32))


def extract_raw(dataloader):
    raw_all, lab_all, cat_all = [], [], []
    for batch in tqdm(dataloader, desc="Raw", leave=False):
        v, a, lab, cat = batch
        raw_all.append(torch.cat([v, a], dim=-1).numpy())
        lab_all.extend(lab.numpy().tolist() if hasattr(lab, "numpy") else lab.tolist())
        cat_all.extend(cat.numpy().tolist() if hasattr(cat, "numpy") else cat.tolist())
    return (np.concatenate(raw_all, axis=0),
            np.array(lab_all, dtype=np.int32),
            np.array(cat_all, dtype=np.int32))


def probe_cv(features, labels, n_splits=5, seed=42):
    """5-fold stratified CV logistic regression. Returns mean AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(features, labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(features[tr])
        X_te = scaler.transform(features[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X_tr, labels[tr])
        proba = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(labels[te], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def probe_transfer(X_tr, y_tr, X_te, y_te, seed=42):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)
    clf.fit(X_tr_s, y_tr)
    proba = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_te, proba))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--av1m_root", required=True)
    parser.add_argument("--av1m_csv", required=True)
    parser.add_argument("--favc_root", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_av1m", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    apply_l2 = config.get("data_info", {}).get("apply_l2", True)
    model, _ = _load_model_from_ckpt(args.ckpt_path)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Ckpt:        {args.ckpt_path}")
    device = torch.device(args.device)

    # ── AV1M val
    av1m_items = _load_av1m_items(args.av1m_root, args.av1m_csv)
    print(f"\nAV1M val: {len(av1m_items)} clips "
          f"(real={sum(1 for _,l,_ in av1m_items if l==0)}, "
          f"fake={sum(1 for _,l,_ in av1m_items if l==1)})")
    if args.max_av1m and len(av1m_items) > args.max_av1m:
        av1m_items = rng.sample(av1m_items, args.max_av1m)
        print(f"  Subsampled to {len(av1m_items)}")

    # ── FAVC w/ category
    favc_items, cat_to_idx = _load_favc_items(args.favc_root)
    idx_to_cat = {v: k for k, v in cat_to_idx.items()}
    cat_counts = {idx_to_cat[i]: sum(1 for _, _, c in favc_items if c == i)
                  for i in idx_to_cat}
    print("\nFAVC clips by category:")
    for c, n in cat_counts.items():
        print(f"  {c:30s} {n}")

    # ── Dataloaders + extract
    av1m_dl = DataLoader(CategoryDataset(av1m_items, apply_l2), batch_size=64, num_workers=4, shuffle=False)
    favc_dl = DataLoader(CategoryDataset(favc_items, apply_l2), batch_size=64, num_workers=4, shuffle=False)

    print("\nExtracting AV1M Z_c / Z_s …")
    av_zc, av_zs, av_lab, _ = extract_repr(model, av1m_dl, device)
    print("Extracting FAVC Z_c / Z_s …")
    fv_zc, fv_zs, fv_lab, fv_cat = extract_repr(model, favc_dl, device)
    print("Extracting raw AV-HuBERT baselines …")
    av_raw, av_lab2, _ = extract_raw(av1m_dl)
    fv_raw, fv_lab2, fv_cat2 = extract_raw(favc_dl)
    assert (av_lab == av_lab2).all() and (fv_lab == fv_lab2).all() and (fv_cat == fv_cat2).all()

    # ── In-domain CV (sanity)
    print("\n── In-domain probe (AV1M val, 5-fold CV) ──")
    in_dom = {}
    for name, X in [("raw", av_raw), ("zc", av_zc), ("zs", av_zs)]:
        m, s = probe_cv(X, av_lab, args.n_folds, args.seed)
        in_dom[name] = {"auc_mean": m, "auc_std": s}
        print(f"  {name:>3s}  AUC = {m:.4f} ± {s:.4f}")

    # ── Per-subset analysis
    real_mask = (fv_cat == cat_to_idx[_FAVC_REAL_CATEGORY])
    n_real = int(real_mask.sum())
    print(f"\nFAVC real clips available: {n_real}")

    per_subset = {}
    for fake_cat in _FAVC_FAKE_CATEGORIES:
        cat_idx = cat_to_idx[fake_cat]
        fake_mask = (fv_cat == cat_idx)
        n_fake = int(fake_mask.sum())
        if n_fake == 0:
            print(f"\n[skip] {fake_cat}: 0 clips")
            continue
        sel_mask = real_mask | fake_mask
        sub_lab = fv_lab[sel_mask]                  # 0=real, 1=fake
        sub_zc, sub_zs, sub_raw = fv_zc[sel_mask], fv_zs[sel_mask], fv_raw[sel_mask]
        n_sub = int(sel_mask.sum())
        alias = _SUBSET_ALIAS.get(fake_cat, fake_cat)
        print(f"\n── Subset: {fake_cat} ({alias}) ──")
        print(f"  N={n_sub}  (real={n_real}, fake={n_fake})")

        # Label probe — train on AV1M val, test on subset
        lbl = {}
        for name, X_tr, X_te in [("raw", av_raw, sub_raw),
                                 ("zc",  av_zc,  sub_zc),
                                 ("zs",  av_zs,  sub_zs)]:
            auc = probe_transfer(X_tr, av_lab, X_te, sub_lab, args.seed)
            lbl[name] = auc
        print(f"  Label  AUC: raw={lbl['raw']:.4f}  zc={lbl['zc']:.4f}  zs={lbl['zs']:.4f}")

        # Domain probe — AV1M(=0) vs subset(=1), balanced 5-fold CV
        n_dom = min(len(av_lab), n_sub)
        dom_rng = np.random.default_rng(args.seed + cat_idx)
        a_idx = dom_rng.choice(len(av_lab), n_dom, replace=False)
        s_idx = dom_rng.choice(n_sub, n_dom, replace=False)
        dom_lab = np.array([0]*n_dom + [1]*n_dom, dtype=np.int32)
        dom = {}
        for name, X_a, X_s in [("raw", av_raw, sub_raw),
                               ("zc",  av_zc,  sub_zc),
                               ("zs",  av_zs,  sub_zs)]:
            X_dom = np.concatenate([X_a[a_idx], X_s[s_idx]], axis=0)
            m, sd = probe_cv(X_dom, dom_lab, args.n_folds, args.seed)
            dom[name] = {"auc_mean": m, "auc_std": sd}
        print(f"  Domain AUC: raw={dom['raw']['auc_mean']:.4f}  "
              f"zc={dom['zc']['auc_mean']:.4f}  zs={dom['zs']['auc_mean']:.4f}")

        per_subset[fake_cat] = {
            "alias": alias,
            "n_real": n_real,
            "n_fake": n_fake,
            "label_auc": lbl,
            "domain_auc": dom,
        }

    # ── All-FAVC overall (matches existing eval_label_probe.py for sanity)
    print("\n── Overall FAVC (all subsets pooled) ──")
    overall_lbl = {}
    for name, X_tr, X_te in [("raw", av_raw, fv_raw),
                             ("zc",  av_zc,  fv_zc),
                             ("zs",  av_zs,  fv_zs)]:
        overall_lbl[name] = probe_transfer(X_tr, av_lab, X_te, fv_lab, args.seed)
    print(f"  Label  AUC: raw={overall_lbl['raw']:.4f}  "
          f"zc={overall_lbl['zc']:.4f}  zs={overall_lbl['zs']:.4f}")

    n_dom = min(len(av_lab), len(fv_lab))
    dom_rng = np.random.default_rng(args.seed)
    a_idx = dom_rng.choice(len(av_lab), n_dom, replace=False)
    f_idx = dom_rng.choice(len(fv_lab), n_dom, replace=False)
    dom_lab = np.array([0]*n_dom + [1]*n_dom, dtype=np.int32)
    overall_dom = {}
    for name, X_a, X_f in [("raw", av_raw, fv_raw),
                           ("zc",  av_zc,  fv_zc),
                           ("zs",  av_zs,  fv_zs)]:
        X_dom = np.concatenate([X_a[a_idx], X_f[f_idx]], axis=0)
        m, sd = probe_cv(X_dom, dom_lab, args.n_folds, args.seed)
        overall_dom[name] = {"auc_mean": m, "auc_std": sd}
    print(f"  Domain AUC: raw={overall_dom['raw']['auc_mean']:.4f}  "
          f"zc={overall_dom['zc']['auc_mean']:.4f}  zs={overall_dom['zs']['auc_mean']:.4f}")

    # ── Markdown summary table
    print("\n" + "=" * 78)
    print("Summary — Label / Domain AUC by FAVC subset")
    print("=" * 78)
    header = f"| {'Subset':<20s} | {'N':>5s} | {'L:Zc':>6s} | {'L:Zs':>6s} | {'L:raw':>6s} | {'D:Zc':>6s} | {'D:Zs':>6s} | {'D:raw':>6s} |"
    sep    = "|" + "-"*22 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|"
    print(header)
    print(sep)
    for fake_cat in _FAVC_FAKE_CATEGORIES:
        if fake_cat not in per_subset:
            continue
        r = per_subset[fake_cat]
        print(f"| {r['alias']:<20s} | {r['n_real']+r['n_fake']:>5d} "
              f"| {r['label_auc']['zc']:>6.4f} | {r['label_auc']['zs']:>6.4f} | {r['label_auc']['raw']:>6.4f} "
              f"| {r['domain_auc']['zc']['auc_mean']:>6.4f} | {r['domain_auc']['zs']['auc_mean']:>6.4f} | {r['domain_auc']['raw']['auc_mean']:>6.4f} |")
    print(sep)
    print(f"| {'OVERALL (all FAVC)':<20s} | {len(fv_lab):>5d} "
          f"| {overall_lbl['zc']:>6.4f} | {overall_lbl['zs']:>6.4f} | {overall_lbl['raw']:>6.4f} "
          f"| {overall_dom['zc']['auc_mean']:>6.4f} | {overall_dom['zs']['auc_mean']:>6.4f} | {overall_dom['raw']['auc_mean']:>6.4f} |")
    print("=" * 78)
    print("L: = label probe (train AV1M, test subset).  D: = domain probe (AV1M vs subset, 5-fold CV).")

    results = {
        "ckpt_path": args.ckpt_path,
        "config_path": args.config,
        "n_av1m": int(len(av_lab)),
        "in_domain_cv": in_dom,
        "per_subset": per_subset,
        "overall": {"label_auc": overall_lbl, "domain_auc": overall_dom, "n_favc": int(len(fv_lab))},
        "seed": args.seed,
        "n_folds": args.n_folds,
    }
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "label_probe_by_subset.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
