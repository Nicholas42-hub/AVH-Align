"""
Three-way anomaly scoring on FAVC using TriRouteUnsupSync representations.
Implements Caren's three methods (2026-05-03):

  (1) Visual-only anomaly       : score = Mahalanobis(u_v_pool, AV1M-real reference)
  (2) A-V disagreement          : score = 1 - cos(s_v_pool, s_a_pool)
  (3) Residual-gated anomaly    : score_1 - alpha * domain_confidence(r_v, r_a)

Reference distribution: AV1M val clips with label==0 (real).
Output: AUC per method (and grid for alpha) for overall + per-FAVC-subset.

Each clip's features are time-pooled (mean over T) before distance / cosine.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_eval_triroute_unsup_sync import TriRouteUnsupSync


_FAVC_REAL = "RealVideo-RealAudio"
_FAVC_FAKE = ("RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio")
_SUBSET_ALIAS = {
    "RealVideo-FakeAudio": "RV-FA",
    "FakeVideo-RealAudio": "FV-RA",
    "FakeVideo-FakeAudio": "FV-FA",
}


class _Dataset(Dataset):
    def __init__(self, items, apply_l2=True):
        self.items = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, cat = self.items[idx]
        d = np.load(path, allow_pickle=True)
        v = d["visual"].astype(np.float32)
        a = d["audio"].astype(np.float32)
        if self.apply_l2:
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(v), torch.tensor(a), label, cat


def _load_av1m_real_items(av1m_root, csv_path):
    items = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        path_i = header.index("path")
        label_i = header.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            if int(parts[label_i]) != 0:
                continue
            rel = parts[path_i]
            npz = os.path.join(av1m_root, rel.replace(".mp4", ".npz"))
            if os.path.exists(npz):
                items.append((npz, 0, -1))
    return items


def _load_favc_items(favc_root, split_csv=None):
    cat_to_idx = {_FAVC_REAL: 0}
    for i, c in enumerate(_FAVC_FAKE):
        cat_to_idx[c] = i + 1

    allowed_paths = None
    if split_csv is not None:
        allowed_paths = set()
        with open(split_csv) as f:
            header = f.readline().strip().split(",")
            path_col = header.index("full_path")
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) <= path_col:
                    continue
                p = parts[path_col].strip()
                # rel_path under favc_root: strip the FakeAVCeleb/ prefix and replace .mp4 with .npz
                if p.startswith("FakeAVCeleb/"):
                    p = p[len("FakeAVCeleb/"):]
                if p.endswith(".mp4"):
                    p = p[:-4] + ".npz"
                allowed_paths.add(p)

    items = []
    for category in os.listdir(favc_root):
        cat_path = os.path.join(favc_root, category)
        if not os.path.isdir(cat_path) or category not in cat_to_idx:
            continue
        cat_idx = cat_to_idx[category]
        label = 0 if category == _FAVC_REAL else 1
        for dirpath, _, fnames in os.walk(cat_path):
            for fname in sorted(fnames):
                if not fname.endswith(".npz"):
                    continue
                full = os.path.join(dirpath, fname)
                if allowed_paths is not None:
                    rel = os.path.relpath(full, favc_root)
                    if rel not in allowed_paths:
                        continue
                items.append((full, label, cat_idx))
    return items, cat_to_idx


def _collate(batch):
    """Variable-length collation: keep as list of tensors."""
    vs, audios, labs, cats = zip(*batch)
    return list(vs), list(audios), list(labs), list(cats)


@torch.no_grad()
def extract_components(model, loader, device, want_modality_logit=False):
    """
    Returns dict of pooled component tensors:
      u_v_pool: [N, spec_dim]
      s_v_pool: [N, syn_dim]
      s_a_pool: [N, syn_dim]
      r_v_pool: [N, spec_dim]
      r_a_pool: [N, spec_dim]
      mod_logit_rv, mod_logit_ra (if want_modality_logit): [N]
      labels:    [N]  (0=real, 1=fake)
      cats:      [N]
    """
    model.eval()
    model.to(device)
    out = {k: [] for k in ["u_v", "s_v", "s_a", "r_v", "r_a", "u_delta", "labels", "cats"]}
    if want_modality_logit:
        out["mod_logit_rv"] = []
        out["mod_logit_ra"] = []
    for vs, audios, labs, cats in tqdm(loader, desc="extract", leave=False):
        for v, a, lab, cat in zip(vs, audios, labs, cats):
            v = v.unsqueeze(0).to(device)  # [1, T, D]
            a = a.unsqueeze(0).to(device)
            _, comp = model.encode(v, a)
            for k in ["u_v", "s_v", "s_a", "r_v", "r_a", "u_delta"]:
                out[k].append(comp[k].mean(dim=1).squeeze(0).cpu().numpy())
            if want_modality_logit:
                lv = model.modality_head(comp["r_v"].mean(dim=1)).squeeze(-1).cpu().numpy()
                la = model.modality_head(comp["r_a"].mean(dim=1)).squeeze(-1).cpu().numpy()
                out["mod_logit_rv"].append(float(lv))
                out["mod_logit_ra"].append(float(la))
            out["labels"].append(int(lab))
            out["cats"].append(int(cat))
    for k in list(out.keys()):
        out[k] = np.array(out[k])
    return out


def mahalanobis_score(x, mean, inv_cov):
    """x: [N, D], mean: [D], inv_cov: [D, D]. Returns [N] distances."""
    d = x - mean
    return np.einsum("nd,de,ne->n", d, inv_cov, d)


def gaussian_density_score(x, mean, inv_cov, log_det):
    """neg log density (lower = more likely real). Higher = more anomalous."""
    md = mahalanobis_score(x, mean, inv_cov)
    return 0.5 * md + 0.5 * log_det  # constant factor cancels for AUC ranking


def knn_score(x, ref, k=10):
    """Mean distance to top-k nearest reference. x: [N, D], ref: [M, D]. Returns [N]."""
    # batched cdist for memory
    out = np.zeros(x.shape[0], dtype=np.float32)
    bs = 256
    for i in range(0, x.shape[0], bs):
        chunk = x[i:i+bs]
        d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        topk = np.partition(d2, k, axis=1)[:, :k]
        out[i:i+bs] = np.sqrt(np.maximum(topk, 0)).mean(axis=1)
    return out


def per_subset_auc(scores, labels, cats, cat_to_idx):
    """Returns dict {subset: auc} including 'overall'."""
    res = {}
    res["overall"] = roc_auc_score(labels, scores) if len(set(labels)) == 2 else None
    real_mask = (cats == cat_to_idx[_FAVC_REAL])
    for fc in _FAVC_FAKE:
        ci = cat_to_idx[fc]
        sel = real_mask | (cats == ci)
        if sel.sum() == 0:
            continue
        sub_lab = labels[sel]
        sub_score = scores[sel]
        if len(set(sub_lab)) < 2:
            res[_SUBSET_ALIAS[fc]] = None
            continue
        res[_SUBSET_ALIAS[fc]] = roc_auc_score(sub_lab, sub_score)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--av1m_root", default="/data/projects/punim2637/nnliang/AVH-Align/data/avh_features/val")
    p.add_argument("--av1m_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m/val_labels.csv")
    p.add_argument("--favc_root", default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    p.add_argument("--favc_split_csv", default=None,
                   help="If provided, restrict FAVC eval to clips listed under 'full_path' in this CSV")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_av1m", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--shrinkage", type=float, default=0.05, help="Ledoit-Wolf shrinkage")
    p.add_argument("--alphas", type=str, default="0.0,0.3,0.5,0.7,1.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading ckpt: {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TriRouteUnsupSync(cfg)
    model.load_state_dict(ck["state_dict"])
    print(f"  ckpt epoch={ck.get('epoch')} overall_auc={ck.get('overall_auc', 'n/a')}", flush=True)

    # ── AV1M-real reference ──────────────────────────────────────────────────
    av_items = _load_av1m_real_items(args.av1m_root, args.av1m_csv)
    if args.max_av1m and len(av_items) > args.max_av1m:
        av_items = av_items[:args.max_av1m]
    print(f"AV1M-real reference clips: {len(av_items)}", flush=True)
    av_ds = _Dataset(av_items)
    av_dl = DataLoader(av_ds, batch_size=8, num_workers=args.num_workers,
                       shuffle=False, collate_fn=_collate)
    av = extract_components(model, av_dl, device, want_modality_logit=False)
    u_v_ref = av["u_v"]                            # [Nref, D]
    s_v_ref = av["s_v"]                            # [Nref, D]
    s_a_ref = av["s_a"]                            # [Nref, D]

    # Reference Gaussian on u_v
    mu = u_v_ref.mean(axis=0)
    cov = np.cov(u_v_ref.T) + args.shrinkage * np.trace(np.cov(u_v_ref.T)) / u_v_ref.shape[1] * np.eye(u_v_ref.shape[1])
    inv_cov = np.linalg.pinv(cov)
    sign, log_det = np.linalg.slogdet(cov)

    print(f"u_v reference: mu shape {mu.shape}, cov shape {cov.shape}", flush=True)

    # ── FAVC test ────────────────────────────────────────────────────────────
    fv_items, cat_to_idx = _load_favc_items(args.favc_root, args.favc_split_csv)
    if args.favc_split_csv:
        print(f"FAVC clips (filtered to split {args.favc_split_csv}): {len(fv_items)}", flush=True)
    else:
        print(f"FAVC clips (all): {len(fv_items)}", flush=True)
    fv_ds = _Dataset(fv_items)
    fv_dl = DataLoader(fv_ds, batch_size=8, num_workers=args.num_workers,
                       shuffle=False, collate_fn=_collate)
    fv = extract_components(model, fv_dl, device, want_modality_logit=True)
    labels = fv["labels"]
    cats = fv["cats"]

    # ── Method 1: visual-only anomaly (Mahalanobis / kNN / Gaussian) ─────────
    print("\n[Method 1] Visual-only anomaly (u_v vs AV1M-real reference)", flush=True)
    s1_mahal = mahalanobis_score(fv["u_v"], mu, inv_cov)
    s1_gauss = gaussian_density_score(fv["u_v"], mu, inv_cov, log_det)
    s1_knn   = knn_score(fv["u_v"], u_v_ref, k=args.knn_k)
    res1_mahal = per_subset_auc(s1_mahal, labels, cats, cat_to_idx)
    res1_gauss = per_subset_auc(s1_gauss, labels, cats, cat_to_idx)
    res1_knn   = per_subset_auc(s1_knn,   labels, cats, cat_to_idx)
    print(f"  Mahalanobis : {res1_mahal}")
    print(f"  Gaussian-LL : {res1_gauss}")
    print(f"  kNN (k={args.knn_k:>2d}): {res1_knn}")

    # ── Method 2: A-V disagreement ───────────────────────────────────────────
    print("\n[Method 2] A-V disagreement (1 - cos(s_v, s_a))", flush=True)
    sv = fv["s_v"]; sa = fv["s_a"]
    cos_sv_sa = (sv * sa).sum(-1) / (np.linalg.norm(sv, axis=-1) * np.linalg.norm(sa, axis=-1) + 1e-8)
    s2_cos = 1.0 - cos_sv_sa
    s2_l2 = np.linalg.norm(sv - sa, axis=-1)
    res2_cos = per_subset_auc(s2_cos, labels, cats, cat_to_idx)
    res2_l2  = per_subset_auc(s2_l2,  labels, cats, cat_to_idx)
    print(f"  1-cos(s_v, s_a): {res2_cos}")
    print(f"  L2(s_v, s_a)   : {res2_l2}")

    # ── Method 3: residual-gated anomaly ─────────────────────────────────────
    # domain_confidence(Z_res) = how decisive modality_head is on r_v / r_a
    # confidence in correct modality direction:
    #   r_v should classify as "video" (label 0 → logit < 0)
    #   r_a should classify as "audio" (label 1 → logit > 0)
    # define: dom_conf = sigmoid(-mod_logit_rv) + sigmoid(mod_logit_ra) - 1   (in [-1,1], higher = more domain leakage)
    # we re-scale to [0, 1].
    sig_rv = 1.0 / (1.0 + np.exp(fv["mod_logit_rv"]))    # P(video|r_v)
    sig_ra = 1.0 / (1.0 + np.exp(-fv["mod_logit_ra"]))   # P(audio|r_a)
    dom_conf = (sig_rv + sig_ra) / 2.0                    # [0,1]
    # z-normalise base anomaly score for fair gating
    base_z = (s1_mahal - s1_mahal.mean()) / (s1_mahal.std() + 1e-8)
    dom_z  = (dom_conf - dom_conf.mean()) / (dom_conf.std() + 1e-8)

    print("\n[Method 3] Residual-gated anomaly (visual_anomaly - alpha * domain_conf)", flush=True)
    res3_grid = {}
    alphas = [float(x) for x in args.alphas.split(",")]
    for alpha in alphas:
        s3 = base_z - alpha * dom_z
        res3_grid[f"alpha={alpha}"] = per_subset_auc(s3, labels, cats, cat_to_idx)
        print(f"  alpha={alpha:>4.2f} : {res3_grid[f'alpha={alpha}']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "ckpt_path": args.ckpt,
        "ckpt_epoch": ck.get("epoch"),
        "ckpt_overall_auc": ck.get("overall_auc"),
        "n_av1m_real_ref": len(av_items),
        "n_favc": len(fv_items),
        "method_1_visual_only": {
            "mahalanobis": res1_mahal,
            "gaussian_loglik": res1_gauss,
            f"knn_k{args.knn_k}": res1_knn,
        },
        "method_2_av_disagreement": {
            "1_minus_cos": res2_cos,
            "l2": res2_l2,
        },
        "method_3_residual_gated": res3_grid,
    }
    out = os.path.join(args.output_dir, "anomaly_three_ways.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved: {out}", flush=True)

    # Markdown table
    print("\n" + "=" * 78)
    print("Three-way anomaly summary (FAVC overall AUC)")
    print("=" * 78)
    rows = [
        ("M1 Mahalanobis (u_v)",    res1_mahal),
        ("M1 Gaussian-LL (u_v)",    res1_gauss),
        (f"M1 kNN-{args.knn_k} (u_v)",     res1_knn),
        ("M2 1-cos(s_v, s_a)",      res2_cos),
        ("M2 L2(s_v, s_a)",         res2_l2),
    ]
    for alpha in alphas:
        rows.append((f"M3 α={alpha}",       res3_grid[f"alpha={alpha}"]))
    print(f"{'Method':<28s} {'overall':>8s} {'RV-FA':>8s} {'FV-RA':>8s} {'FV-FA':>8s}")
    print("-" * 78)
    for name, r in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  n/a "
        print(f"{name:<28s} {fmt(r.get('overall')):>8s} {fmt(r.get('RV-FA')):>8s} {fmt(r.get('FV-RA')):>8s} {fmt(r.get('FV-FA')):>8s}")
    print("=" * 78)


if __name__ == "__main__":
    main()
