"""
Caren's Method 1 + Method 2 evaluation on FAVC test (70/30 split, N=6667).

Method 1 — Late-fusion variants:
  (1.0) kNN unsupervised (baseline, already in three_ways)
  (1a)  TriRoute + kNN rank fusion       — avg(rank(sup), rank(kNN))
  (1b)  TriRoute + Mahalanobis rank fusion — avg(rank(sup), rank(Mahal))
  (1c)  TriRoute + sync-head rank fusion  — avg(rank(sup), rank(sync))

Method 2 — Residual-gated late fusion:
  gate = sigmoid(α · ||Z_res||)
  final = (1 - gate) · TriRoute_score + gate · unsup_kNN_score
  α grid: {0.5, 1.0, 2.0, 5.0}

Inputs:
  - Supervised TriRoute A6 (paper-clean ckpt, policy-compliant)
  - Unsupervised TriRoute (vox2-trained, seed43 best_auc.pt = 0.918)
  - AV1M val real-only as kNN reference for Mahalanobis / kNN-on-u_v
  - FAVC 70/30 test split (6667 clips)
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
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mlp_fcd_a6 import AVH_FCD_A6
from train_eval_triroute_unsup_sync import TriRouteUnsupSync, score_clip as _sync_score_clip


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


def _load_favc_items(favc_root, split_csv):
    cat_to_idx = {_FAVC_REAL: 0}
    for i, c in enumerate(_FAVC_FAKE):
        cat_to_idx[c] = i + 1
    allowed = set()
    with open(split_csv) as f:
        header = f.readline().strip().split(",")
        path_col = header.index("full_path")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= path_col:
                continue
            p = parts[path_col].strip()
            if p.startswith("FakeAVCeleb/"):
                p = p[len("FakeAVCeleb/"):]
            if p.endswith(".mp4"):
                p = p[:-4] + ".npz"
            allowed.add(p)
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
                rel = os.path.relpath(full, favc_root)
                if rel not in allowed:
                    continue
                items.append((full, label, cat_idx))
    return items, cat_to_idx


def _collate(batch):
    vs, audios, labs, cats = zip(*batch)
    return list(vs), list(audios), list(labs), list(cats)


@torch.no_grad()
def extract_sup(model, loader, device):
    """Returns dict with sup score, sup logit, ||Z_res||, u_v, etc."""
    model.eval()
    model.to(device)
    out = {k: [] for k in ["sup_score", "sup_logit", "z_res_norm",
                            "labels", "cats"]}
    for vs, audios, labs, cats in tqdm(loader, desc="sup", leave=False):
        for v, a, lab, cat in zip(vs, audios, labs, cats):
            v = v.unsqueeze(0).to(device)
            a = a.unsqueeze(0).to(device)
            causal_repr, spurious_repr, _ = model._encode(v, a)
            task_repr = model._task_repr(causal_repr, spurious_repr)
            logit = model._cls_score(model.causal_head, task_repr).squeeze().cpu().numpy()
            score = float(torch.sigmoid(torch.tensor(float(logit))))
            z_res_pool = spurious_repr.mean(dim=1)  # [1, D]
            z_res_norm = float(torch.linalg.norm(z_res_pool, dim=-1).item())
            out["sup_score"].append(score)
            out["sup_logit"].append(float(logit))
            out["z_res_norm"].append(z_res_norm)
            out["labels"].append(int(lab))
            out["cats"].append(int(cat))
    return {k: np.array(v) for k, v in out.items()}


@torch.no_grad()
def extract_unsup(model, loader, device, want_components=True):
    """
    Returns u_v (pooled), sync_score (the ORIGINAL training-time readout
    = score_clip from train_eval_triroute_unsup_sync.py), labels, cats.
    """
    model.eval()
    model.to(device)
    out = {k: [] for k in ["u_v", "sync_score", "labels", "cats"]}
    for vs, audios, labs, cats in tqdm(loader, desc="unsup", leave=False):
        for v, a, lab, cat in zip(vs, audios, labs, cats):
            v_in = v.unsqueeze(0).to(device)  # [1, T, D]
            a_in = a.unsqueeze(0).to(device)
            _, comp = model.encode(v_in, a_in)
            u_v_pool = comp["u_v"].mean(dim=1).squeeze(0).cpu().numpy()
            out["u_v"].append(u_v_pool)
            # use the ORIGINAL training-time score_clip to match the 0.918
            # cross_domain_history number; this passes (video, audio) directly
            # to model.logits and applies logsumexp(-logits) over the time axis.
            try:
                sync_list = _sync_score_clip(model, v_in, a_in, device)
                sync = float(sync_list[0]) if isinstance(sync_list, list) else float(sync_list)
            except Exception as e:
                print(f"WARN sync_score failed: {e}", flush=True)
                sync = 0.0
            out["sync_score"].append(sync)
            out["labels"].append(int(lab))
            out["cats"].append(int(cat))
    return {
        "u_v": np.array(out["u_v"]),
        "sync_score": np.array(out["sync_score"]),
        "labels": np.array(out["labels"]),
        "cats": np.array(out["cats"]),
    }


def compute_unsup_anomaly_scores(u_v_test, u_v_ref, knn_k=10, shrinkage=0.05):
    """Mahalanobis + kNN distance vs AV1M-real reference."""
    mu = u_v_ref.mean(axis=0)
    cov = np.cov(u_v_ref.T)
    cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    diff = u_v_test - mu
    mahal = np.einsum("nd,de,ne->n", diff, inv_cov, diff)
    # kNN
    knn = np.zeros(u_v_test.shape[0], dtype=np.float32)
    bs = 256
    for i in range(0, u_v_test.shape[0], bs):
        chunk = u_v_test[i:i+bs]
        d2 = ((chunk[:, None, :] - u_v_ref[None, :, :]) ** 2).sum(-1)
        topk = np.partition(d2, knn_k, axis=1)[:, :knn_k]
        knn[i:i+bs] = np.sqrt(np.maximum(topk, 0)).mean(axis=1)
    return mahal, knn


def per_subset_auc(scores, labels, cats, cat_to_idx):
    res = {}
    res["overall"] = roc_auc_score(labels, scores) if len(set(labels)) == 2 else None
    real_mask = (cats == cat_to_idx[_FAVC_REAL])
    for fc in _FAVC_FAKE:
        sel = real_mask | (cats == cat_to_idx[fc])
        if sel.sum() == 0:
            continue
        sub_lab = labels[sel]
        sub_score = scores[sel]
        if len(set(sub_lab)) < 2:
            res[_SUBSET_ALIAS[fc]] = None
            continue
        res[_SUBSET_ALIAS[fc]] = roc_auc_score(sub_lab, sub_score)
    return res


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sup_ckpt", required=True)
    p.add_argument("--sup_config", required=True)
    p.add_argument("--unsup_ckpt", required=True)
    p.add_argument("--av1m_root", default="/data/projects/punim2637/nnliang/AVH-Align/data/avh_features/val")
    p.add_argument("--av1m_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m/val_labels.csv")
    p.add_argument("--favc_root", default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    p.add_argument("--favc_split_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical/test_split.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--betas", default="0.0,0.5,1.0,2.0")
    p.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load supervised A6 ──────────────────────────────────────────────────
    print(f"Loading sup: {args.sup_ckpt}", flush=True)
    with open(args.sup_config) as f:
        sup_cfg = yaml.safe_load(f)
    sup_model = AVH_FCD_A6(config=sup_cfg)
    sup_ck = torch.load(args.sup_ckpt, map_location="cpu", weights_only=False)
    sd = sup_ck.get("state_dict", sup_ck)
    sup_model.load_state_dict({k: v for k, v in sd.items()
                                if k in sup_model.state_dict() and v.shape == sup_model.state_dict()[k].shape},
                               strict=False)

    # ── Load unsup TriRoute ────────────────────────────────────────────────
    print(f"Loading unsup: {args.unsup_ckpt}", flush=True)
    un_ck = torch.load(args.unsup_ckpt, map_location="cpu", weights_only=False)
    unsup_model = TriRouteUnsupSync(un_ck["config"])
    unsup_model.load_state_dict(un_ck["state_dict"])
    print(f"  unsup ckpt epoch={un_ck.get('epoch')} overall_auc={un_ck.get('overall_auc')}", flush=True)

    apply_l2 = sup_cfg.get("data_info", {}).get("apply_l2", True)

    # ── AV1M-real reference for kNN on u_v ─────────────────────────────────
    av_items = _load_av1m_real_items(args.av1m_root, args.av1m_csv)
    print(f"AV1M-real ref clips: {len(av_items)}", flush=True)
    av_ds = _Dataset(av_items, apply_l2)
    av_dl = DataLoader(av_ds, batch_size=8, num_workers=4, collate_fn=_collate, shuffle=False)
    av_unsup = extract_unsup(unsup_model, av_dl, device)
    u_v_ref = av_unsup["u_v"]

    # ── FAVC test ──────────────────────────────────────────────────────────
    fv_items, cat_to_idx = _load_favc_items(args.favc_root, args.favc_split_csv)
    print(f"FAVC test clips: {len(fv_items)}", flush=True)
    fv_ds = _Dataset(fv_items, apply_l2)
    fv_dl = DataLoader(fv_ds, batch_size=8, num_workers=4, collate_fn=_collate, shuffle=False)

    print("\nExtracting sup features on FAVC...", flush=True)
    sup_out = extract_sup(sup_model, fv_dl, device)
    print("Extracting unsup features on FAVC...", flush=True)
    un_out = extract_unsup(unsup_model, fv_dl, device)
    assert (sup_out["labels"] == un_out["labels"]).all() and (sup_out["cats"] == un_out["cats"]).all()
    labels = sup_out["labels"]
    cats = sup_out["cats"]

    # ── Unsup anomaly scores from u_v vs reference ────────────────────────
    mahal_score, knn_score = compute_unsup_anomaly_scores(un_out["u_v"], u_v_ref, knn_k=args.knn_k)
    sync_score = un_out["sync_score"]

    # ── Baselines (for context) ───────────────────────────────────────────
    res = {}
    res["sup_alone"]      = per_subset_auc(sup_out["sup_score"], labels, cats, cat_to_idx)
    res["unsup_kNN"]      = per_subset_auc(knn_score,            labels, cats, cat_to_idx)
    res["unsup_Mahal"]    = per_subset_auc(mahal_score,          labels, cats, cat_to_idx)
    res["unsup_sync"]     = per_subset_auc(sync_score,           labels, cats, cat_to_idx)

    # ── Method 1: rank fusion variants ────────────────────────────────────
    sup_rank   = rankdata(sup_out["sup_score"])
    knn_rank   = rankdata(knn_score)
    mahal_rank = rankdata(mahal_score)
    sync_rank  = rankdata(sync_score)

    res["1a_sup_kNN_rank"]    = per_subset_auc((sup_rank + knn_rank)   / 2, labels, cats, cat_to_idx)
    res["1b_sup_Mahal_rank"]  = per_subset_auc((sup_rank + mahal_rank) / 2, labels, cats, cat_to_idx)
    res["1c_sup_sync_rank"]   = per_subset_auc((sup_rank + sync_rank)  / 2, labels, cats, cat_to_idx)

    # ── Method 1 alt: sigmoid logit-fusion (β grid) ───────────────────────
    knn_z = normalize(knn_score)
    res_logit_grid = {}
    for beta in [float(b) for b in args.betas.split(",")]:
        s = 1.0 / (1.0 + np.exp(-(sup_out["sup_logit"] + beta * knn_z)))
        res_logit_grid[f"beta={beta}"] = per_subset_auc(s, labels, cats, cat_to_idx)
    res["1_logit_fusion_grid"] = res_logit_grid

    # ── Method 2: residual-gated late fusion ──────────────────────────────
    z_res_norm_z = normalize(sup_out["z_res_norm"])
    knn_z_safe   = normalize(knn_score)  # both in z-score for fair gating
    sup_z        = normalize(sup_out["sup_score"])
    res2_grid = {}
    for alpha in [float(a) for a in args.alphas.split(",")]:
        gate = 1.0 / (1.0 + np.exp(-alpha * z_res_norm_z))
        s = (1.0 - gate) * sup_z + gate * knn_z_safe
        res2_grid[f"alpha={alpha}"] = per_subset_auc(s, labels, cats, cat_to_idx)
    res["2_residual_gated_grid"] = res2_grid

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "caren_methods_12.json")
    with open(out_path, "w") as f:
        json.dump({
            "sup_ckpt": args.sup_ckpt,
            "unsup_ckpt": args.unsup_ckpt,
            "n_av1m_ref": len(av_items),
            "n_favc": len(fv_items),
            "results": res,
        }, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Caren Methods 1 + 2 — FAVC overall AUC (test split, N=6667)")
    print("=" * 78)
    rows = [
        ("Sup A6 alone",              res["sup_alone"]),
        ("Unsup kNN (M1.0)",          res["unsup_kNN"]),
        ("Unsup Mahal",               res["unsup_Mahal"]),
        ("Unsup sync",                res["unsup_sync"]),
        ("Rank fusion: Sup+kNN (1a)", res["1a_sup_kNN_rank"]),
        ("Rank fusion: Sup+Mahal(1b)", res["1b_sup_Mahal_rank"]),
        ("Rank fusion: Sup+sync (1c)", res["1c_sup_sync_rank"]),
    ]
    for beta in [float(b) for b in args.betas.split(",")]:
        rows.append((f"Logit fusion β={beta}", res_logit_grid[f"beta={beta}"]))
    for alpha in [float(a) for a in args.alphas.split(",")]:
        rows.append((f"Residual-gate α={alpha}", res2_grid[f"alpha={alpha}"]))
    print(f"{'Method':<32s} {'overall':>8s} {'RV-FA':>8s} {'FV-RA':>8s} {'FV-FA':>8s}")
    print("-" * 78)
    for name, r in rows:
        def fmt(v): return f"{v:.4f}" if v is not None else "  n/a "
        print(f"{name:<32s} {fmt(r.get('overall')):>8s} {fmt(r.get('RV-FA')):>8s} {fmt(r.get('FV-RA')):>8s} {fmt(r.get('FV-FA')):>8s}")
    print("=" * 78)


if __name__ == "__main__":
    main()
