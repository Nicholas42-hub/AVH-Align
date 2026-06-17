"""
TriRoute + kNN fusion with source-calibrated gates.

This keeps the original Method 1a outputs, then adds source-safe late-fusion
variants whose kNN branch is gated by the AV1M-real reference self-kNN
distribution. The intent is to keep the gate mostly closed on AV1M-like clips
while allowing kNN anomaly evidence to help on shifted FAVC clips.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mlp_fcd_a6 import AVH_FCD_A6
from train_eval_triroute_unsup_sync import TriRouteUnsupSync, score_clip as _sync_score_clip


_FAVC_REAL = "RealVideo-RealAudio"
_FAVC_FAKE = ("RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio")
_FAVC_ALIAS = {
    "RealVideo-FakeAudio": "RV-FA",
    "FakeVideo-RealAudio": "FV-RA",
    "FakeVideo-FakeAudio": "FV-FA",
}

# AV1M categories are encoded in the filename of each clip
_AV1M_FNAME_TO_CAT = {
    "real": "real",
    "real_video_fake_audio": "RV-FA",
    "fake_video_real_audio": "FV-RA",
    "fake_video_fake_audio": "FV-FA",
}
_AV1M_FAKE_ALIASES = ("RV-FA", "FV-RA", "FV-FA")


class _Dataset(Dataset):
    def __init__(self, items, apply_l2=True):
        self.items = items
        self.apply_l2 = apply_l2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, cat_idx, key = self.items[idx]
        d = np.load(path, allow_pickle=True)
        v = d["visual"].astype(np.float32)
        a = d["audio"].astype(np.float32)
        if self.apply_l2:
            v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
            a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(v), torch.tensor(a), label, cat_idx, key


def _collate(batch):
    vs, audios, labs, cats, keys = zip(*batch)
    return list(vs), list(audios), list(labs), list(cats), list(keys)


# ─── AV1M loaders ──────────────────────────────────────────────────────────────

def _load_av1m_real_items(av1m_root, csv_path):
    """AV1M val real-only, used as the kNN reference."""
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
            npz_rel = rel.replace(".mp4", ".npz")
            npz = os.path.join(av1m_root, "val", npz_rel)
            if os.path.exists(npz):
                items.append((npz, 0, -1, npz_rel))
    return items


def _load_av1m_full_items(av1m_root, csv_path):
    """AV1M val full (real + fake) for in-domain eval; category from filename."""
    cat_to_idx = {"real": 0, "RV-FA": 1, "FV-RA": 2, "FV-FA": 3}
    items = []
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        path_i = header.index("path")
        label_i = header.index("label")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            label = int(parts[label_i])
            rel = parts[path_i]
            stem = os.path.splitext(os.path.basename(rel))[0]
            if stem not in _AV1M_FNAME_TO_CAT:
                continue
            cat = _AV1M_FNAME_TO_CAT[stem]
            cat_idx = cat_to_idx[cat]
            npz_rel = rel.replace(".mp4", ".npz")
            npz = os.path.join(av1m_root, "val", npz_rel)
            if not os.path.exists(npz):
                continue
            items.append((npz, label, cat_idx, npz_rel))
    return items, cat_to_idx


# ─── FAVC loader ───────────────────────────────────────────────────────────────

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
                items.append((full, label, cat_idx, rel))
    return items, cat_to_idx


# ─── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_sup(model, loader, device):
    model.eval()
    model.to(device)
    out = {k: [] for k in ["sup_score", "sup_logit", "z_res_norm", "labels", "cats", "keys"]}
    for vs, audios, labs, cats, keys in tqdm(loader, desc="sup", leave=False):
        for v, a, lab, cat, key in zip(vs, audios, labs, cats, keys):
            v = v.unsqueeze(0).to(device)
            a = a.unsqueeze(0).to(device)
            causal_repr, spurious_repr, _ = model._encode(v, a)
            task_repr = model._task_repr(causal_repr, spurious_repr)
            logit = model._cls_score(model.causal_head, task_repr).squeeze().cpu().numpy()
            score = float(torch.sigmoid(torch.tensor(float(logit))))
            z_res_pool = spurious_repr.mean(dim=1)
            z_res_norm = float(torch.linalg.norm(z_res_pool, dim=-1).item())
            out["sup_score"].append(score)
            out["sup_logit"].append(float(logit))
            out["z_res_norm"].append(z_res_norm)
            out["labels"].append(int(lab))
            out["cats"].append(int(cat))
            out["keys"].append(key)
    arr = {k: np.array(v) for k, v in out.items() if k != "keys"}
    arr["keys"] = out["keys"]
    return arr


@torch.no_grad()
def extract_unsup(model, loader, device):
    model.eval()
    model.to(device)
    out = {k: [] for k in ["u_v", "sync_score", "labels", "cats", "keys"]}
    for vs, audios, labs, cats, keys in tqdm(loader, desc="unsup", leave=False):
        for v, a, lab, cat, key in zip(vs, audios, labs, cats, keys):
            v_in = v.unsqueeze(0).to(device)
            a_in = a.unsqueeze(0).to(device)
            _, comp = model.encode(v_in, a_in)
            u_v_pool = comp["u_v"].mean(dim=1).squeeze(0).cpu().numpy()
            out["u_v"].append(u_v_pool)
            try:
                sync_list = _sync_score_clip(model, v_in, a_in, device)
                sync = float(sync_list[0]) if isinstance(sync_list, list) else float(sync_list)
            except Exception as e:
                print(f"WARN sync_score failed: {e}", flush=True)
                sync = 0.0
            out["sync_score"].append(sync)
            out["labels"].append(int(lab))
            out["cats"].append(int(cat))
            out["keys"].append(key)
    return {
        "u_v": np.array(out["u_v"]),
        "sync_score": np.array(out["sync_score"]),
        "labels": np.array(out["labels"]),
        "cats": np.array(out["cats"]),
        "keys": out["keys"],
    }


# ─── Anomaly scores ────────────────────────────────────────────────────────────

def compute_unsup_anomaly_scores(u_v_test, u_v_ref, knn_k=10, shrinkage=0.05,
                                 self_exclude_mask=None):
    """
    Mahalanobis + kNN distance vs AV1M-real reference.

    self_exclude_mask : optional bool array of shape [N_test] indicating which
        test points are themselves contained in u_v_ref. For those points we
        pull k+1 nearest neighbours and drop the closest one (which is the
        self match at distance ~0).
    """
    mu = u_v_ref.mean(axis=0)
    cov = np.cov(u_v_ref.T)
    cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    diff = u_v_test - mu
    mahal = np.einsum("nd,de,ne->n", diff, inv_cov, diff)

    knn = np.zeros(u_v_test.shape[0], dtype=np.float32)
    bs = 256
    if self_exclude_mask is None:
        self_exclude_mask = np.zeros(u_v_test.shape[0], dtype=bool)
    for i in range(0, u_v_test.shape[0], bs):
        chunk = u_v_test[i:i + bs]
        d2 = ((chunk[:, None, :] - u_v_ref[None, :, :]) ** 2).sum(-1)
        # always pull k+1 neighbours so we can drop the self-match if needed
        kk = min(knn_k + 1, d2.shape[1])
        topk = np.partition(d2, kk - 1, axis=1)[:, :kk]
        topk_sorted = np.sort(topk, axis=1)
        for j in range(chunk.shape[0]):
            if self_exclude_mask[i + j]:
                d_use = topk_sorted[j, 1:knn_k + 1]
            else:
                d_use = topk_sorted[j, :knn_k]
            knn[i + j] = np.sqrt(np.maximum(d_use, 0)).mean()
    return mahal, knn


# ─── Per-subset metrics (AUC + AP) ─────────────────────────────────────────────

def _safe_metrics(labels, scores):
    if len(set(labels)) < 2:
        return {"AUC": None, "AP": None}
    return {
        "AUC": float(roc_auc_score(labels, scores)),
        "AP": float(average_precision_score(labels, scores)),
    }


def per_subset_auc_favc(scores, labels, cats, cat_to_idx):
    res = {"overall": _safe_metrics(labels, scores)}
    real_mask = (cats == cat_to_idx[_FAVC_REAL])
    for fc in _FAVC_FAKE:
        sel = real_mask | (cats == cat_to_idx[fc])
        if sel.sum() == 0:
            continue
        res[_FAVC_ALIAS[fc]] = _safe_metrics(labels[sel], scores[sel])
    return res


def per_subset_auc_av1m(scores, labels, cats, cat_to_idx):
    res = {"overall": _safe_metrics(labels, scores)}
    real_mask = (cats == cat_to_idx["real"])
    for alias in _AV1M_FAKE_ALIASES:
        sel = real_mask | (cats == cat_to_idx[alias])
        if sel.sum() == 0:
            continue
        res[alias] = _safe_metrics(labels[sel], scores[sel])
    return res


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sup_ckpt", required=True)
    p.add_argument("--sup_config", required=True)
    p.add_argument("--unsup_ckpt", required=True)
    p.add_argument("--eval_dataset", choices=["favc", "av1m"], required=True)
    p.add_argument("--eval_features_root", required=True,
                   help="favc: data/favc_features[_trimmed]; av1m: dir whose val/ holds .npz files")
    p.add_argument("--ref_features_root", required=True,
                   help="dir whose val/ holds AV1M val-real .npz files (kNN reference)")
    p.add_argument("--av1m_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m/val_labels.csv")
    p.add_argument("--favc_split_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical/test_split.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--betas", default="0.0,0.5,1.0,2.0")
    p.add_argument("--alphas", default="0.5,1.0,2.0,5.0")
    p.add_argument("--gate_betas", default="0.25,0.5,1.0,2.0")
    p.add_argument("--gate_alphas", default="1.0,2.0,5.0")
    p.add_argument("--gate_quantiles", default="0.90,0.95,0.975")
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

    # ── kNN reference: AV1M val-real from ref_features_root ────────────────
    av_items = _load_av1m_real_items(args.ref_features_root, args.av1m_csv)
    print(f"AV1M-real ref clips ({args.ref_features_root}): {len(av_items)}", flush=True)
    av_ds = _Dataset(av_items, apply_l2)
    av_dl = DataLoader(av_ds, batch_size=8, num_workers=4, collate_fn=_collate, shuffle=False)
    av_unsup = extract_unsup(unsup_model, av_dl, device)
    u_v_ref = av_unsup["u_v"]
    ref_keys = set(av_unsup["keys"])

    # Source calibration: estimate what "normal" kNN distance looks like on
    # AV1M real clips. Later gates are defined from this reference distribution,
    # not from target labels.
    _, knn_ref_self = compute_unsup_anomaly_scores(
        u_v_ref,
        u_v_ref,
        knn_k=args.knn_k,
        self_exclude_mask=np.ones(u_v_ref.shape[0], dtype=bool),
    )
    knn_ref_mu = float(knn_ref_self.mean())
    knn_ref_std = float(knn_ref_self.std() + 1e-8)
    knn_ref_z_self = (knn_ref_self - knn_ref_mu) / knn_ref_std
    gate_thresholds = {
        str(q): float(np.quantile(knn_ref_z_self, q))
        for q in parse_float_list(args.gate_quantiles)
    }

    # ── Eval set ───────────────────────────────────────────────────────────
    if args.eval_dataset == "favc":
        ev_items, cat_to_idx = _load_favc_items(args.eval_features_root, args.favc_split_csv)
        per_subset = per_subset_auc_favc
    else:
        ev_items, cat_to_idx = _load_av1m_full_items(args.eval_features_root, args.av1m_csv)
        per_subset = per_subset_auc_av1m

    print(f"Eval set [{args.eval_dataset}] clips ({args.eval_features_root}): {len(ev_items)}", flush=True)
    ev_ds = _Dataset(ev_items, apply_l2)
    ev_dl = DataLoader(ev_ds, batch_size=8, num_workers=4, collate_fn=_collate, shuffle=False)

    print("\nExtracting sup features...", flush=True)
    sup_out = extract_sup(sup_model, ev_dl, device)
    print("Extracting unsup features...", flush=True)
    un_out = extract_unsup(unsup_model, ev_dl, device)
    assert (sup_out["labels"] == un_out["labels"]).all() and (sup_out["cats"] == un_out["cats"]).all()
    labels = sup_out["labels"]
    cats = sup_out["cats"]

    # ── Self-exclusion mask: when the eval is AV1M val and the ref is the
    #    same trimmed status, val-real items appear in both ref and test;
    #    drop the self-match (closest distance = 0) when computing kNN.
    test_keys = un_out["keys"]
    self_exclude = np.array([k in ref_keys for k in test_keys], dtype=bool)
    n_self = int(self_exclude.sum())
    print(f"Self-exclude mask: {n_self}/{len(test_keys)} test points overlap the ref", flush=True)

    # ── Anomaly scores from u_v vs reference ──────────────────────────────
    mahal_score, knn_score = compute_unsup_anomaly_scores(
        un_out["u_v"], u_v_ref, knn_k=args.knn_k, self_exclude_mask=self_exclude
    )
    sync_score = un_out["sync_score"]

    # ── Baselines ──────────────────────────────────────────────────────────
    res = {}
    res["sup_alone"] = per_subset(sup_out["sup_score"], labels, cats, cat_to_idx)
    res["unsup_kNN"] = per_subset(knn_score, labels, cats, cat_to_idx)
    res["unsup_Mahal"] = per_subset(mahal_score, labels, cats, cat_to_idx)
    res["unsup_sync"] = per_subset(sync_score, labels, cats, cat_to_idx)

    # ── Method 1: rank fusion ─────────────────────────────────────────────
    sup_rank = rankdata(sup_out["sup_score"])
    knn_rank = rankdata(knn_score)
    mahal_rank = rankdata(mahal_score)
    sync_rank = rankdata(sync_score)

    res["1a_sup_kNN_rank"] = per_subset((sup_rank + knn_rank) / 2, labels, cats, cat_to_idx)
    res["1b_sup_Mahal_rank"] = per_subset((sup_rank + mahal_rank) / 2, labels, cats, cat_to_idx)
    res["1c_sup_sync_rank"] = per_subset((sup_rank + sync_rank) / 2, labels, cats, cat_to_idx)

    # ── Method 1 alt: sigmoid logit-fusion ────────────────────────────────
    knn_z = normalize(knn_score)
    res_logit_grid = {}
    for beta in parse_float_list(args.betas):
        s = 1.0 / (1.0 + np.exp(-(sup_out["sup_logit"] + beta * knn_z)))
        res_logit_grid[f"beta={beta}"] = per_subset(s, labels, cats, cat_to_idx)
    res["1_logit_fusion_grid"] = res_logit_grid

    # ── Method 2: residual-gated late fusion ──────────────────────────────
    z_res_norm_z = normalize(sup_out["z_res_norm"])
    knn_z_safe = normalize(knn_score)
    sup_z = normalize(sup_out["sup_score"])
    res2_grid = {}
    for alpha in parse_float_list(args.alphas):
        gate = 1.0 / (1.0 + np.exp(-alpha * z_res_norm_z))
        s = (1.0 - gate) * sup_z + gate * knn_z_safe
        res2_grid[f"alpha={alpha}"] = per_subset(s, labels, cats, cat_to_idx)
    res["2_residual_gated_grid"] = res2_grid

    # ── Method 3: source-calibrated OOD-gated fusion ───────────────────────
    # kNN is standardized against AV1M-real self-kNN, then only allowed to move
    # the supervised score when it exceeds a source-calibrated high quantile.
    knn_ref_z = (knn_score - knn_ref_mu) / knn_ref_std
    sup_prob = 1.0 / (1.0 + np.exp(-np.clip(sup_out["sup_logit"], -50.0, 50.0)))
    sup_uncertainty = 1.0 - np.abs(2.0 * sup_prob - 1.0)
    res3_logit_grid = {}
    res3_rescue_grid = {}
    res3_uncertain_rescue_grid = {}
    for q_key, tau_z in gate_thresholds.items():
        for alpha in parse_float_list(args.gate_alphas):
            gate = sigmoid(alpha * (knn_ref_z - tau_z))
            rescue = gate * np.maximum(0.0, knn_ref_z - tau_z)
            uncertain_rescue = rescue * sup_uncertainty
            for beta in parse_float_list(args.gate_betas):
                key = f"q={q_key},alpha={alpha},beta={beta}"
                s_logit = sigmoid(sup_out["sup_logit"] + beta * gate * knn_ref_z)
                s_rescue = sigmoid(sup_out["sup_logit"] + beta * rescue)
                s_uncertain = sigmoid(sup_out["sup_logit"] + beta * uncertain_rescue)
                res3_logit_grid[key] = per_subset(s_logit, labels, cats, cat_to_idx)
                res3_rescue_grid[key] = per_subset(s_rescue, labels, cats, cat_to_idx)
                res3_uncertain_rescue_grid[key] = per_subset(s_uncertain, labels, cats, cat_to_idx)
    res["3_source_gate_logit_grid"] = res3_logit_grid
    res["3_source_gate_rescue_grid"] = res3_rescue_grid
    res["3_source_gate_uncertain_rescue_grid"] = res3_uncertain_rescue_grid

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "caren_methods_12.json")
    with open(out_path, "w") as f:
        json.dump({
            "sup_ckpt": args.sup_ckpt,
            "unsup_ckpt": args.unsup_ckpt,
            "eval_dataset": args.eval_dataset,
            "eval_features_root": args.eval_features_root,
            "ref_features_root": args.ref_features_root,
            "n_ref": len(av_items),
            "n_eval": len(ev_items),
            "n_self_excluded": n_self,
            "knn_ref_mu": knn_ref_mu,
            "knn_ref_std": knn_ref_std,
            "gate_thresholds_z": gate_thresholds,
            "results": res,
        }, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Caren Methods 1 + 2 — {args.eval_dataset} (N={len(ev_items)})  ref_self_excl={n_self}")
    print(f"  eval_features_root = {args.eval_features_root}")
    print(f"  ref_features_root  = {args.ref_features_root}")
    print("=" * 80)
    if args.eval_dataset == "favc":
        cols = ("overall", "RV-FA", "FV-RA", "FV-FA")
    else:
        cols = ("overall", "RV-FA", "FV-RA", "FV-FA")
    rows = [
        ("Sup A6 alone",               res["sup_alone"]),
        ("Unsup kNN (M1.0)",           res["unsup_kNN"]),
        ("Unsup Mahal",                res["unsup_Mahal"]),
        ("Unsup sync",                 res["unsup_sync"]),
        ("Rank fusion: Sup+kNN (1a)",  res["1a_sup_kNN_rank"]),
        ("Rank fusion: Sup+Mahal (1b)", res["1b_sup_Mahal_rank"]),
        ("Rank fusion: Sup+sync (1c)", res["1c_sup_sync_rank"]),
    ]
    for beta in parse_float_list(args.betas):
        rows.append((f"Logit fusion β={beta}", res_logit_grid[f"beta={beta}"]))
    for alpha in parse_float_list(args.alphas):
        rows.append((f"Residual-gate α={alpha}", res2_grid[f"alpha={alpha}"]))
    for key in sorted(res3_rescue_grid):
        rows.append((f"Src-gate rescue {key}", res3_rescue_grid[key]))
    for key in sorted(res3_uncertain_rescue_grid):
        rows.append((f"Src-gate uncertain {key}", res3_uncertain_rescue_grid[key]))

    def fmt(v): return f"{v:.4f}" if v is not None else "  n/a "
    header = f"{'Method':<32s} " + " ".join(f"{c+'/AUC':>10s} {c+'/AP':>10s}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, r in rows:
        cells = []
        for c in cols:
            m = r.get(c) or {"AUC": None, "AP": None}
            cells.append(f"{fmt(m.get('AUC')):>10s} {fmt(m.get('AP')):>10s}")
        print(f"{name:<32s} " + " ".join(cells))
    print("=" * len(header))


if __name__ == "__main__":
    main()
