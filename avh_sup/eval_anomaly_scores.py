"""
Three unsupervised anomaly scoring approaches on top of TriRoute representations.

Approach 1 — Visual-only anomaly (u_v):
    Fit a reference distribution on AV1M real train u_v.
    Score = distance(test u_v, reference).
    Methods: Mahalanobis, kNN, diagonal Gaussian NLL.

Approach 2 — Audio-visual disagreement (s_v vs s_a):
    Score = 1 - cosine(s_v_pooled, s_a_pooled).

Approach 3 — Residual-gated anomaly:
    Score = visual_anomaly(u_v) - alpha * ||r_v||
    The residual norm acts as a domain-shift gate.

Supports both:
  --model_type unsup   : TriRouteUnsupSync checkpoint
  --model_type sup     : AVH_FCD_A6 Lightning checkpoint
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset

FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]
REAL_CATEGORY = "RealVideo-RealAudio"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_unsup_model(ckpt_path, device):
    from train_eval_triroute_unsup_sync import TriRouteUnsupSync
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TriRouteUnsupSync(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, "unsup"


def load_sup_model(ckpt_path, device):
    from mlp_fcd_a6 import AVH_FCD_A6
    import lightning as L
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Lightning checkpoint stores hparams under "hyper_parameters"
    config = ckpt.get("hyper_parameters", {})
    if not config:
        raise ValueError("Could not find hyper_parameters in Lightning checkpoint.")
    model = AVH_FCD_A6.load_from_checkpoint(ckpt_path, config=config, map_location="cpu")
    model.to(device)
    model.eval()
    return model, "sup"


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, model_type, loader, device):
    """Returns dict of pooled features + labels + paths."""
    all_u_v, all_s_v, all_s_a, all_r_v = [], [], [], []
    labels_list, paths_list = [], []

    for batch in tqdm.tqdm(loader, desc="extracting"):
        video, audio, label, path = batch
        video = video.to(device)
        audio = audio.to(device)
        if video.ndim == 2:
            video = video.unsqueeze(0)
            audio = audio.unsqueeze(0)

        if model_type == "unsup":
            _, comp = model.encode(video, audio)
        else:
            _, _, comp = model._encode(video, audio)

        all_u_v.append(comp["u_v"].mean(1).cpu().float().numpy())
        all_s_v.append(comp["s_v"].mean(1).cpu().float().numpy())
        all_s_a.append(comp["s_a"].mean(1).cpu().float().numpy())
        all_r_v.append(comp["r_v"].mean(1).cpu().float().numpy())

        lbl = label.numpy().astype(int) if isinstance(label, torch.Tensor) else np.array(label, dtype=int)
        labels_list.extend(lbl.tolist())
        paths_list.extend(list(path) if not isinstance(path, str) else [path])

    return {
        "u_v":    np.concatenate(all_u_v, axis=0),
        "s_v":    np.concatenate(all_s_v, axis=0),
        "s_a":    np.concatenate(all_s_a, axis=0),
        "r_v":    np.concatenate(all_r_v, axis=0),
        "labels": np.array(labels_list, dtype=int),
        "paths":  paths_list,
    }


# ── Reference distribution fitting ───────────────────────────────────────────

def fit_mahalanobis(ref):
    mu = ref.mean(0)
    cov = np.cov(ref.T) + np.eye(ref.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    return mu, cov_inv


def score_mahalanobis(feats, mu, cov_inv):
    d = feats - mu
    return np.einsum("bi,ij,bj->b", d, cov_inv, d)


def fit_knn(ref, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="ball_tree")
    nn.fit(ref)
    return nn


def score_knn(feats, nn_model):
    dists, _ = nn_model.kneighbors(feats)
    return dists.mean(axis=1)


def fit_gaussian(ref):
    mu  = ref.mean(0)
    std = ref.std(0) + 1e-6
    return mu, std


def score_gaussian(feats, mu, std):
    return ((feats - mu) ** 2 / (2 * std ** 2)).sum(1)


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_av_disagreement(s_v, s_a):
    sv = s_v / (np.linalg.norm(s_v, axis=1, keepdims=True) + 1e-8)
    sa = s_a / (np.linalg.norm(s_a, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (sv * sa).sum(1)


def score_residual_gated(base_score, r_v, alpha):
    domain_conf = np.linalg.norm(r_v, axis=1)
    return base_score - alpha * domain_conf


# ── Metrics ───────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else None


def safe_ap(y_true, y_score):
    return average_precision_score(y_true, y_score) if len(set(y_true)) >= 2 else None


def per_category_metrics(paths, scores, labels):
    results = {
        "overall": {
            "auc": safe_auc(labels, scores),
            "ap":  safe_ap(labels, scores),
            "n":   len(labels),
        }
    }
    for cat in FAKE_CATEGORIES:
        mask = [(cat in p) or (REAL_CATEGORY in p) for p in paths]
        ys = [y for y, m in zip(labels, mask) if m]
        ss = [s for s, m in zip(scores, mask) if m]
        results[cat] = {"auc": safe_auc(ys, ss), "ap": safe_ap(ys, ss), "n": len(ys)}
    return results


def write_results(out_dir, title, metrics):
    os.makedirs(out_dir, exist_ok=True)
    lines = [title, "=" * 60]
    for key, m in metrics.items():
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
        ap  = f"{m['ap']:.4f}"  if m["ap"]  is not None else "  N/A "
        lines.append(f"  {key:<30}  AUC={auc}  AP={ap}  N={m['n']}")
    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "eval_results.txt"), "w") as f:
        f.write(text + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["unsup", "sup"])
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--av1m_features",  default="/data/projects/punim2637/nnliang/AVH-Align/data/avh_features")
    parser.add_argument("--av1m_csv_root",  default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m")
    parser.add_argument("--favc_features",  default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    parser.add_argument("--favc_csv_root",  default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical")
    parser.add_argument("--knn_k",   type=int,   default=5)
    parser.add_argument("--alpha",   type=float, default=1.0, help="residual gate weight")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    if args.model_type == "unsup":
        model, mtype = load_unsup_model(args.ckpt_path, device)
    else:
        model, mtype = load_sup_model(args.ckpt_path, device)
    print(f"Loaded {mtype} model from {args.ckpt_path}", flush=True)

    # Reference features: AV1M real train
    print("\nExtracting AV1M real train features (reference)...", flush=True)
    av1m_cfg = {"root_path": args.av1m_features, "csv_root_path": args.av1m_csv_root, "apply_l2": True}
    ref_ds = AV1M_trainval_dataset(av1m_cfg, split="train")
    ref_ds.df = ref_ds.df[ref_ds.df["label"] == 0].reset_index(drop=True)
    ref_loader = DataLoader(ref_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    ref = extract_features(model, mtype, ref_loader, device)
    print(f"Reference: {len(ref['u_v'])} real clips", flush=True)

    # Fit reference distributions on u_v
    print("\nFitting reference distributions...", flush=True)
    mu_maha, cov_inv  = fit_mahalanobis(ref["u_v"])
    nn_model          = fit_knn(ref["u_v"], k=args.knn_k)
    mu_gauss, std_gauss = fit_gaussian(ref["u_v"])

    # Test features: FAVC
    print("\nExtracting FAVC test features...", flush=True)
    favc_cfg = {"root_path": args.favc_features, "csv_root_path": args.favc_csv_root, "apply_l2": True}
    test_ds  = FakeAVCeleb_NPZ_Dataset(favc_cfg, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test = extract_features(model, mtype, test_loader, device)
    labels = test["labels"].tolist()
    paths  = test["paths"]
    print(f"Test: {len(test['u_v'])} FAVC clips", flush=True)

    # Compute all scores
    maha_scores = score_mahalanobis(test["u_v"], mu_maha, cov_inv)
    knn_scores  = score_knn(test["u_v"], nn_model)
    gauss_scores = score_gaussian(test["u_v"], mu_gauss, std_gauss)
    av_dis_scores = score_av_disagreement(test["s_v"], test["s_a"])

    scorers = {
        "1a_visual_mahalanobis": maha_scores,
        "1b_visual_knn":         knn_scores,
        "1c_visual_gaussian":    gauss_scores,
        "2_av_disagreement":     av_dis_scores,
    }
    for alpha in [0.5, 1.0, 2.0]:
        scorers[f"3_residual_gated_maha_a{alpha}"] = score_residual_gated(maha_scores, test["r_v"], alpha)

    # Evaluate and write
    print("\n" + "=" * 68)
    for name, scores in scorers.items():
        out_dir = os.path.join(args.output_dir, name)
        metrics = per_category_metrics(paths, scores.tolist(), labels)
        write_results(out_dir, f"[{name}]", metrics)

    print(f"\nAll results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
