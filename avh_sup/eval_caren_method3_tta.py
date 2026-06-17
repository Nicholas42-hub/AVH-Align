"""
Caren's Method 3: Test-time unsupervised adaptation (TTA) on FAVC.

Spec:
  - Freeze the task head (causal_head + domain heads).
  - Adapt CCD + URD + cross_incon modules on unlabeled FAVC features.
  - TENT-style entropy minimization on prediction logits as the TTA loss.
  - Compare FAVC test AUC before vs. after TTA, per FAVC subset.

We use the supervised A6 (paper-clean ckpt, seed=43) as the starting point.
The same FAVC test features used for the baseline AUC are used for TTA;
no FAVC labels are used at any point.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from mlp_fcd_a6 import AVH_FCD_A6


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
def eval_auc(model, loader, device, cat_to_idx):
    model.eval()
    scores, labels, cats = [], [], []
    for vs, audios, labs, cs in loader:
        for v, a, lab, c in zip(vs, audios, labs, cs):
            v = v.unsqueeze(0).to(device)
            a = a.unsqueeze(0).to(device)
            logit = model(v, a, mode="causal").squeeze().item()
            scores.append(float(torch.sigmoid(torch.tensor(float(logit)))))
            labels.append(int(lab))
            cats.append(int(c))
    scores = np.array(scores); labels = np.array(labels); cats = np.array(cats)
    res = {"overall": roc_auc_score(labels, scores) if len(set(labels)) == 2 else None}
    real_mask = (cats == cat_to_idx[_FAVC_REAL])
    for fc in _FAVC_FAKE:
        sel = real_mask | (cats == cat_to_idx[fc])
        if sel.sum() == 0 or len(set(labels[sel])) < 2:
            continue
        res[_SUBSET_ALIAS[fc]] = roc_auc_score(labels[sel], scores[sel])
    return res


def freeze_task_head(model):
    """Freeze causal_head + redundant_head + domain heads. Adapt CCD/URD/incon."""
    frozen_keywords = ("causal_head", "redundant_head", "domain_head_c", "domain_head_s")
    n_frozen, n_trainable = 0, 0
    trainable_params = []
    for n, p in model.named_parameters():
        if any(k in n for k in frozen_keywords):
            p.requires_grad_(False)
            n_frozen += 1
        else:
            p.requires_grad_(True)
            n_trainable += 1
            trainable_params.append(p)
    print(f"  freeze: {n_frozen} params  trainable: {n_trainable} params", flush=True)
    return trainable_params


def tta_tent(model, loader, device, lr=1e-4, n_steps=200, log_every=20):
    """TENT-style entropy minimization on FAVC unlabeled features."""
    trainable = freeze_task_head(model)
    opt = torch.optim.Adam(trainable, lr=lr)
    model.train()  # so dropout / training behaviour active
    step = 0
    losses = []
    while step < n_steps:
        for vs, audios, labs, cats in loader:
            for v, a, _, _ in zip(vs, audios, labs, cats):
                v = v.unsqueeze(0).to(device)
                a = a.unsqueeze(0).to(device)
                logit = model(v, a, mode="causal").squeeze()
                p = torch.sigmoid(logit)
                eps = 1e-7
                ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
                loss = ent.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                step += 1
                if step % log_every == 0:
                    recent = np.mean(losses[-log_every:])
                    print(f"    TTA step {step}/{n_steps}  ent={recent:.4f}", flush=True)
                if step >= n_steps:
                    break
            if step >= n_steps:
                break
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sup_ckpt", required=True)
    p.add_argument("--sup_config", required=True)
    p.add_argument("--favc_root", default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    p.add_argument("--favc_split_csv", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical/test_split.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=1e-5, help="TTA learning rate")
    p.add_argument("--n_steps", type=int, default=300, help="TTA steps (entropy minimisation)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load sup A6
    print(f"Loading: {args.sup_ckpt}", flush=True)
    with open(args.sup_config) as f:
        cfg = yaml.safe_load(f)
    apply_l2 = cfg.get("data_info", {}).get("apply_l2", True)

    def fresh_model():
        m = AVH_FCD_A6(config=cfg)
        ck = torch.load(args.sup_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        m.load_state_dict({k: v for k, v in sd.items()
                           if k in m.state_dict() and v.shape == m.state_dict()[k].shape},
                          strict=False)
        return m

    fv_items, cat_to_idx = _load_favc_items(args.favc_root, args.favc_split_csv)
    print(f"FAVC test clips: {len(fv_items)}", flush=True)
    fv_ds = _Dataset(fv_items, apply_l2)
    fv_dl = DataLoader(fv_ds, batch_size=4, num_workers=4, collate_fn=_collate, shuffle=True)

    # ── Baseline (no TTA) ───────────────────────────────────────────────
    print("\n[Baseline] eval before TTA...", flush=True)
    model = fresh_model().to(device)
    res_before = eval_auc(model, fv_dl, device, cat_to_idx)
    print(f"  before TTA: {res_before}", flush=True)

    # ── Different TTA configs ───────────────────────────────────────────
    configs = [
        {"lr": 1e-5, "n_steps": 200, "name": "lr1e-5_steps200"},
        {"lr": 5e-6, "n_steps": 500, "name": "lr5e-6_steps500"},
    ]
    if args.n_steps != 300 or args.lr != 1e-5:
        configs = [{"lr": args.lr, "n_steps": args.n_steps, "name": f"lr{args.lr}_steps{args.n_steps}"}]

    results = {"before_tta": res_before, "configs": {}}
    for cfg_i in configs:
        print(f"\n[TTA] config={cfg_i['name']}  (lr={cfg_i['lr']}  n_steps={cfg_i['n_steps']})", flush=True)
        # fresh model each time
        model = fresh_model().to(device)
        # use a shuffling loader for TTA (different epoch each time)
        fv_dl_tta = DataLoader(fv_ds, batch_size=4, num_workers=4, collate_fn=_collate, shuffle=True)
        losses = tta_tent(model, fv_dl_tta, device,
                           lr=cfg_i["lr"], n_steps=cfg_i["n_steps"], log_every=50)
        # eval post-TTA
        fv_dl_eval = DataLoader(fv_ds, batch_size=4, num_workers=4, collate_fn=_collate, shuffle=False)
        res_after = eval_auc(model, fv_dl_eval, device, cat_to_idx)
        results["configs"][cfg_i["name"]] = {
            "after_tta": res_after,
            "loss_first_50": float(np.mean(losses[:50])) if len(losses) >= 50 else None,
            "loss_last_50": float(np.mean(losses[-50:])) if len(losses) >= 50 else None,
        }
        print(f"  after TTA: {res_after}", flush=True)
        delta = {k: (res_after.get(k, 0) - res_before.get(k, 0)) if res_after.get(k) is not None and res_before.get(k) is not None else None
                 for k in res_before}
        print(f"  Δ (TTA - baseline): {delta}", flush=True)

    out_path = os.path.join(args.output_dir, "caren_method3_tta.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Caren Method 3 — Test-time adaptation (TENT, FAVC test, N=6667)")
    print("=" * 78)
    print(f"{'Config':<32s} {'overall':>8s} {'RV-FA':>8s} {'FV-RA':>8s} {'FV-FA':>8s}")
    print("-" * 78)
    def fmt(v): return f"{v:.4f}" if v is not None else "  n/a "
    print(f"{'baseline (no TTA)':<32s} {fmt(res_before.get('overall')):>8s} {fmt(res_before.get('RV-FA')):>8s} {fmt(res_before.get('FV-RA')):>8s} {fmt(res_before.get('FV-FA')):>8s}")
    for cfg_i in configs:
        r = results["configs"][cfg_i["name"]]["after_tta"]
        print(f"{('TTA: ' + cfg_i['name']):<32s} {fmt(r.get('overall')):>8s} {fmt(r.get('RV-FA')):>8s} {fmt(r.get('FV-RA')):>8s} {fmt(r.get('FV-FA')):>8s}")
    print("=" * 78)


if __name__ == "__main__":
    main()
