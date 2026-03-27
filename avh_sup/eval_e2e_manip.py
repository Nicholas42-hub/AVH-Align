"""
Manipulation-Type Probe Evaluation for A2_e2e_manip checkpoint
===============================================================
Tests:
  1. Manip-type probe (4-class):  can a linear probe predict manip type
     from Z_c vs Z_s?
     → After training, Z_c should be manip-type-INVARIANT (low AUC/acc)
     → Z_s should be manip-type-SPECIFIC (higher acc is fine / expected)

  2. Label probe (binary real/fake) in-domain (AV1M holdout)
     → Z_c AUC should remain high (~0.99)

  3. Label probe OOD (FakeAVCeleb)
     → Z_c should generalise (high AUC)

  4. AV1M-vs-FAVC domain probe (same as eval_e2e_domain.py)
     → baseline comparison

Usage
-----
  python eval_e2e_manip.py \\
    --ckpt  outputs_A2_e2e_manip/ckpts/model-epoch=14-val_auc_causal=0.9961.ckpt \\
    --config configs/A2_e2e_manip.yaml \\
    --av1m_raw /data/scratch/.../avd1m_preprocessed \\
    --favc_raw /data/scratch/.../favc_preprocessed \\
    --av1m_csv csv_metadata/av1m \\
    --output_dir outputs_A2_e2e_manip/results \\
    --holdout_only
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_e2e import E2EModel
from datasets_e2e import (
    AV1M_E2E_Dataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn
)


# ── Manip-type mapping ────────────────────────────────────────────────────────
MANIP_NAMES = {0: "real", 1: "real_video_fake_audio",
               2: "fake_video_real_audio", 3: "fake_video_fake_audio"}


# ── Representation extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_reps(model: E2EModel, dataloader: DataLoader,
                 device: torch.device, domain_label: int,
                 has_manip: bool = False):
    """
    Returns mean-pooled Z_c, Z_s [N, D] and label/manip arrays.
    If has_manip=True the dataloader yields 5-tuples (video,audio,label,manip,path).
    """
    model.eval()
    model.to(device)

    all_zc, all_zs = [], []
    all_cls, all_manip, all_dom = [], [], []

    for batch in tqdm(dataloader, leave=False):
        if batch is None:
            continue

        if has_manip:
            video, audio, cls_labels, manip_labels, _ = batch
            manip_labels = manip_labels.tolist()
        else:
            video, audio, cls_labels, _ = batch
            manip_labels = [-1] * len(cls_labels)

        video = video.to(device)
        audio = audio.to(device)

        video_feats, audio_feats = model.encoder(video, audio)
        causal_repr, spurious_repr, _ = model.head._encode(video_feats, audio_feats)

        zc = causal_repr.mean(dim=1).cpu().numpy()   # [B, D]
        zs = spurious_repr.mean(dim=1).cpu().numpy() # [B, D]

        all_zc.append(zc)
        all_zs.append(zs)
        all_cls.extend(cls_labels.tolist())
        all_manip.extend(manip_labels)
        all_dom.extend([domain_label] * len(cls_labels))

    return (np.concatenate(all_zc),
            np.concatenate(all_zs),
            np.array(all_cls,   dtype=np.int32),
            np.array(all_manip, dtype=np.int32),
            np.array(all_dom,   dtype=np.int32))


# ── Probes ────────────────────────────────────────────────────────────────────

def binary_probe_auc(features, labels, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(features, labels):
        sc  = StandardScaler().fit(features[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(features[tr]), labels[tr])
        prob = clf.predict_proba(sc.transform(features[te]))[:, 1]
        try:
            aucs.append(roc_auc_score(labels[te], prob))
        except ValueError:
            pass
    return float(np.mean(aucs)), float(np.std(aucs))


def multiclass_probe_acc(features, labels, n_splits=5, seed=42):
    """5-fold CV accuracy for 4-class manip-type probe."""
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(features, labels):
        sc  = StandardScaler().fit(features[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                 solver="lbfgs")
        clf.fit(sc.transform(features[tr]), labels[tr])
        preds = clf.predict(sc.transform(features[te]))
        accs.append((preds == labels[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--av1m_raw",    required=True)
    parser.add_argument("--favc_raw",    required=True)
    parser.add_argument("--av1m_csv",    required=True)
    parser.add_argument("--output_dir",  default="outputs_A2_e2e_manip/results")
    parser.add_argument("--max_frames",  type=int, default=150)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_clips",   type=int, default=500)
    parser.add_argument("--holdout_only", action="store_true")
    parser.add_argument("--train_seed",  type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model = E2EModel.load_from_checkpoint(args.ckpt, config=config)
    model.eval()
    model.to(device)
    print("Model loaded OK")

    # ── AV1M with manip labels ─────────────────────────────────────────────
    av1m_ds = AV1M_E2E_Dataset(
        args.av1m_raw,
        os.path.join(args.av1m_csv, "val_manip.csv"),
        split="val", max_frames=args.max_frames,
        return_manip_type=True)

    if args.holdout_only:
        n_total   = len(av1m_ds)
        n_train   = int(n_total * 0.8)
        n_holdout = n_total - n_train
        _, holdout_split = random_split(
            av1m_ds, [n_train, n_holdout],
            generator=torch.Generator().manual_seed(args.train_seed))
        av1m_eval = holdout_split
        print(f"[holdout_only] evaluating {n_holdout} unseen clips")
    else:
        n = min(args.max_clips, len(av1m_ds))
        av1m_eval = Subset(av1m_ds, list(range(n)))

    av1m_loader = DataLoader(av1m_eval, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=e2e_collate_fn)

    # ── Extract AV1M representations ───────────────────────────────────────
    print("Extracting AV1M Z_c / Z_s ...")
    zc_av1m, zs_av1m, cls_av1m, manip_av1m, _ = extract_reps(
        model, av1m_loader, device, domain_label=0, has_manip=True)
    print(f"  AV1M clips: {len(zc_av1m)}")
    unique, counts = np.unique(manip_av1m, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    manip={u} ({MANIP_NAMES.get(u,'?')}): {c}")

    # ── 1. Manip-type probe (4-class) ─────────────────────────────────────
    print("\n=== 1. Manip-type probe (4-class linear) ===")
    chance = 1.0 / len(np.unique(manip_av1m))
    zc_mt_acc, zc_mt_std = multiclass_probe_acc(zc_av1m, manip_av1m)
    zs_mt_acc, zs_mt_std = multiclass_probe_acc(zs_av1m, manip_av1m)
    print(f"  Z_c manip-type acc: {zc_mt_acc:.4f} ± {zc_mt_std:.4f}  (chance={chance:.2f})")
    print(f"  Z_s manip-type acc: {zs_mt_acc:.4f} ± {zs_mt_std:.4f}  (chance={chance:.2f})")
    print(f"  → Z_c manip-invariant? {'YES ✓' if zc_mt_acc < chance + 0.05 else 'NO'} "
          f"(target ≤{chance+0.05:.2f}, got {zc_mt_acc:.4f})")
    print(f"  → Z_s manip-specific?  {'YES ✓' if zs_mt_acc > chance + 0.05 else 'NO'} "
          f"(target >{chance+0.05:.2f}, got {zs_mt_acc:.4f})")

    # ── 2. Label probe in-domain (AV1M holdout) ────────────────────────────
    print("\n=== 2. Label probe (binary real/fake) — in-domain AV1M ===")
    lbl_zc_id, lbl_zc_id_std = binary_probe_auc(zc_av1m, cls_av1m)
    lbl_zs_id, lbl_zs_id_std = binary_probe_auc(zs_av1m, cls_av1m)
    print(f"  Z_c label AUC: {lbl_zc_id:.4f} ± {lbl_zc_id_std:.4f}")
    print(f"  Z_s label AUC: {lbl_zs_id:.4f} ± {lbl_zs_id_std:.4f}")

    # ── 3. OOD probe on FakeAVCeleb ────────────────────────────────────────
    favc_ds = FakeAVCeleb_E2E_Dataset(args.favc_raw, max_frames=args.max_frames)
    n_favc  = min(args.max_clips, len(favc_ds))
    rng     = np.random.default_rng(42)
    favc_idx = rng.choice(len(favc_ds), size=n_favc, replace=False).tolist()
    favc_loader = DataLoader(Subset(favc_ds, favc_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=e2e_collate_fn)

    print("\nExtracting FAVC Z_c / Z_s ...")
    zc_favc, zs_favc, cls_favc, _, dom_favc = extract_reps(
        model, favc_loader, device, domain_label=1, has_manip=False)
    print(f"  FAVC clips: {len(zc_favc)}")

    valid = (cls_favc == 0) | (cls_favc == 1)
    zc_fv, zs_fv, cls_fv = zc_favc[valid], zs_favc[valid], cls_favc[valid]
    print(f"\n=== 3. Label probe (OOD FakeAVCeleb) real={( cls_fv==0).sum()} fake={(cls_fv==1).sum()} ===")
    if len(np.unique(cls_fv)) > 1:
        lbl_zc_ood, lbl_zc_ood_std = binary_probe_auc(zc_fv, cls_fv)
        lbl_zs_ood, lbl_zs_ood_std = binary_probe_auc(zs_fv, cls_fv)
        print(f"  Z_c label AUC (OOD): {lbl_zc_ood:.4f} ± {lbl_zc_ood_std:.4f}")
        print(f"  Z_s label AUC (OOD): {lbl_zs_ood:.4f} ± {lbl_zs_ood_std:.4f}")
    else:
        lbl_zc_ood = lbl_zc_ood_std = lbl_zs_ood = lbl_zs_ood_std = None
        print("  Skipped: need both real and fake FAVC clips.")

    # ── 4. AV1M-vs-FAVC domain probe ──────────────────────────────────────
    print("\n=== 4. AV1M-vs-FAVC domain probe ===")
    zc_all  = np.concatenate([zc_av1m, zc_favc])
    zs_all  = np.concatenate([zs_av1m, zs_favc])
    dom_all = np.concatenate([np.zeros(len(zc_av1m), dtype=np.int32),
                               np.ones(len(zc_favc), dtype=np.int32)])
    dom_zc, dom_zc_std = binary_probe_auc(zc_all, dom_all)
    dom_zs, dom_zs_std = binary_probe_auc(zs_all, dom_all)
    print(f"  Z_c domain AUC: {dom_zc:.4f} ± {dom_zc_std:.4f}")
    print(f"  Z_s domain AUC: {dom_zs:.4f} ± {dom_zs_std:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_name = "holdout" if args.holdout_only else "full"
    out_path = os.path.join(args.output_dir, f"eval_manip_{out_name}.json")
    results = {
        "checkpoint": args.ckpt,
        "n_av1m":     int(len(zc_av1m)),
        "n_favc":     int(len(zc_favc)),
        "manip_type_probe": {
            "Z_c_acc_mean": zc_mt_acc, "Z_c_acc_std": zc_mt_std,
            "Z_s_acc_mean": zs_mt_acc, "Z_s_acc_std": zs_mt_std,
            "chance": chance,
        },
        "label_probe_indomain": {
            "Z_c_auc_mean": lbl_zc_id, "Z_c_auc_std": lbl_zc_id_std,
            "Z_s_auc_mean": lbl_zs_id, "Z_s_auc_std": lbl_zs_id_std,
        },
        "label_probe_ood_favc": {
            "Z_c_auc_mean": lbl_zc_ood, "Z_c_auc_std": lbl_zc_ood_std,
            "Z_s_auc_mean": lbl_zs_ood, "Z_s_auc_std": lbl_zs_ood_std,
        },
        "domain_probe_av1m_favc": {
            "Z_c_auc_mean": dom_zc, "Z_c_auc_std": dom_zc_std,
            "Z_s_auc_mean": dom_zs, "Z_s_auc_std": dom_zs_std,
        },
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
