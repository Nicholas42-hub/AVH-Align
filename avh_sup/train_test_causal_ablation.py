"""
Ablation study training entry-point for AVH-Align causal disentanglement.

Supports three ablation configurations (set via config['ablation'] flags):
  A1 — Adversarial loss only
  A2 — Adversarial + Stop Gradient (Z_c-only classification)
  A3 — Adversarial + Stop Gradient + Domain-aware constraint

For A3, a DomainMixedDataset interleaves AV1M and FAVC training samples,
assigning domain_label=0 (AV1M) and domain_label=1 (FAVC).  FAVC cls_label
is set to -1 and masked out of the classification loss.

Usage:
  python train_test_causal_ablation.py --config_path configs/ablation_a1.yaml
  python train_test_causal_ablation.py --config_path configs/ablation_a2.yaml
  python train_test_causal_ablation.py --config_path configs/ablation_a3.yaml
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import lightning as L

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_causal_ablation import AVH_Causal_Ablation


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Domain-aware Dataset Wrapper ──────────────────────────────────────────────

class DomainLabeledDataset(Dataset):
    """
    Wraps an existing dataset and appends a domain_label to every item.

    Original item  : (video, audio, cls_label, path)
    New item       : (video, audio, cls_label, domain_label, path)

    For FAVC samples used as unlabelled domain data, pass cls_label_override=-1
    to mask them out of the classification loss during training.
    """
    def __init__(self, dataset: Dataset, domain_label: int, cls_label_override: int = None):
        self.dataset            = dataset
        self.domain_label       = domain_label
        self.cls_label_override = cls_label_override

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.dataset[idx]
        if self.cls_label_override is not None:
            cls_label = self.cls_label_override
        return video, audio, cls_label, self.domain_label, path


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_for_ablation(config: dict, use_domain: bool):
    """
    Returns (train_dl, val_dl).

    If use_domain is True and domain_data_info is present in config:
        - Train = DomainLabeledDataset(AV1M, domain=0)
                  + DomainLabeledDataset(FAVC, domain=1, cls_override=-1)
        - Val   = DomainLabeledDataset(AV1M, domain=0)   [AV1M val only]
    Else:
        - Plain AV1M train/val datasets (no domain label)
    """
    data_cfg = config["data_info"]

    if use_domain and "domain_data_info" in config:
        dom_cfg = config["domain_data_info"]

        # AV1M: domain_label = 0
        av1m_train = AV1M_trainval_dataset(data_cfg, split="train")
        av1m_val   = AV1M_trainval_dataset(data_cfg, split="val")
        av1m_train_dom = DomainLabeledDataset(av1m_train, domain_label=0)
        av1m_val_dom   = DomainLabeledDataset(av1m_val,   domain_label=0)

        # FAVC train: domain_label = 1, cls_label = -1 (unlabelled for cls task)
        favc_train = FakeAVCeleb_NPZ_Dataset(dom_cfg, split="train")
        favc_train_dom = DomainLabeledDataset(favc_train, domain_label=1, cls_label_override=-1)

        print(
            f"Domain-mixed train: AV1M={len(av1m_train)}, FAVC={len(favc_train)} "
            f"→ total={len(av1m_train) + len(favc_train)}",
            flush=True,
        )

        train_ds = ConcatDataset([av1m_train_dom, favc_train_dom])
        val_ds   = av1m_val_dom

    else:
        # A1 / A2: plain AV1M, no domain label
        train_ds = AV1M_trainval_dataset(data_cfg, split="train")
        val_ds   = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)
    return train_dl, val_dl


# ── Callbacks ─────────────────────────────────────────────────────────────────

def init_callbacks(config: dict):
    log_cfg = config["logger"]
    if log_cfg["name"] == "tensorboard":
        logger = TensorBoardLogger(log_cfg["log_path"])
    elif log_cfg["name"] == "csv":
        logger = CSVLogger(log_cfg["log_path"])
    else:
        raise ValueError(f"Logger '{log_cfg['name']}' not implemented")

    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=config["ckpt_args"]["ckpt_dir"],
                filename="model-{epoch:02d}",
                mode=config["ckpt_args"]["mode"],
                save_top_k=1,
            )
        )
    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                patience=config["early_stopping"]["patience"],
            )
        )
    return logger, callbacks


# ── Train ──────────────────────────────────────────────────────────────────────

def train(config: dict):
    use_domain = config.get("ablation", {}).get("use_domain", False)
    train_dl, val_dl = load_data_for_ablation(config, use_domain=use_domain)

    model = AVH_Causal_Ablation(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ── Test ───────────────────────────────────────────────────────────────────────

def test(config: dict):
    from datasets import load_data
    test_dl = load_data(config=config["data_info"], test=True)

    model = AVH_Causal_Ablation.load_from_checkpoint(config["ckpt_path"])
    model.to("cuda")
    model.eval()

    all_labels = []
    all_paths  = []
    scores_by_mode = {"full": [], "causal": [], "spurious": []}

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dl):
            video_feats, audio_feats, labels, paths = batch
            video_feats = video_feats.to("cuda")
            audio_feats = audio_feats.to("cuda")
            all_labels.extend(labels.cpu().numpy().tolist())
            all_paths.extend(paths)
            for mode in ("full", "causal", "spurious"):
                s = model.predict_scores(video_feats, audio_feats, mode=mode)
                scores_by_mode[mode].extend(s.cpu().numpy().tolist())

    labels_arr = np.array(all_labels)
    os.makedirs(config["output_path"], exist_ok=True)

    pd.DataFrame({
        "path":          all_paths,
        "label":         labels_arr,
        "score_full":    scores_by_mode["full"],
        "score_causal":  scores_by_mode["causal"],
        "score_spurious":scores_by_mode["spurious"],
    }).to_csv(os.path.join(config["output_path"], "scores.csv"), index=False)

    print("\n" + "="*68)
    print("  Ablation evaluation results")
    print("="*68)
    report_lines = []
    for mode in ("full", "causal", "spurious"):
        scores = np.array(scores_by_mode[mode])
        auc = roc_auc_score(y_true=labels_arr, y_score=scores)
        ap  = average_precision_score(y_true=labels_arr, y_score=scores)
        print(f"  {mode:<10}  AUC={auc:.4f}  AP={ap:.4f}  N={len(labels_arr)}")
        report_lines.append(f"{mode:<10}  AUC={auc:.4f}  AP={ap:.4f}  N={len(labels_arr)}")
    print("="*68)

    # ── Causal hypothesis check ────────────────────────────────────────────────
    scores_causal   = np.array(scores_by_mode["causal"])
    scores_spurious = np.array(scores_by_mode["spurious"])
    auc_causal   = roc_auc_score(y_true=labels_arr, y_score=scores_causal)
    auc_spurious = roc_auc_score(y_true=labels_arr, y_score=scores_spurious)

    delta = abs(auc_causal - roc_auc_score(
        y_true=labels_arr, y_score=np.array(scores_by_mode["full"])
    ))
    spur_near_half = abs(auc_spurious - 0.5) < 0.05

    print(f"\n  Hypothesis check:")
    print(f"  |AUC_full - AUC_causal| = {delta:.4f}  "
          f"{'✓ PASS' if delta < 0.05 else '✗ FAIL'}  (threshold 0.05)")
    print(f"  AUC_spurious near 0.5?  = {auc_spurious:.4f}  "
          f"{'✓ PASS' if spur_near_half else '✗ FAIL'}  (±0.05 band)")

    with open(os.path.join(config["output_path"], "results.txt"), "w") as f:
        f.write("\n".join(report_lines))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    ablation_id = config.get("ablation_id", "A?")
    print(f"\n{'='*68}")
    print(f"  Ablation {ablation_id}  |  use_adv={config['ablation']['use_adv']}  "
          f"use_sg={config['ablation']['use_sg']}  "
          f"use_domain={config['ablation']['use_domain']}")
    print(f"{'='*68}\n")

    if args.mode == "train":
        train(config)
    else:
        test(config)
