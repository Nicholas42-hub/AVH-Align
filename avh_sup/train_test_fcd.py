"""
A5 — FCD model training entry-point for AVH-Align.

Usage:
  python train_test_fcd.py --config_path configs/A5.yaml --mode train
  python train_test_fcd.py --config_path configs/A5.yaml --mode test

Supports the same config structure as the ablation training scripts
(data_info, model_hparams, callbacks, epochs, seed, early_stopping).
No domain-mixing:  A5 trains on AV1M only (same as A1 / A2).
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
from torch.utils.data import DataLoader
import lightning as L

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_fcd import AVH_FCD


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(config: dict):
    data_cfg   = config["data_info"]
    if data_cfg.get("name") == "FAVC":
        train_ds = FakeAVCeleb_NPZ_Dataset(data_cfg, split="train")
        val_ds   = FakeAVCeleb_NPZ_Dataset(data_cfg, split="val")
    else:
        train_ds   = AV1M_trainval_dataset(data_cfg, split="train")
        val_ds     = AV1M_trainval_dataset(data_cfg, split="val")
    train_dl   = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl     = DataLoader(val_ds,   shuffle=False, batch_size=1)
    print(f"Train: {len(train_ds)} clips   Val: {len(val_ds)} clips", flush=True)
    return train_dl, val_dl


# ── Callbacks ──────────────────────────────────────────────────────────────────

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
    train_dl, val_dl = load_data(config)
    model = AVH_FCD(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ── Test ───────────────────────────────────────────────────────────────────────

def test(config: dict):
    from datasets import load_data as _load_test
    test_dl = _load_test(config=config["data_info"], test=True)

    model = AVH_FCD.load_from_checkpoint(config["ckpt_path"])
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
        "path":           all_paths,
        "label":          labels_arr,
        "score_full":     scores_by_mode["full"],
        "score_causal":   scores_by_mode["causal"],
        "score_spurious": scores_by_mode["spurious"],
    }).to_csv(os.path.join(config["output_path"], "scores.csv"), index=False)

    print("\n" + "=" * 68)
    print("  A5 (FCD) evaluation results")
    print("=" * 68)
    report_lines = []
    for mode in ("full", "causal", "spurious"):
        scores = np.array(scores_by_mode[mode])
        auc = roc_auc_score(y_true=labels_arr, y_score=scores)
        ap  = average_precision_score(y_true=labels_arr, y_score=scores)
        line = f"{mode:<10}  AUC={auc:.4f}  AP={ap:.4f}  N={len(labels_arr)}"
        print(f"  {line}")
        report_lines.append(line)
    print("=" * 68)

    # Causal hypothesis check
    auc_causal   = roc_auc_score(y_true=labels_arr, y_score=np.array(scores_by_mode["causal"]))
    auc_spurious = roc_auc_score(y_true=labels_arr, y_score=np.array(scores_by_mode["spurious"]))
    auc_full     = roc_auc_score(y_true=labels_arr, y_score=np.array(scores_by_mode["full"]))

    delta = abs(auc_full - auc_causal)
    spur_near_half = abs(auc_spurious - 0.5) < 0.05

    print(f"\n  Causal hypothesis check:")
    print(f"  |AUC_full - AUC_causal|  = {delta:.4f}  "
          f"{'✓ PASS' if delta < 0.05 else '✗ FAIL'}  (threshold 0.05)")
    print(f"  AUC_spurious ≈ 0.5?       = {auc_spurious:.4f}  "
          f"{'✓ PASS' if spur_near_half else '✗ FAIL'}  (±0.05 band)")

    with open(os.path.join(config["output_path"], "results.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")
        f.write(f"\nAUC_full={auc_full:.4f}  AUC_causal={auc_causal:.4f}  AUC_spurious={auc_spurious:.4f}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mode",        type=str, default="train", choices=["train", "test"])
    parser.add_argument("--seed",        type=int, default=None, help="Override seed in config")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Override all output dirs (ckpts, logs, results) with this base path")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        base = args.output_dir
        config["callbacks"]["logger"]["log_path"] = os.path.join(base, "logs")
        config["callbacks"]["ckpt_args"]["ckpt_dir"] = os.path.join(base, "ckpts")
        config["output_path"] = os.path.join(base, "results")
        os.makedirs(os.path.join(base, "logs"),    exist_ok=True)
        os.makedirs(os.path.join(base, "ckpts"),   exist_ok=True)
        os.makedirs(os.path.join(base, "results"), exist_ok=True)

    set_seed(config.get("seed", 42))

    print(f"\n{'='*68}")
    print(f"  A5 — FCD Feature Causality Decomposition")
    print(f"  syn_dim={config['model_hparams']['syn_dim']}  "
          f"spec_dim={config['model_hparams']['spec_dim']}  "
          f"lr={config['model_hparams']['lr']}")
    print(f"  λ_mi={config['model_hparams']['lambda_mi']}  "
          f"λ_sda={config['model_hparams']['lambda_sda']}  "
          f"λ_dis={config['model_hparams']['lambda_dis']}  "
          f"λ_orth={config['model_hparams']['lambda_orth']}")
    print(f"{'='*68}\n")

    if args.mode == "train":
        train(config)
    else:
        test(config)
