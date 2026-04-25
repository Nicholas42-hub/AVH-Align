"""
AVH-Align/sup (baseline) training on FakeAVCeleb.

Trains AVH_Sup (MLP alignment network + log-sum-exp + cross-entropy) entirely
on FakeAVCeleb train split, validates on FAVC val split.

Architecture: AVH_Sup from mlp.py (model_type="mlp", input_type="both").
Loss: cross-entropy on logsumexp-pooled per-frame alignment scores.
Optimizer: Adam lr=1e-3 (fixed).

This script is the FAVC-trained counterpart to the original AV1M-trained
baseline (checkpoints/avh_sup/AVH_Sup_FAVC.ckpt from the paper).

Usage:
  python train_test_baseline_favc.py --config_path configs/baseline_favc_seed42.yaml
  python train_test_baseline_favc.py --config_path configs/baseline_favc_seed42.yaml --seed 42
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
import lightning as L

from datasets import FakeAVCeleb_NPZ_Dataset
from mlp import AVH_Sup


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
    data_cfg = config["data_info"]
    train_ds = FakeAVCeleb_NPZ_Dataset(data_cfg, split="train")
    val_ds   = FakeAVCeleb_NPZ_Dataset(data_cfg, split="val")
    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)
    print(f"Train: {len(train_ds)} clips   Val: {len(val_ds)} clips", flush=True)
    return train_dl, val_dl


# ── Callbacks ──────────────────────────────────────────────────────────────────

def init_callbacks(config: dict):
    log_cfg = config["logger"]
    logger  = CSVLogger(log_cfg["log_path"])

    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(ModelCheckpoint(
            monitor=config["ckpt_args"]["metric"],
            dirpath=config["ckpt_args"]["ckpt_dir"],
            filename="model-{epoch:02d}",
            mode=config["ckpt_args"]["mode"],
            save_top_k=1,
        ))
    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(EarlyStopping(
            monitor=config["early_stopping"]["metric"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        ))
    return logger, callbacks


# ── Train ──────────────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config.get("seed", 42))
    train_dl, val_dl = load_data(config)
    model  = AVH_Sup(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--seed",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    print(f"\n{'='*68}")
    print(f"  AVH-Align/sup baseline — FAVC training")
    print(f"  model_type={config['model_hparams']['model_type']}  "
          f"input_type={config['model_hparams']['input_type']}")
    print(f"  seed={config.get('seed', 42)}  epochs={config['epochs']}")
    print(f"{'='*68}\n")

    train(config)
