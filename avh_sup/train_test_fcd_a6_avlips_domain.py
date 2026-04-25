"""
A6-AVLips-Domain — FCD + Domain Push-Pull, AVLips real clips as domain signal.

Domain labels:
    AV1M clips (real + fake)  →  domain 0   (source domain)
    AVLips real clips         →  domain 1   (third-party domain)

Key difference from A6-trainvalreal:
    - No FAVC clips used anywhere (no data leakage from target test set)
    - AVLips is a third dataset, fully disjoint from both AV1M and FakeAVCeleb
    - Diagnostic question: does domain push-pull help even without any FAVC exposure?

Expected result: FV-RA AUC > A5 (0.613) if the method generalises beyond FAVC supervision.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as L

from datasets import AV1M_trainval_dataset
from mlp_fcd_a6 import AVH_FCD_A6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


class AV1M_DomainWrapper(Dataset):
    """Wraps AV1M dataset and assigns domain label 0."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.base[idx]
        return video, audio, cls_label, torch.tensor(0.0), path


class AVLips_RealDomainDataset(Dataset):
    """
    AVLips real-only clips used solely as domain signal (domain label = 1).

    Reads individual .npz files from avlips_features/0_real/*.npz.
    Task label is set to -1 (not used in fake/real loss).
    """

    def __init__(self, config: dict):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)

        real_dir = os.path.join(self.root_path, "0_real")
        self.items = sorted([
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.endswith(".npz")
        ])
        print(f"AVLips real domain clips: {len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        abs_path = self.items[idx]
        feats = np.load(abs_path, allow_pickle=True)
        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        rel_path = os.path.relpath(abs_path, self.root_path)
        return torch.tensor(video), torch.tensor(audio), -1, torch.tensor(1.0), rel_path


def load_data(config: dict):
    data_cfg     = config["data_info"]
    avlips_cfg   = config["avlips_domain_info"]

    av1m_train  = AV1M_DomainWrapper(AV1M_trainval_dataset(data_cfg, split="train"))
    avlips_real = AVLips_RealDomainDataset(avlips_cfg)
    train_ds    = ConcatDataset([av1m_train, avlips_real])
    val_ds      = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(avlips_real)} AVLips-real "
        f"= {len(train_ds)} total   Val(AV1M): {len(val_ds)}",
        flush=True,
    )
    return train_dl, val_dl


def init_callbacks(config: dict):
    log_cfg   = config["logger"]
    logger    = CSVLogger(log_cfg["log_path"])
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


def train(config: dict):
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model  = AVH_FCD_A6(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    ablation_id = config.get("ablation_id", "A6_avlips_domain")
    print("\n" + "=" * 68)
    print(f"  {ablation_id} — FCD + Domain Push-Pull (AVLips real domain signal)")
    print(f"  seed={config.get('seed', 43)}")
    print(f"  domain: AV1M=0, AVLips-real=1   (zero FAVC exposure)")
    print("=" * 68 + "\n")

    if args.mode == "train":
        train(config)
