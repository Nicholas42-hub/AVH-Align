"""
A6-FAVC-PseudoDomain — FCD + Domain Push-Pull, FAVC speaker-group pseudo-labels.

Trains entirely on FakeAVCeleb (train split) and validates on FAVC val split.
Domain labels are derived from FAVC speaker IDs via hash:
    domain = hash(speaker_id) % 2  →  0 or 1

This is the in-domain counterpart to the AV1M-trained TriRoute model.
It uses the same A6 architecture (AVH_FCD_A6 from mlp_fcd_a6.py) and
speaker-group pseudo-domain labels within FAVC.

All model architecture and losses are identical to A6 (mlp_fcd_a6.py).
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import lightning as L

from datasets import FakeAVCeleb_NPZ_Dataset
from mlp_fcd_a6 import AVH_FCD_A6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


class FAVC_SpeakerGroupDomainDataset(Dataset):
    """
    FakeAVCeleb (train split) clips with binary pseudo-domain labels from speaker ID.

    Domain supervision uses REAL clips only (label==0), matching the AV1M A6 design
    where only real FAVC clips carry domain labels. Fake clips get domain_label=-1
    so mlp_fcd_a6.py domain_mask skips them — avoiding correlation between domain
    and classification gradients.

    Domain assignment (real clips only):
        hash(speaker_id) % 2 == 0  →  domain 0.0  (speaker group A)
        hash(speaker_id) % 2 == 1  →  domain 1.0  (speaker group B)
    Fake clips: domain_label = -1.0  (excluded from domain loss)

    speaker_id is parsed from rel_path: e.g.,
        "RealVideo-RealAudio/African/men/id00166/00010_xxx.npz"
        → parts[3] = "id00166"

    Returns 5-tuple: (video, audio, cls_label, domain_label, rel_path)
    """

    def __init__(self, config: dict):
        self._base = FakeAVCeleb_NPZ_Dataset(config, split="train")
        n_groupA = n_groupB = n_fake = 0
        for _, rel_path, label, _ in self._base.items:
            if label == 1:  # fake clip — no domain supervision
                n_fake += 1
                continue
            parts = rel_path.replace("\\", "/").split("/")
            speaker_id = parts[3] if len(parts) >= 4 else rel_path
            if hash(speaker_id) % 2 == 0:
                n_groupA += 1
            else:
                n_groupB += 1
        print(
            f"FAVC train speaker-group domain dataset (real-only): "
            f"{n_groupA} group-A (d=0), {n_groupB} group-B (d=1), "
            f"{n_fake} fake clips (d=-1, no domain supervision), "
            f"total {len(self._base)}",
            flush=True,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        video, audio, label, rel_path = self._base[idx]
        if label == 1:  # fake clip — excluded from domain supervision
            domain_label = -1.0
        else:
            parts = rel_path.replace("\\", "/").split("/")
            speaker_id = parts[3] if len(parts) >= 4 else rel_path
            domain_label = float(hash(speaker_id) % 2)
        return video, audio, label, torch.tensor(domain_label), rel_path


def load_data(config: dict):
    data_cfg = config["data_info"]

    train_ds = FAVC_SpeakerGroupDomainDataset(data_cfg)
    val_ds   = FakeAVCeleb_NPZ_Dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train (FAVC speaker-group domain): {len(train_ds)} clips   "
        f"Val (FAVC): {len(val_ds)} clips",
        flush=True,
    )
    return train_dl, val_dl


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


def train(config: dict):
    set_seed(config.get("seed", 42))
    train_dl, val_dl = load_data(config)
    model  = AVH_FCD_A6(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


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
    print(f"  A6-FAVC-PseudoDomain — FAVC Speaker-Group Domain Push-Pull")
    print(f"  syn_dim={config['model_hparams']['syn_dim']}  "
          f"spec_dim={config['model_hparams']['spec_dim']}  "
          f"lr={config['model_hparams']['lr']}")
    print(f"{'='*68}\n")

    train(config)
