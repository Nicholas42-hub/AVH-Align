"""
A6-train+val-real — FCD + Domain Adversarial, FAVC train+val real support.

Uses both FAVC train_split.csv and val_split.csv real clips as domain signal.
Diagnostic question: does more support coverage fix the drop seen in A6_leakage_trainreal?

All model architecture and losses identical to A6. Only the domain data source changes.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
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
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.base[idx]
        return video, audio, cls_label, torch.tensor(0.0), path


class FAVC_TrainValRealDataset(Dataset):
    """
    FAVC real-only clips aggregated from BOTH train_split.csv and val_split.csv.
    Supports split in {"train", "val", "trainval"}.
    """

    REAL_CATEGORY = "RealVideo-RealAudio"

    def __init__(self, config: dict):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        split          = config.get("split", "val")

        csv_root = config["csv_root_path"]

        # Determine which CSV files to load
        if split == "trainval":
            splits_to_load = ["train", "val"]
        else:
            splits_to_load = [split]

        # Collect allowed (dir, basename) pairs from all CSV files
        allowed = set()
        for sp in splits_to_load:
            csv_path = os.path.join(csv_root, f"{sp}_split.csv")
            df = pd.read_csv(csv_path)
            real_df = df[df["full_path"].str.contains(self.REAL_CATEGORY)]
            for _, row in real_df.iterrows():
                base_path = row["full_path"].replace("FakeAVCeleb/", "").replace(".mp4", "")
                parts = base_path.rsplit("/", 1)
                if len(parts) == 2:
                    allowed.add(tuple(parts))

        # Walk feature directory to find matching .npz files
        self.items = []
        cat_dir = os.path.join(self.root_path, self.REAL_CATEGORY)
        if not os.path.isdir(cat_dir):
            print(f"WARNING: FAVC real-category dir not found: {cat_dir}", flush=True)
            return

        for dirpath, _, filenames in os.walk(cat_dir):
            for fname in sorted(filenames):
                if not fname.endswith(".npz"):
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, self.root_path)
                base_path = rel_path.replace(".npz", "")
                parts = base_path.rsplit("/", 1)
                if len(parts) != 2:
                    continue
                dir_p, full_fname = parts
                base_fname = full_fname.split("_")[0]
                for ad, an in allowed:
                    if dir_p == ad and (an == full_fname or an == base_fname
                                        or full_fname.startswith(an + "_")):
                        self.items.append((abs_path, rel_path))
                        break

        print(f"FAVC {split}-real domain clips: {len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        abs_path, rel_path = self.items[idx]
        feats = np.load(abs_path, allow_pickle=True)
        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(video), torch.tensor(audio), -1, torch.tensor(1.0), rel_path


def load_data(config: dict):
    data_cfg = config["data_info"]
    favc_cfg = config["favc_domain_info"]

    av1m_train = AV1M_DomainWrapper(AV1M_trainval_dataset(data_cfg, split="train"))
    favc_real  = FAVC_TrainValRealDataset(favc_cfg)
    train_ds   = ConcatDataset([av1m_train, favc_real])
    val_ds     = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(favc_real)} FAVC-real "
        f"= {len(train_ds)} total   Val(AV1M): {len(val_ds)}",
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

    print("\n" + "=" * 68)
    print("  A6-trainvalreal — FCD + Domain Push-Pull (train+val real support)")
    print(f"  seed={config.get('seed', 43)}")
    print(f"  favc_split=trainval (train_split.csv + val_split.csv)")
    print("=" * 68 + "\n")

    if args.mode == "train":
        train(config)
