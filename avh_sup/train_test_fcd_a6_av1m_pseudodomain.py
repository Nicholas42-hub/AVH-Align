"""
A6-AV1M-PseudoDomain — FCD + Domain Push-Pull, AV1M real-only speaker-group.

Controlled ablation for the main claim that FAVC real clips provide a unique
cross-domain supervision signal.  The experimental setup is identical to A6
in every way except the source of domain labels:

    A6              : domain = AV1M (0)  vs  FAVC real clips (1)
    This ablation   : domain = speaker-group-A (0)  vs  speaker-group-B (1)
                      derived entirely from AV1M real-only clips

Training data (two streams, mixed via ConcatDataset):
    Task stream   : AV1M full train (real + fake), cls_label=0/1, domain_label=-1
    Domain stream : AV1M real-only, cls_label=-1, domain_label=hash(speaker_id)%2

Binary speaker-group assignment:
    speaker_id = path.split("/")[0]   (e.g. "id02148")
    domain = hash(speaker_id) % 2     →  0 or 1

Expected result: ≤ A5, because speaker-group nuisance is intra-source and
does not capture the AV1M→FAVC codec/acquisition shift.  If this falls
below A6, it proves that FAVC real clips provide target-specific signal
beyond generic domain regularisation.

All model architecture and losses are identical to A6 (mlp_fcd_a6.py).
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
from torch.utils.data import DataLoader, Dataset
import lightning as L

from mlp_fcd_a6 import AVH_FCD_A6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


class AV1M_TaskDataset(Dataset):
    """
    AV1M full train set (real + fake) for task supervision.
    cls_label = 0/1, domain_label = -1 (skips domain loss).
    """

    def __init__(self, config: dict, split: str = "train"):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        csv_root       = config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(csv_root, f"{split}_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, split)

        n_real = (self.df["label"] == 0).sum()
        n_fake = (self.df["label"] == 1).sum()
        print(
            f"AV1M {split} task dataset: {n_real} real, {n_fake} fake, total {len(self.df)}",
            flush=True,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row["path"]
        npz_path = os.path.join(self.feats_dir, path[:-4] + ".npz")
        feats = np.load(npz_path, allow_pickle=True)

        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)

        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)

        return (
            torch.tensor(video),
            torch.tensor(audio),
            int(row["label"]),           # 0 or 1
            torch.tensor(-1.0),          # no domain supervision
            path,
        )


class AV1M_SpeakerGroupDomainDataset(Dataset):
    """
    AV1M real-only clips with binary pseudo-domain labels derived from speaker ID.
    cls_label = -1 (skips task loss), domain_label = 0/1 (speaker group).

    Domain assignment:
        hash(speaker_id) % 2 == 0  →  domain 0.0  (speaker group A)
        hash(speaker_id) % 2 == 1  →  domain 1.0  (speaker group B)
    """

    def __init__(self, config: dict, split: str = "train"):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        csv_root       = config["csv_root_path"]

        df = pd.read_csv(os.path.join(csv_root, f"{split}_labels.csv"))
        df = df[df["label"] == 0].reset_index(drop=True)
        self.df = df

        self.feats_dir = os.path.join(self.root_path, split)

        n_groupA = sum(hash(p.split("/")[0]) % 2 == 0 for p in df["path"])
        n_groupB = len(df) - n_groupA
        print(
            f"AV1M {split} speaker-group domain dataset (real-only): "
            f"{n_groupA} group-A (d=0), {n_groupB} group-B (d=1), "
            f"total {len(df)}",
            flush=True,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row["path"]
        npz_path = os.path.join(self.feats_dir, path[:-4] + ".npz")
        feats = np.load(npz_path, allow_pickle=True)

        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)

        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)

        speaker_id   = path.split("/")[0]
        domain_label = float(hash(speaker_id) % 2)  # 0.0 or 1.0

        return (
            torch.tensor(video),
            torch.tensor(audio),
            -1,                          # skip task loss
            torch.tensor(domain_label),
            path,
        )


class AV1M_ValDataset(Dataset):
    """
    AV1M validation clips (no domain label; val is AV1M-only as in A6).
    """

    def __init__(self, config: dict):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        csv_root       = config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(csv_root, "val_labels.csv"))
        self.feats_dir = os.path.join(self.root_path, "val")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row["path"]
        npz_path = os.path.join(self.feats_dir, path[:-4] + ".npz")
        feats = np.load(npz_path, allow_pickle=True)

        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)

        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)

        return torch.tensor(video), torch.tensor(audio), int(row["label"]), path


def load_data(config: dict):
    data_cfg = config["data_info"]

    task_ds   = AV1M_TaskDataset(data_cfg, split="train")
    domain_ds = AV1M_SpeakerGroupDomainDataset(data_cfg, split="train")
    from torch.utils.data import ConcatDataset
    train_ds  = ConcatDataset([task_ds, domain_ds])
    val_ds    = AV1M_ValDataset(data_cfg)

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(task_ds)} task clips + {len(domain_ds)} domain clips = {len(train_ds)} total   "
        f"Val: {len(val_ds)} AV1M clips",
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
            save_last=True,
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
    ckpt_args = config.get("callbacks", {}).get("ckpt_args")
    resume_path = None
    if ckpt_args:
        candidate = os.path.join(ckpt_args["ckpt_dir"], "last.ckpt")
        if os.path.isfile(candidate):
            resume_path = candidate
            print(f"Resuming from {resume_path}", flush=True)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl,
                ckpt_path=resume_path)
    ckpt_args = config.get("callbacks", {}).get("ckpt_args")
    if ckpt_args:
        last_path = os.path.join(ckpt_args["ckpt_dir"], "last.ckpt")
        trainer.save_checkpoint(last_path)
        print(
            f"Saved final checkpoint to {last_path} "
            f"(epoch={trainer.current_epoch - 1}, global_step={trainer.global_step})",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    ablation_id = config.get("ablation_id", "A6_av1m_pseudodomain")
    seed        = config.get("seed", 43)

    print("\n" + "=" * 68)
    print(f"  {ablation_id} — FCD + Domain Push-Pull (AV1M real-only speaker-group)")
    print(f"  seed={seed}")
    print(f"  domain labels: hash(speaker_id)%2  →  0.0 (group A) / 1.0 (group B)")
    print("=" * 68 + "\n")

    train(config)
