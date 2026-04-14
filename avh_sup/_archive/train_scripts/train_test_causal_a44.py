"""
A44 — Causal + Domain Adversarial training entry-point (frozen encoder).

Training data:
  - AV1M train clips    (domain=0, task-labelled)
  - FAVC val-real clips (domain=1, cls_label=-1 → task losses masked)

Usage:
  python train_test_causal_a44.py --config_path configs/A44.yaml --mode train
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

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_causal_a44 import AVH_Causal_A44


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


# ── Domain-labelled wrapper datasets ──────────────────────────────────────────

class AV1M_DomainWrapper(Dataset):
    """Wraps AV1M_trainval_dataset, adds domain=0 label."""
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.base[idx]
        return video, audio, cls_label, torch.tensor(0.0), path


class FAVC_RealDomainDataset(Dataset):
    """
    FAVC val-real clips with domain=1, cls_label=-1 (no task supervision).
    Returns 5-tuple: (video, audio, cls_label, domain_label, path)
    """

    REAL_CATEGORY = "RealVideo-RealAudio"

    def __init__(self, config: dict):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        split          = config.get("split", "val")

        csv_path = os.path.join(config["csv_root_path"], f"{split}_split.csv")
        df = pd.read_csv(csv_path)
        real_df = df[df["full_path"].str.contains(self.REAL_CATEGORY)]

        allowed = set()
        for _, row in real_df.iterrows():
            base_path = row["full_path"].replace("FakeAVCeleb/", "").replace(".mp4", "")
            parts = base_path.rsplit("/", 1)
            if len(parts) == 2:
                allowed.add(tuple(parts))

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

        print(f"FAVC val-real domain clips: {len(self.items)}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        abs_path, rel_path = self.items[idx]
        data = np.load(abs_path)
        # Frozen feature files use keys 'visual'/'audio' (not 'video_features'/'audio_features')
        video = torch.from_numpy(data["visual"]).float()  # [T, 1024]
        audio = torch.from_numpy(data["audio"]).float()   # [T, 1024]
        if self.apply_l2:
            video = torch.nn.functional.normalize(video, dim=-1)
            audio = torch.nn.functional.normalize(audio, dim=-1)
        return video, audio, torch.tensor(-1), torch.tensor(1.0), rel_path


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(config: dict):
    data_cfg = config["data_info"]
    favc_cfg = config["favc_domain_info"]

    av1m_train_base = AV1M_trainval_dataset(data_cfg, split="train")
    av1m_train      = AV1M_DomainWrapper(av1m_train_base)

    favc_real = FAVC_RealDomainDataset(favc_cfg)

    favc_oversample = config.get("favc_oversample", 1)
    if favc_oversample > 1:
        from torch.utils.data import ConcatDataset as CD
        favc_real = CD([favc_real] * favc_oversample)
        print(f"FAVC oversampled x{favc_oversample} → {len(favc_real)} clips", flush=True)

    train_ds = ConcatDataset([av1m_train, favc_real])
    val_ds   = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(favc_real)} FAVC-real = "
        f"{len(train_ds)} total   Val(AV1M): {len(val_ds)}",
        flush=True,
    )
    return train_dl, val_dl


# ── Callbacks ─────────────────────────────────────────────────────────────────

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


# ── Train ─────────────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model  = AVH_Causal_A44(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(
        max_epochs        = config.get("epochs", 100),
        logger            = logger,
        callbacks         = callbacks,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        devices           = 1,
        log_every_n_steps = 10,
        enable_progress_bar = False,
    )
    trainer.fit(model, train_dl, val_dl)


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", default="train", choices=["train"])
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        train(config)


if __name__ == "__main__":
    main()
