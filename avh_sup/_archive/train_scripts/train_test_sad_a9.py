"""
A9 — SAD + Multi-Kernel MMD domain alignment training entry-point.

Usage:
  python train_test_sad_a9.py --config_path configs/A9.yaml --mode train

Key difference from train_test_sad_a8.py:
  • batch_size comes from config (default 16) instead of hardcoded 1.
  • FAVC val-real clips are oversampled by config['favc_oversample']
    (default 14) so that mini-batches regularly contain ≥2 FAVC clips,
    which is required for the per-batch MMD estimate to be non-trivial.
  • Model class is AVH_SAD_A9 (MMD on Z_sync) instead of AVH_SAD_A8 (GRL).
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
from mlp_sad_a9 import AVH_SAD_A9


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Domain-labelled wrapper datasets  (identical to A6/A7/A8)
# ─────────────────────────────────────────────────────────────────────────────

class AV1M_DomainWrapper(Dataset):
    """Wraps AV1M_trainval_dataset to add domain label (0 = AV1M)."""
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.base[idx]
        domain_label = torch.tensor(0.0)
        return video, audio, cls_label, domain_label, path


class FAVC_RealDomainDataset(Dataset):
    """
    FAVC val-real clips with domain label (1 = FAVC).
      cls_label    = -1     (no task supervision for FAVC)
      domain_label = 1.0
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
        feats = np.load(abs_path, allow_pickle=True)
        video = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)
        if self.apply_l2:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-8)
            audio = audio / (np.linalg.norm(audio, axis=-1, keepdims=True) + 1e-8)
        return torch.tensor(video), torch.tensor(audio), -1, torch.tensor(1.0), rel_path


# ─────────────────────────────────────────────────────────────────────────────
#  Collate (handles variable-length clips)
# ─────────────────────────────────────────────────────────────────────────────

def _pad_collate(batch):
    """
    Pad variable-length [T, feat_dim] clips to the maximum T in the batch.
    Returns: (video [B,T,D], audio [B,T,D], cls_labels [B], domain_labels [B], paths)
    """
    videos, audios, cls_lbs, dom_lbs, paths = zip(*batch)

    max_t = max(v.shape[0] for v in videos)

    def _pad(seqs):
        out = []
        for s in seqs:
            t, d = s.shape
            if t < max_t:
                pad = torch.zeros(max_t - t, d, dtype=s.dtype)
                s = torch.cat([s, pad], dim=0)
            out.append(s)
        return torch.stack(out)   # [B, max_T, D]

    videos_pad = _pad(videos)
    audios_pad = _pad(audios)
    cls_lbs    = torch.tensor(cls_lbs,   dtype=torch.long)
    dom_lbs    = torch.stack(dom_lbs)
    return videos_pad, audios_pad, cls_lbs, dom_lbs, list(paths)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: dict):
    data_cfg        = config["data_info"]
    favc_cfg        = config["favc_domain_info"]
    batch_size      = int(config.get("batch_size", 16))
    favc_oversample = int(config.get("favc_oversample", 14))

    av1m_train_base = AV1M_trainval_dataset(data_cfg, split="train")
    av1m_train      = AV1M_DomainWrapper(av1m_train_base)
    favc_real       = FAVC_RealDomainDataset(favc_cfg)

    # Oversample FAVC so each batch is likely to contain ≥2 FAVC clips.
    # This makes per-batch MMD non-degenerate.
    if favc_oversample > 1 and len(favc_real) > 0:
        favc_repeated = ConcatDataset([favc_real] * favc_oversample)
    else:
        favc_repeated = favc_real

    train_ds = ConcatDataset([av1m_train, favc_repeated])
    val_ds   = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=_pad_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,   # avoids degenerate last batch for MMD
    )
    val_dl = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=1,
    )

    print(
        f"Train: {len(av1m_train)} AV1M  +  "
        f"{len(favc_real)} FAVC-real × {favc_oversample} = {len(favc_repeated)} "
        f"oversampled  →  {len(train_ds)} total  "
        f"(batch_size={batch_size}, steps/epoch≈{len(train_ds)//batch_size})\n"
        f"Val (AV1M): {len(val_ds)}",
        flush=True,
    )
    return train_dl, val_dl


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def init_callbacks(config: dict):
    log_cfg  = config["logger"]
    logger   = CSVLogger(log_cfg["log_path"])
    callbacks = []

    if "ckpt_args" in config:
        callbacks.append(ModelCheckpoint(
            monitor=config["ckpt_args"]["metric"],
            dirpath=config["ckpt_args"]["ckpt_dir"],
            filename="model-{epoch:02d}-{val_auc_causal:.4f}",
            mode=config["ckpt_args"]["mode"],
            save_top_k=3,
        ))

    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(EarlyStopping(
            monitor=config["early_stopping"]["metric"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        ))

    return logger, callbacks


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model  = AVH_SAD_A9(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,   # protect against large MMD gradients
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", default="train", choices=["train"])
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.mode == "train":
        train(config)
