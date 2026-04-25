"""
DANN-UDA and CORAL-UDA training entry-point.

Usage (DANN):
  python train_da_baselines.py \
      --method dann \
      --config_path configs/DANN.yaml \
      --seed 43 \
      --output_dir outputs_DANN_seed43

Usage (CORAL):
  python train_da_baselines.py \
      --method coral \
      --config_path configs/CORAL.yaml \
      --seed 43 \
      --output_dir outputs_CORAL_seed43

Both methods share:
  - AV1M train (domain=0, task-labelled) + FAVC val-real (domain=1, no task label)
  - AV1M val used for early stopping (val_auc)
  - Identical dataset/callback setup to A6

CORAL-specific:
  - Collate function mean-pools variable-length clips so batch_size > 1 works.
  - Default batch_size=16 (8 src + 8 tgt per step on average).
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
import pandas as pd

from datasets import AV1M_trainval_dataset


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed: {seed}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Domain-labelled wrapper datasets (identical to train_test_fcd_a6.py)
# ─────────────────────────────────────────────────────────────────────────────

class AV1M_DomainWrapper(Dataset):
    """Adds domain=0 label to AV1M samples → 5-tuple."""
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.base[idx]
        return video, audio, cls_label, torch.tensor(0.0), path


class FAVC_RealDomainDataset(Dataset):
    """FAVC val real-only clips; cls_label=-1, domain=1."""
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
            print(f"WARNING: FAVC real dir not found: {cat_dir}", flush=True)
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
#  Collate functions
# ─────────────────────────────────────────────────────────────────────────────

def collate_temporal(batch):
    """
    Standard collate for variable-length clips.
    Returns batch_size=1 tensors per sample (no padding), so DataLoader
    batch_size should be 1.
    Each item: (video [T,1024], audio [T,1024], cls_label, domain_label, path)
    → unsqueeze to (video [1,T,1024], ...) for model forward.
    """
    videos, audios, cls_labels, dom_labels, paths = zip(*batch)
    # All items have same shape since DataLoader batch_size=1
    video = videos[0].unsqueeze(0)   # [1, T, 1024]
    audio = audios[0].unsqueeze(0)
    cls   = torch.tensor([cls_labels[0]])
    dom   = torch.tensor([dom_labels[0]])
    return video, audio, cls, dom, list(paths)


def collate_pooled(batch):
    """
    Collate for CORAL: mean-pool over T so clips become fixed [1024].
    Supports batch_size > 1.
    """
    videos, audios, cls_labels, dom_labels, paths = zip(*batch)
    v_pooled = torch.stack([v.mean(dim=0) if v.dim() == 2 else v for v in videos])
    a_pooled = torch.stack([a.mean(dim=0) if a.dim() == 2 else a for a in audios])
    cls = torch.tensor(cls_labels)
    dom = torch.tensor([float(d) for d in dom_labels])
    return v_pooled, a_pooled, cls, dom, list(paths)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: dict, method: str):
    data_cfg = config["data_info"]
    favc_cfg = config["favc_domain_info"]
    bs       = config.get("batch_size", 1)

    av1m_train_base = AV1M_trainval_dataset(data_cfg, split="train")
    av1m_train      = AV1M_DomainWrapper(av1m_train_base)
    favc_real       = FAVC_RealDomainDataset(favc_cfg)
    train_ds        = ConcatDataset([av1m_train, favc_real])

    av1m_val = AV1M_trainval_dataset(data_cfg, split="val")

    if method == "coral":
        collate_fn = collate_pooled
        # Validation uses pooled too (CORAL model handles both)
        val_dl = DataLoader(av1m_val,  shuffle=False, batch_size=bs,
                            collate_fn=collate_pooled)
        # For validation 5-tuple, we need AV1M_DomainWrapper
        av1m_val_wrapped = AV1M_DomainWrapper(av1m_val)
        val_dl = DataLoader(av1m_val_wrapped, shuffle=False, batch_size=bs,
                            collate_fn=collate_pooled)
    else:
        collate_fn = collate_temporal
        av1m_val_wrapped = AV1M_DomainWrapper(av1m_val)
        val_dl = DataLoader(av1m_val_wrapped, shuffle=False, batch_size=1,
                            collate_fn=collate_temporal)

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=bs,
                          collate_fn=collate_fn)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(favc_real)} FAVC-real = "
        f"{len(train_ds)} total   Val(AV1M): {len(av1m_val)}  "
        f"batch_size={bs}  collate={collate_fn.__name__}",
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
            filename="model-{epoch:02d}",
            mode=config["ckpt_args"]["mode"],
            save_top_k=1,
        ))
    if "early_stopping" in config and config["early_stopping"] is not None:
        es = config["early_stopping"]
        callbacks.append(EarlyStopping(
            monitor=es["metric"],
            mode=es["mode"],
            patience=es["patience"],
        ))
    return logger, callbacks


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",      required=True, choices=["dann", "coral"])
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--output_dir",  type=str, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    if args.output_dir is not None:
        base = args.output_dir
        config["callbacks"]["logger"]["log_path"]       = os.path.join(base, "logs")
        config["callbacks"]["ckpt_args"]["ckpt_dir"]    = os.path.join(base, "ckpts")
        for d in ["logs", "ckpts"]:
            os.makedirs(os.path.join(base, d), exist_ok=True)

    set_seed(config.get("seed", 43))

    print(f"\n{'='*60}")
    print(f"  {args.method.upper()}-UDA  seed={config.get('seed', 43)}")
    print(f"{'='*60}\n")

    if args.method == "dann":
        from mlp_dann import DANN_UDA
        model_cls = DANN_UDA
    else:
        from mlp_coral import CORAL_UDA
        model_cls = CORAL_UDA

    train_dl, val_dl = load_data(config, args.method)
    model    = model_cls(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
