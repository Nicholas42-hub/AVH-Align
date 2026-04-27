"""
A6 — FCD + Domain Adversarial training entry-point for AVH-Align.

Usage:
  python train_test_fcd_a6.py --config_path configs/A6.yaml --mode train
  python train_test_fcd_a6.py --config_path configs/A6.yaml --mode train --seed 42 \
      --output_dir /path/to/outputs_A6_seed42

Key difference from A5: trains on a mixed dataset:
  - AV1M train clips   (domain=0, task-labelled)
  - FAVC val-real clips (domain=1, no task label → cls_label=-1)

Domain adversarial losses operate on every batch; task losses are masked
to AV1M clips only (cls_label >= 0).
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

from sklearn.metrics import roc_auc_score

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_fcd_a6 import AVH_FCD_A6


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
#  Domain-labelled wrapper datasets
# ─────────────────────────────────────────────────────────────────────────────

class AV1M_DomainWrapper(Dataset):
    """
    Wraps AV1M_trainval_dataset to add domain label (0 = AV1M).
    Returns 5-tuple: (video, audio, cls_label, domain_label, path)
    """
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
    Loads FAVC val real-only clips and assigns:
      cls_label   = -1   (no task supervision)
      domain_label = 1.0 (FAVC domain)
    Returns 5-tuple: (video, audio, cls_label, domain_label, path)
    """

    REAL_CATEGORY = "RealVideo-RealAudio"

    def __init__(self, config: dict):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        split          = config.get("split", "val")

        # Load split CSV and filter to real-only
        csv_path = os.path.join(config["csv_root_path"], f"{split}_split.csv")
        df = pd.read_csv(csv_path)
        real_df = df[df["full_path"].str.contains(self.REAL_CATEGORY)]

        # Build allowed set (same format as FakeAVCeleb_NPZ_Dataset)
        allowed = set()
        for _, row in real_df.iterrows():
            base_path = row["full_path"].replace("FakeAVCeleb/", "").replace(".mp4", "")
            parts = base_path.rsplit("/", 1)
            if len(parts) == 2:
                allowed.add(tuple(parts))

        # Walk feature directory to find matching .npz files
        self.items = []  # (abs_path, rel_path)
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
        # cls_label = -1 (skip task loss), domain_label = 1.0
        return torch.tensor(video), torch.tensor(audio), -1, torch.tensor(1.0), rel_path


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config: dict):
    data_cfg  = config["data_info"]
    favc_cfg  = config["favc_domain_info"]

    # AV1M train (wrapped with domain=0)
    av1m_train_base = AV1M_trainval_dataset(data_cfg, split="train")
    av1m_train      = AV1M_DomainWrapper(av1m_train_base)

    # FAVC val real (domain=1, no task label)
    favc_real = FAVC_RealDomainDataset(favc_cfg)

    # Mixed train set: AV1M + FAVC real
    train_ds = ConcatDataset([av1m_train, favc_real])

    # AV1M val (4-tuple — handled by validation_step directly)
    val_ds = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(favc_real)} FAVC-real = "
        f"{len(train_ds)} total   Val(AV1M): {len(val_ds)}",
        flush=True,
    )
    return train_dl, val_dl


# ─────────────────────────────────────────────────────────────────────────────
#  FAVC FV-RA monitoring callback
# ─────────────────────────────────────────────────────────────────────────────

class FAVCFvraCallback(L.Callback):
    """
    Periodically evaluates FV-RA AUC on FakeAVCeleb test set during training.

    Logs `favc_fvra_auc` every `check_every_n_epochs` epochs.
    Uses the causal head for scoring, same as the paper's evaluation protocol.
    Monitoring only — does not affect checkpoint selection by default.

    Config keys (favc_eval_info section):
      root_path            : path to favc_features dir
      csv_root_path        : path to dir with {split}_split.csv files
      split                : CSV split to load (default: "test")
      apply_l2             : whether to L2-normalise features (default: True)
      check_every_n_epochs : eval frequency (default: 5)
    """

    FVRA = "FakeVideo-RealAudio"
    REAL = "RealVideo-RealAudio"

    def __init__(
        self,
        favc_eval_cfg: dict,
        check_every_n_epochs: int = 5,
        best_ckpt_dir: str = None,
    ):
        super().__init__()
        self.cfg = favc_eval_cfg
        self.check_every_n_epochs = check_every_n_epochs
        self.best_ckpt_dir = best_ckpt_dir
        self.best_auc = float("-inf")
        self.best_path = None
        self._loader = None

    def setup(self, trainer, pl_module, stage):
        if stage != "fit" or self._loader is not None:
            return
        ds = FakeAVCeleb_NPZ_Dataset(self.cfg, split=self.cfg.get("split", "test"))
        self._loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        print(
            f"[FAVCFvraCallback] FAVC eval dataset: {len(ds)} clips  "
            f"(split={self.cfg.get('split', 'test')})",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if (epoch + 1) % self.check_every_n_epochs != 0:
            return
        if self._loader is None:
            return

        device = next(pl_module.parameters()).device
        was_training = pl_module.training
        pl_module.eval()

        all_scores, all_labels, all_paths = [], [], []
        with torch.no_grad():
            for video, audio, labels, paths in self._loader:
                video = video.to(device)
                audio = audio.to(device)
                score = pl_module.predict_scores(video, audio, mode="causal")
                all_scores.extend(score.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_paths.extend(list(paths))

        if was_training:
            pl_module.train()

        # FV-RA AUC: FakeVideo-RealAudio vs RealVideo-RealAudio
        fvra_mask = [
            (self.FVRA in p) or (self.REAL in p)
            for p in all_paths
        ]
        fvra_scores = [s for s, m in zip(all_scores, fvra_mask) if m]
        fvra_labels = [lbl for lbl, m in zip(all_labels, fvra_mask) if m]

        try:
            fvra_auc = roc_auc_score(y_true=fvra_labels, y_score=fvra_scores)
        except ValueError:
            fvra_auc = float("nan")

        pl_module.log("favc_fvra_auc", fvra_auc, prog_bar=True)
        print(
            f"\n[Epoch {epoch}] FAVC FV-RA AUC (causal) = {fvra_auc:.4f}"
            f"  [{sum(fvra_mask)} clips: "
            f"{sum(l == 1 for l, m in zip(all_labels, fvra_mask) if m)} fake + "
            f"{sum(l == 0 for l, m in zip(all_labels, fvra_mask) if m)} real]",
            flush=True,
        )

        if self.best_ckpt_dir is None or not np.isfinite(fvra_auc):
            return

        if fvra_auc > self.best_auc:
            self.best_auc = fvra_auc
            os.makedirs(self.best_ckpt_dir, exist_ok=True)
            ckpt_name = f"best-favc-fvra-epoch={epoch:02d}-auc={fvra_auc:.4f}.ckpt"
            ckpt_path = os.path.join(self.best_ckpt_dir, ckpt_name)
            trainer.save_checkpoint(ckpt_path)
            self.best_path = ckpt_path
            marker_path = os.path.join(self.best_ckpt_dir, "best_favc_fvra.txt")
            with open(marker_path, "w") as f:
                f.write(f"best_auc={fvra_auc:.8f}\n")
                f.write(f"epoch={epoch}\n")
                f.write(f"checkpoint={ckpt_path}\n")
            print(
                f"[FAVCFvraCallback] Saved new best FV-RA checkpoint: {ckpt_path}",
                flush=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict):
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model  = AVH_FCD_A6(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])

    # Optional: periodic FAVC FV-RA monitoring (add callback when config present)
    if "favc_eval_info" in config:
        n_freq = int(config["favc_eval_info"].get("check_every_n_epochs", 5))
        best_ckpt_dir = config["favc_eval_info"].get(
            "best_ckpt_dir",
            os.path.join(config["callbacks"]["ckpt_args"]["ckpt_dir"], "favc_fvra"),
        )
        callbacks.append(
            FAVCFvraCallback(
                config["favc_eval_info"],
                check_every_n_epochs=n_freq,
                best_ckpt_dir=best_ckpt_dir,
            )
        )
        print(f"[train] FAVCFvraCallback enabled (every {n_freq} epochs)", flush=True)

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override logger/ckpt/results base directory")
    parser.add_argument("--favc_domain_split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Override FAVC domain split (default from config)")
    parser.add_argument("--favc_domain_root_path", type=str, default=None,
                        help="Override FAVC feature root for domain data")
    parser.add_argument("--favc_domain_csv_root_path", type=str, default=None,
                        help="Override FAVC csv root for domain data")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    # CLI overrides for submission sweeps / leakage-control ablations
    if args.seed is not None:
        config["seed"] = args.seed

    if args.output_dir is not None:
        base = args.output_dir
        config["callbacks"]["logger"]["log_path"] = os.path.join(base, "logs")
        config["callbacks"]["ckpt_args"]["ckpt_dir"] = os.path.join(base, "ckpts")
        config["output_path"] = os.path.join(base, "results")
        os.makedirs(os.path.join(base, "logs"), exist_ok=True)
        os.makedirs(os.path.join(base, "ckpts"), exist_ok=True)
        os.makedirs(os.path.join(base, "results"), exist_ok=True)

    favc_cfg = config.setdefault("favc_domain_info", {})
    if args.favc_domain_split is not None:
        favc_cfg["split"] = args.favc_domain_split
    if args.favc_domain_root_path is not None:
        favc_cfg["root_path"] = args.favc_domain_root_path
    if args.favc_domain_csv_root_path is not None:
        favc_cfg["csv_root_path"] = args.favc_domain_csv_root_path

    print("\n" + "=" * 68)
    print("  A6 — FCD + Domain Push-Pull")
    print(f"  seed={config.get('seed', 43)}")
    print(f"  favc_domain_split={favc_cfg.get('split', 'val')}")
    print(f"  favc_domain_root={favc_cfg.get('root_path', '<missing>')}")
    print(f"  favc_domain_csv_root={favc_cfg.get('csv_root_path', '<missing>')}")
    if args.output_dir is not None:
        print(f"  output_dir={args.output_dir}")
    print("=" * 68 + "\n")

    if args.mode == "train":
        train(config)
