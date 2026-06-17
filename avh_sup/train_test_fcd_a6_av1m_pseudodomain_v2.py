"""
A6-AV1M-PseudoDomain-v2 — FCD + Domain Push-Pull, AV1M speaker-group.

Controlled ablation for the main claim that FAVC real clips provide a unique
cross-domain supervision signal.  The experimental setup is identical to A6
in every way except the source of domain labels:

    A6              : domain = AV1M (0)  vs  FAVC real clips (1)
    This ablation   : domain = speaker-group-A (0)  vs  speaker-group-B (1)
                      derived from AV1M clips (all real + fake)

Binary speaker-group assignment:
    speaker_id = path.split("/")[0]   (e.g. "id02148")
    domain = stable_hash(speaker_id) % 2     →  0 or 1

v2 fix: includes ALL AV1M clips (real + fake) so the task head trains on
both classes, matching the training regime of every other A6 variant.
Domain labels are derived from speaker hash for all clips.

Expected result: ≤ A5, because speaker-group nuisance is intra-source and
does not capture the AV1M→FAVC codec/acquisition shift.  If this falls
below A5, it proves that FAVC real clips provide target-specific signal
beyond generic domain regularisation.

All model architecture and losses are identical to A6 (mlp_fcd_a6.py).
"""

import argparse
import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import lightning as L

from datasets import FakeAVCeleb_NPZ_Dataset
from mlp_fcd_a6 import AVH_FCD_A6


def stable_speaker_group(speaker_id: str) -> int:
    """Deterministic 0/1 speaker partition, independent of PYTHONHASHSEED."""
    digest = hashlib.md5(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 2


def pseudo_domain_label(path: str, mode: str, seed: int) -> float:
    """Return a deterministic binary pseudo-domain label for ablations."""
    speaker_id = path.split("/")[0]
    if mode == "speaker_hash":
        value = stable_speaker_group(speaker_id)
    elif mode == "random_speaker":
        digest = hashlib.md5(f"random_speaker:{seed}:{speaker_id}".encode("utf-8")).hexdigest()
        value = int(digest, 16) % 2
    elif mode == "random_clip":
        digest = hashlib.md5(f"random_clip:{seed}:{path}".encode("utf-8")).hexdigest()
        value = int(digest, 16) % 2
    else:
        raise ValueError(
            f"Unknown domain_label_mode={mode!r}. "
            "Expected one of: speaker_hash, random_speaker, random_clip."
        )
    return float(value)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


class AV1M_SpeakerGroupDomainDataset(Dataset):
    """
    AV1M clips (real AND fake) with binary pseudo-domain labels from speaker ID.

    All clips are included so the task head trains on both classes.
    Domain labels are derived from speaker hash for all clips, mirroring
    the A6 design (domain = 0 or 1 assigned per clip).

    Domain assignment:
        stable_hash(speaker_id) % 2 == 0  →  domain 0.0  (speaker group A)
        stable_hash(speaker_id) % 2 == 1  →  domain 1.0  (speaker group B)

    where speaker_id = path.split("/")[0]  (e.g. "id02148").

    Reads from train_labels.csv (columns: path, label).
    """

    def __init__(self, config: dict, split: str = "train"):
        self.root_path = config["root_path"]
        self.apply_l2  = config.get("apply_l2", False)
        self.domain_label_mode = config.get("domain_label_mode", "speaker_hash")
        self.domain_label_seed = int(config.get("domain_label_seed", 0))
        csv_root       = config["csv_root_path"]

        df = pd.read_csv(os.path.join(csv_root, f"{split}_labels.csv"))
        self.df = df

        self.feats_dir = os.path.join(self.root_path, split)

        n_real   = (df["label"] == 0).sum()
        n_fake   = (df["label"] == 1).sum()
        n_groupA = sum(pseudo_domain_label(p, self.domain_label_mode, self.domain_label_seed) == 0.0 for p in df["path"])
        n_groupB = len(df) - n_groupA
        print(
            f"AV1M {split} speaker-group domain dataset (all clips): "
            f"{n_real} real, {n_fake} fake | "
            f"{n_groupA} group-A (d=0), {n_groupB} group-B (d=1), "
            f"total {len(df)} | mode={self.domain_label_mode}, seed={self.domain_label_seed}",
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

        domain_label = pseudo_domain_label(path, self.domain_label_mode, self.domain_label_seed)

        return (
            torch.tensor(video),
            torch.tensor(audio),
            int(row["label"]),           # 0 (real) or 1 (fake)
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
        self.domain_label_mode = config.get("domain_label_mode", "speaker_hash")
        self.domain_label_seed = int(config.get("domain_label_seed", 0))
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

        domain_label = pseudo_domain_label(path, self.domain_label_mode, self.domain_label_seed)

        return (
            torch.tensor(video),
            torch.tensor(audio),
            int(row["label"]),
            torch.tensor(domain_label),
            path,
        )


def load_data(config: dict):
    data_cfg = config["data_info"]

    train_ds = AV1M_SpeakerGroupDomainDataset(data_cfg, split="train")
    val_ds   = AV1M_ValDataset(data_cfg)

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(f"Train (all AV1M, speaker-group domain): {len(train_ds)} clips   Val: {len(val_ds)} AV1M clips", flush=True)
    return train_dl, val_dl


class FAVCFvraCallback(L.Callback):
    FVRA = "FakeVideo-RealAudio"
    REAL = "RealVideo-RealAudio"

    def __init__(self, favc_eval_cfg: dict, check_every_n_epochs: int = 5, best_ckpt_dir: str = None):
        super().__init__()
        self.cfg = favc_eval_cfg
        self.check_every_n_epochs = check_every_n_epochs
        self.best_ckpt_dir = best_ckpt_dir
        self.best_auc = float("-inf")
        self._loader = None

    def setup(self, trainer, pl_module, stage):
        if stage != "fit" or self._loader is not None:
            return
        ds = FakeAVCeleb_NPZ_Dataset(self.cfg, split=self.cfg.get("split", "test"))
        self._loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        print(
            f"[FAVCFvraCallback] FAVC eval dataset: {len(ds)} clips "
            f"(split={self.cfg.get('split', 'test')})",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        if (epoch + 1) % self.check_every_n_epochs != 0 or self._loader is None:
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

        fvra_mask = [(self.FVRA in p) or (self.REAL in p) for p in all_paths]
        fvra_scores = [s for s, m in zip(all_scores, fvra_mask) if m]
        fvra_labels = [lbl for lbl, m in zip(all_labels, fvra_mask) if m]

        try:
            fvra_auc = roc_auc_score(y_true=fvra_labels, y_score=fvra_scores)
        except ValueError:
            fvra_auc = float("nan")

        pl_module.log("favc_fvra_auc", fvra_auc, prog_bar=True)
        print(
            f"\n[Epoch {epoch}] FAVC FV-RA AUC (causal) = {fvra_auc:.4f}"
            f"  [{sum(fvra_mask)} clips]",
            flush=True,
        )

        if self.best_ckpt_dir is None or not np.isfinite(fvra_auc) or fvra_auc <= self.best_auc:
            return

        self.best_auc = fvra_auc
        os.makedirs(self.best_ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.best_ckpt_dir,
            f"best-favc-fvra-epoch={epoch:02d}-auc={fvra_auc:.4f}.ckpt",
        )
        trainer.save_checkpoint(ckpt_path)
        with open(os.path.join(self.best_ckpt_dir, "best_favc_fvra.txt"), "w") as f:
            f.write(f"best_auc={fvra_auc:.8f}\n")
            f.write(f"epoch={epoch}\n")
            f.write(f"checkpoint={ckpt_path}\n")
        print(f"[FAVCFvraCallback] Saved new best FV-RA checkpoint: {ckpt_path}", flush=True)


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
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=config.get("ckpt_path"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt_path", default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path

    ablation_id = config.get("ablation_id", "A6_av1m_pseudodomain_v2")
    seed        = config.get("seed", 43)

    print("\n" + "=" * 68)
    print(f"  {ablation_id} — FCD + Domain Push-Pull (AV1M all-clips speaker-group) v2")
    print(f"  seed={seed}")
    data_cfg = config.get("data_info", {})
    print(f"  domain_label_mode={data_cfg.get('domain_label_mode', 'speaker_hash')}")
    print(f"  domain_label_seed={data_cfg.get('domain_label_seed', 0)}")
    print(f"  task labels:   0 (real) / 1 (fake) — ALL clips included")
    if config.get("ckpt_path"):
        print(f"  resume:        {config['ckpt_path']}")
    print("=" * 68 + "\n")

    train(config)
