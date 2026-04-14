"""
End-to-end training for AVH-Align causal disentanglement.

Unlike train_test_causal_ablation.py (which reads pre-extracted .npz features),
this script loads raw preprocessed video/audio files and fine-tunes the top
layers of AV-HuBERT jointly with the causal/spurious projection heads.

Architecture
------------
  AV-HuBERT (top N layers unfrozen)
       ↓ video_feats [B,T,1024]  audio_feats [B,T,1024]
  AVH_Causal_Ablation  (A1 / A2 / A3 objectives)
       ↓
  Z_c (causal) / Z_s (spurious)

Configuration
-------------
All hyperparameters are in configs/A*_e2e.yaml, which extend the frozen
baseline configs with three extra keys under model_hparams:
  avhubert_ckpt        : path to self_large_vox_433h.pt
  avhubert_root        : path to av_hubert/avhubert/
  n_unfreeze_layers    : how many top AV-HuBERT layers to unfreeze (default 4)

Data paths
----------
  av1m_raw_root        : avd1m_preprocessed/{split}/ (has *_roi.mp4 + *.wav)
  favc_raw_root        : favc_preprocessed/
  av1m_csv_root        : csv_metadata/av1m/

Usage
-----
  python train_e2e.py --config_path configs/A2_e2e.yaml
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import lightning as L

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_causal_ablation import AVH_Causal_Ablation
from datasets_e2e import (
    AV1M_E2E_Dataset, FakeAVCeleb_E2E_Dataset, e2e_collate_fn
)


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Domain wrapper (mirrors train_test_causal_ablation.py) ────────────────────

class DomainLabeledDataset(Dataset):
    def __init__(self, dataset, domain_label, cls_label_override=None):
        self.dataset            = dataset
        self.domain_label       = domain_label
        self.cls_label_override = cls_label_override

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.dataset[idx]
        if self.cls_label_override is not None:
            cls_label = self.cls_label_override
        return video, audio, cls_label, self.domain_label, path


# ── E2E Lightning Module ───────────────────────────────────────────────────────

class E2EModel(L.LightningModule):
    """
    Wraps AVHubertWrapper + AVH_Causal_Ablation into a single Lightning module.

    The AVHubertWrapper handles the feature extraction (with optional partial
    fine-tuning) and AVH_Causal_Ablation handles the disentanglement losses
    exactly as in the frozen-encoder experiments.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        # ── Build AV-HuBERT encoder ──────────────────────────────────────────
        avhubert_ckpt   = hp["avhubert_ckpt"]
        avhubert_root   = hp.get("avhubert_root",
                                  os.path.join(os.path.dirname(__file__),
                                               "..", "av_hubert", "avhubert"))
        n_unfreeze      = int(hp.get("n_unfreeze_layers", 4))

        self.encoder = AVHubertWrapper(
            ckpt_path        = avhubert_ckpt,
            avhubert_root    = avhubert_root,
            n_unfreeze_layers= n_unfreeze,
        )

        # ── Build causal disentanglement head ────────────────────────────────
        # Re-use AVH_Causal_Ablation directly; it expects pre-extracted features
        # as input so it is fully compatible — we just feed it encoder outputs.
        self.head = AVH_Causal_Ablation(config=config)

        self.use_domain       = bool(config.get("ablation", {}).get("use_domain", False))
        self.use_manip_domain = bool(config.get("ablation", {}).get("use_manip_domain", False))
        # has_domain_labels: batch is a 5-tuple whenever either domain flag is set
        self.has_domain_labels = self.use_domain or self.use_manip_domain
        self.val_outputs = []

    # ── Optimisers ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder = float(hp.get("lr_encoder", 1e-5))   # small LR for encoder
        lr_head    = float(hp.get("lr",         1e-3))   # larger LR for heads
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params,    "lr": lr_head},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler,
                              "monitor":   "val_auc_causal",
                              "interval":  "epoch"},
        }

    # ── Shared step: run encoder then head ────────────────────────────────────

    def _encode_batch(self, batch):
        """Run raw video/audio through AV-HuBERT encoder."""
        if self.has_domain_labels:
            video, audio, cls_labels, domain_labels, _ = batch
        else:
            video, audio, cls_labels, _ = batch
            domain_labels = None

        # video: [B, T, 1, H, W] → encoder expects [B, T, 1, H, W]
        # audio: [B, T, 104]
        video_feats, audio_feats = self.encoder(video, audio)
        # video_feats, audio_feats: [B, T, 1024]

        return video_feats, audio_feats, cls_labels, domain_labels

    # ── Training step: delegate to head's training_step ───────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        video_feats, audio_feats, cls_labels, domain_labels = self._encode_batch(batch)

        if self.has_domain_labels:
            fake_batch = (video_feats, audio_feats, cls_labels, domain_labels,
                          [""] * video_feats.shape[0])
        else:
            fake_batch = (video_feats, audio_feats, cls_labels,
                          [""] * video_feats.shape[0])

        # AVH_Causal_Ablation.training_step calls self.log() which requires
        # being the trainer-managed LightningModule.  Wire its _trainer and
        # _current_fx_name so that logging is routed through the Trainer.
        self.head._trainer = self._trainer
        self.head._current_fx_name = "training_step"
        try:
            return self.head.training_step(fake_batch, batch_idx)
        finally:
            self.head._trainer = None
            self.head._current_fx_name = None

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_feats, audio_feats, cls_labels, domain_labels = self._encode_batch(batch)
        scores = self.head(video_feats, audio_feats, mode="causal")
        self.val_outputs.append((scores.detach().cpu(), cls_labels.cpu()))

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        scores = torch.cat([o[0] for o in self.val_outputs]).numpy()
        labels = torch.cat([o[1] for o in self.val_outputs]).numpy()
        self.val_outputs.clear()
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.5
        self.log("val_auc_causal", auc, prog_bar=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(config: dict, use_domain: bool, use_manip_domain: bool = False):
    dp = config["data_paths"]
    av1m_raw   = dp["av1m_raw_root"]
    favc_raw   = dp["favc_raw_root"]
    av1m_csv   = dp["av1m_csv_root"]
    max_frames = int(config.get("model_hparams", {}).get("max_frames", 150))

    if use_manip_domain:
        # ---------------------
        # Manip-type domain path: load val_manip.csv which has manip_type column.
        # Dataset returns 5-tuples directly (no DomainLabeledDataset wrapper needed).
        # ---------------------
        csv_file = os.path.join(av1m_csv, "val_manip.csv")
        full_ds = AV1M_E2E_Dataset(av1m_raw, csv_file, split="val",
                                    max_frames=max_frames, return_manip_type=True)
        n_total = len(full_ds)
        n_train = int(n_total * 0.8)
        n_val   = n_total - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)))
        print(f"[load_data] manip_domain: val 80/20 split → "
              f"train={n_train}, holdout={n_val}", flush=True)
    else:
        # Only avd1m_preprocessed/val/ exists on scratch (train was never preprocessed).
        # For this proof-of-concept e2e experiment we split the 2000 val clips 80/20.
        full_ds = AV1M_E2E_Dataset(av1m_raw, os.path.join(av1m_csv, "val_labels.csv"),
                                    split="val", max_frames=max_frames)
        n_total = len(full_ds)
        n_train = int(n_total * 0.8)
        n_val   = n_total - n_train
        train_base, val_base = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.get("seed", 42)))
        print(f"[load_data] e2e POC: val data 80/20 split → "
              f"train={n_train}, holdout={n_val}", flush=True)
        train_ds = train_base
        val_ds   = val_base

        if use_domain:
            favc_ds   = FakeAVCeleb_E2E_Dataset(favc_raw, max_frames=max_frames)
            train_ds  = ConcatDataset([
                DomainLabeledDataset(train_ds, domain_label=0),
                DomainLabeledDataset(favc_ds,  domain_label=1, cls_label_override=-1),
            ])
            val_ds = DomainLabeledDataset(val_ds, domain_label=0)

    batch_size  = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 4))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=e2e_collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=e2e_collate_fn,
                              pin_memory=True)
    return train_loader, val_loader


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    use_domain       = bool(config.get("ablation", {}).get("use_domain", False))
    use_manip_domain = bool(config.get("ablation", {}).get("use_manip_domain", False))
    train_loader, val_loader = load_data(config, use_domain, use_manip_domain)

    model = E2EModel(config=config)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cb_cfg     = config.get("callbacks", {})
    ckpt_cfg   = cb_cfg.get("ckpt_args", {})
    ckpt_dir   = ckpt_cfg.get("ckpt_dir", "outputs_e2e/ckpts")
    log_path   = cb_cfg.get("logger", {}).get("log_path", "outputs_e2e/logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath    = ckpt_dir,
        filename   = "model-{epoch:02d}-{val_auc_causal:.4f}",
        monitor    = "val_auc_causal",
        mode       = "max",
        save_top_k = 3,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val_auc_causal",
        mode     = "max",
        patience = config.get("early_stopping", {}).get("patience", 15),
    )
    logger = CSVLogger(save_dir=log_path, name="")

    trainer = L.Trainer(
        max_epochs         = config.get("epochs", 50),
        callbacks          = [checkpoint_cb, early_stop_cb],
        logger             = logger,
        precision          = "16-mixed",   # AMP for memory efficiency
        gradient_clip_val  = 1.0,
        log_every_n_steps  = 20,
        val_check_interval = 1.0,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
