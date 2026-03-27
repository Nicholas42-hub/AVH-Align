"""
E2E Training — A9 + Multi-Kernel MMD Domain Alignment, Full E2E
================================================================

Fully end-to-end: AV-HuBERT fully unfrozen + A9 SAD head with MMD-based
domain alignment on Z_sync.

Key difference from A9 (frozen encoder):
  MMD gradients now propagate through the AV-HuBERT encoder, directly
  forcing the encoder to produce domain-neutral cross-modal features.
  This is the fix for the frozen-encoder bottleneck observed in A1–A9:
    - Frozen A9: Z_sync domain AUC = 0.988  (domain baked into encoder)
    - A9_e2e:    MMD gradients reach encoder  → target AUC ≪ 0.9

Architecture:
  AV-HuBERT (all layers unfrozen, lr_encoder=1e-6)
       ↓  video_feats [B,T,1024],  audio_feats [B,T,1024]
  AVH_SAD_A9 (SAD decomposition):
    Z_sync = SynchronyEncoder(cat(V, A, V⊙A, V−A))  →  causal_head (fake/real)
    Z_app  = AppearanceEncoder(V-only) ║ AppearanceEncoder(A-only)

  Domain alignment:
    MMD²(Z_sync|d=AV1M, Z_sync|d=FAVC)   [gradients flow through encoder]
    Z_app → domain_head_s                  (discriminative: concentrate domain)
    GRL(Z_app) → label_head_s             (adversarial: Z_app ⊥ fake/real)
    Z_sync ⊥ Z_app                        (orthogonality regularisation)

Usage:
  python train_e2e_a9.py --config_path configs/A9_e2e.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_sad_a9 import AVH_SAD_A9, _mmd_rbf
from mlp_sad_a8 import grad_reverse
from datasets_e2e import (
    AV1M_E2E_FullPathDataset,
    FakeAVCeleb_E2E_Dataset,
    e2e_collate_fn,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Domain label wrapper ──────────────────────────────────────────────────────

class DomainLabeledDataset(Dataset):
    """
    Wraps a dataset returning (video, audio, cls_label, path) and adds a
    domain label.  cls_label_override=-1 skips task loss for FAVC clips.
    """

    def __init__(self, dataset, domain_label: int, cls_label_override=None):
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


# ── E2E Lightning Module ──────────────────────────────────────────────────────

class E2EModelA9(L.LightningModule):
    """
    End-to-end model: AVHubertWrapper (fully unfrozen) + AVH_SAD_A9 head.

    Batch format:
      train: (video [B,T,1,H,W], audio [B,T,104], cls_labels [B],
              domain_labels [B], paths)
      val:   same (domain_labels all 0 for AV1M)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        # ── AV-HuBERT encoder (fully unfrozen) ───────────────────────────────
        self.encoder = AVHubertWrapper(
            ckpt_path         = hp["avhubert_ckpt"],
            avhubert_root     = hp.get("avhubert_root", ""),
            n_unfreeze_layers = int(hp.get("n_unfreeze_layers", -1)),
        )

        # ── A9 SAD head ───────────────────────────────────────────────────────
        self.head = AVH_SAD_A9(config=config)

        # ── Loss hyperparameters ──────────────────────────────────────────────
        self.lambda_mmd      = float(hp.get("lambda_mmd",       25.0))
        self.lambda_ddis     = float(hp.get("lambda_ddis",       1.0))
        self.lambda_ladv     = float(hp.get("lambda_ladv",       1.0))
        self.lambda_orth     = float(hp.get("lambda_orth",       0.1))
        self.grl_alpha       = float(hp.get("grl_alpha",         1.0))
        self.mmd_min_samples = int  (hp.get("mmd_min_samples",     2))

        self._val_scores: list = []
        self._val_labels: list = []

    # ── Optimiser (two LR groups) ─────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder",   1e-6))
        lr_head      = float(hp.get("lr",           1e-4))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = list(self.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params,    "lr": lr_head},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_auc_causal",
                "interval":  "epoch",
            },
        }

    # ── Training step (A9 losses inlined) ────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        video_raw, audio_raw, cls_labels, domain_labels, _ = batch

        # Step 1: AV-HuBERT encoder on raw video/audio
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)

        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        # Step 2: SAD decomposition
        z_sync, z_app, _ = self.head._encode(video_feats, audio_feats)

        z_sync_pool = z_sync.mean(1)   # [B, sync_dim]
        z_app_pool  = z_app.mean(1)    # [B, spurious_dim]

        # ── MMD: align Z_sync across domains (grads flow through encoder) ─────
        mask0 = (domain_labels < 0.5)    # AV1M clips
        mask1 = (domain_labels >= 0.5)   # FAVC clips
        n0, n1 = mask0.sum().item(), mask1.sum().item()

        if n0 >= self.mmd_min_samples and n1 >= self.mmd_min_samples:
            loss_mmd = _mmd_rbf(z_sync_pool[mask0], z_sync_pool[mask1])
        else:
            loss_mmd = torch.zeros(1, device=z_sync_pool.device).squeeze()

        # ── Z_app discriminative domain loss ──────────────────────────────────
        d_score_app = self.head.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Task + label-adversarial + orth losses (AV1M only) ───────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            z_sync_av1m     = z_sync[av1m_mask]
            z_app_av1m      = z_app[av1m_mask]
            lbl_av1m        = cls_labels[av1m_mask]
            z_app_pool_av1m = z_app_pool[av1m_mask]

            score_task = self.head._cls_score(self.head.causal_head, z_sync_av1m)
            loss_task  = self.head._ce_loss(score_task, lbl_av1m)

            loss_orth  = self.head._orth_loss(z_sync_av1m, z_app_av1m)

            # Appearance branch must NOT predict fake/real (label adversarial)
            z_app_rev   = grad_reverse(z_app_pool_av1m, self.grl_alpha)
            l_score_app = self.head.label_head_s(z_app_rev).squeeze(-1)
            loss_ladv   = F.binary_cross_entropy_with_logits(
                l_score_app, lbl_av1m.float()
            )

            # Spurious probe (diagnostic only — detached, no gradient)
            score_spu      = self.head._cls_score(
                self.head.spurious_head, z_app_av1m.detach()
            )
            loss_spu_probe = self.head._ce_loss(score_spu, lbl_av1m)
        else:
            zero = torch.zeros(1, device=video_feats.device).squeeze()
            loss_task = loss_orth = loss_ladv = loss_spu_probe = zero

        loss = (
              loss_task
            + self.lambda_mmd  * loss_mmd
            + self.lambda_ddis * loss_ddis
            + self.lambda_ladv * loss_ladv
            + self.lambda_orth * loss_orth
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_mmd",       loss_mmd,       **log_kw)
        self.log("train_loss_ddis",      loss_ddis,      **log_kw)
        self.log("train_loss_ladv",      loss_ladv,      **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
        self.log("train_n_domain0",      float(n0),      **log_kw)
        self.log("train_n_domain1",      float(n1),      **log_kw)
        return loss

    # ── Validation step (AV1M only) ───────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _domain, _paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        scores = self.head(video_feats, audio_feats, mode="causal")
        self._val_scores.append(scores.detach().cpu())
        self._val_labels.append(cls_labels.cpu())

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        scores = torch.cat(self._val_scores).numpy()
        labels = torch.cat(self._val_labels).numpy()
        self._val_scores.clear()
        self._val_labels.clear()
        mask = labels >= 0
        if mask.sum() < 2:
            return
        try:
            auc = roc_auc_score(labels[mask], scores[mask])
        except Exception:
            auc = 0.5
        self.log("val_auc_causal", auc, prog_bar=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(config: dict):
    dp          = config["data_paths"]
    train_csv   = dp["e2e_train_csv"]
    val_csv     = dp["e2e_val_csv"]
    favc_root   = dp.get("favc_raw_root", "")
    hp          = config.get("model_hparams", {})
    max_frames  = int(hp.get("max_frames", 150))
    batch_size  = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 6))

    # ── AV1M train (domain=0, task-labelled) ─────────────────────────────────
    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    print(f"[load_data] AV1M train: {len(av1m_base)} clips", flush=True)

    # ── FAVC real (domain=1, cls_label=-1 → skip task loss) ──────────────────
    favc_oversample = int(config.get("favc_oversample", 1))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(
            favc_root, max_frames=max_frames, real_only=True
        )
        favc_dom = DomainLabeledDataset(
            favc_ds, domain_label=1, cls_label_override=-1
        )
        print(f"[load_data] FAVC real domain clips: {len(favc_ds)}", flush=True)
        if favc_oversample > 1:
            favc_dom = ConcatDataset([favc_dom] * favc_oversample)
            print(f"[load_data] FAVC oversampled {favc_oversample}× → {len(favc_dom)} clips",
                  flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — "
              "training without FAVC domain signal.", flush=True)
        train_ds = av1m_train

    # ── AV1M val (domain=0, AV1M labelled clips) ─────────────────────────────
    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 43))
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader = load_data(config)
    model = E2EModelA9(config=config)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",   {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A9_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A9_e2e/ckpts"),
        filename   = "model-{epoch:02d}-{val_auc_causal:.4f}",
        monitor    = ckpt_cfg.get("metric", "val_auc_causal"),
        mode       = ckpt_cfg.get("mode",   "max"),
        save_top_k = 1,
        save_last  = True,
    )
    es_cfg = config.get("early_stopping", {})
    es_cb  = EarlyStopping(
        monitor  = es_cfg.get("metric",   "val_auc_causal"),
        mode     = es_cfg.get("mode",     "max"),
        patience = int(es_cfg.get("patience", 10)),
        verbose  = True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    accum  = int(config.get("accumulate_grad_batches", 4))
    clip   = float(config.get("gradient_clip_val", 1.0))
    prec   = config.get("precision", "16-mixed")

    trainer = L.Trainer(
        max_epochs              = int(config.get("epochs", 30)),
        accelerator             = "gpu",
        devices                 = 1,
        precision               = prec,
        accumulate_grad_batches = accum,
        gradient_clip_val       = clip,
        callbacks               = [ckpt_cb, es_cb],
        logger                  = logger,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
