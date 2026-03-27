"""
E2E Training — A6 + Domain Adversarial Push-Pull (A6_e2e)
==========================================================

Fully end-to-end training: AV-HuBERT fully unfrozen + A6 FCD head
with push-pull domain adversarial disentanglement.

Architecture:
  AV-HuBERT (all layers unfrozen, lr_encoder=1e-6)
       ↓  video_feats [B,T,1024],  audio_feats [B,T,1024]
  AVH_FCD_A6 (FCD decomposition):
    Z_c = [s_v, u_v, s_a, u_a]  →  causal_head  (fake/real)
    Z_s = [r_v, r_a]
  Domain adversarial (push-pull):
    GRL(Z_c) → domain_head_c  (expel domain from Z_c - adversarial)
    Z_s      → domain_head_s  (concentrate domain in Z_s - discriminative)

Training data:
  AV1M:  ALL available preprocessed clips via comprehensive CSV (~63k)
  FAVC real (domain=1): val split real-only clips, no task labels

Usage:
  python train_e2e_a6.py --config_path configs/A6_e2e.yaml
"""

import argparse
import math
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
from mlp_fcd_a6 import AVH_FCD_A6, grad_reverse, _sinkhorn_divergence
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
    Adds a domain label to a base dataset returning (v, a, cls, path).
    When cls_label_override is set, overrides the task label (used for FAVC
    clips that have no AV1M task label: set to -1 to skip task loss).
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

class E2EModelA6(L.LightningModule):
    """
    End-to-end model: AVHubertWrapper (fully unfrozen) + AVH_FCD_A6.

    Inlines A6's training_step logic directly (rather than via trainer
    patching) for cleaner gradient flow and explicit loss control.

    Training batch: (video [B,T,1,H,W], audio [B,T,104],
                     cls_labels [B] (-1 for FAVC), domain_labels [B], paths)
    Validation batch: AV1M only (domain_labels all 0)
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

        # ── A6 FCD head ───────────────────────────────────────────────────────
        self.head = AVH_FCD_A6(config=config)

        # Cache hyper-params needed in training_step
        self.lambda_mi    = float(hp.get("lambda_mi",    1.0))
        self.lambda_sda   = float(hp.get("lambda_sda",   0.5))
        self.lambda_dis   = float(hp.get("lambda_dis",   1.0))
        self.lambda_orth  = float(hp.get("lambda_orth",  0.1))
        self.lambda_dadv  = float(hp.get("lambda_dadv",  1.0))
        self.lambda_ddis  = float(hp.get("lambda_ddis",  1.0))
        self.grl_alpha    = float(hp.get("grl_alpha",    1.0))
        self.sda_eps      = float(hp.get("sda_eps",      0.05))
        self.sda_n_iter   = int  (hp.get("sda_n_iter",   5))

        self._val_scores: list = []
        self._val_labels: list = []

    # ── Optimizer (two LR groups) ─────────────────────────────────────────────

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

    # ── Training step (A6 losses inlined) ────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        # Progressive DANN alpha: ramp grl_alpha from 0 → max over training
        hp = self.config.get("model_hparams", {})
        if hp.get("dann_progressive", False):
            gamma     = float(hp.get("dann_gamma", 10.0))
            alpha_max = float(hp.get("grl_alpha", 1.0))
            p         = self.global_step / max(self.trainer.estimated_stepping_batches, 1)
            self.grl_alpha = alpha_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

        video_raw, audio_raw, cls_labels, domain_labels, _ = batch

        # Step 1: run AV-HuBERT encoder on raw video/audio
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)

        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        # Step 2: FCD decomposition via A6 head
        causal_repr, spurious_repr, comp = self.head._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]

        Z_c_pool = causal_repr.mean(1)    # [B, causal_dim]
        Z_s_pool = spurious_repr.mean(1)  # [B, spurious_dim]

        # ── Domain losses (all clips: AV1M + FAVC) ───────────────────────────
        # Adversarial: GRL on Z_c → encoder cannot encode domain into causal
        Z_c_rev   = grad_reverse(Z_c_pool, self.grl_alpha)
        d_score_c = self.head.domain_head_c(Z_c_rev).squeeze(-1)
        loss_dadv = F.binary_cross_entropy_with_logits(d_score_c, domain_labels)

        # Discriminative: Z_s should concentrate domain info
        d_score_s = self.head.domain_head_s(Z_s_pool).squeeze(-1)
        loss_ddis = F.binary_cross_entropy_with_logits(d_score_s, domain_labels)

        # ── Task losses (AV1M only, cls_label >= 0) ──────────────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            c_repr_av = causal_repr[av1m_mask]
            s_repr_av = spurious_repr[av1m_mask]
            lbl_av    = cls_labels[av1m_mask]
            s_v_a, u_v_a, r_v_a = s_v[av1m_mask], u_v[av1m_mask], r_v[av1m_mask]
            s_a_a, u_a_a, r_a_a = s_a[av1m_mask], u_a[av1m_mask], r_a[av1m_mask]

            score_task = self.head._cls_score(self.head.causal_head,  c_repr_av)
            loss_task  = self.head._ce_loss(score_task, lbl_av)

            score_uv  = self.head._cls_score(self.head.unique_head_v, u_v_a)
            score_ua  = self.head._cls_score(self.head.unique_head_a, u_a_a)
            loss_mi   = (self.head._ce_loss(score_uv, lbl_av)
                       + self.head._ce_loss(score_ua, lbl_av))

            s_v_proj  = self.head.sda(s_v_a.mean(1))
            s_a_proj  = self.head.sda(s_a_a.mean(1))
            loss_sda  = _sinkhorn_divergence(
                s_v_proj, s_a_proj,
                eps=self.sda_eps, n_iter=self.sda_n_iter,
            )

            r_v_pool  = r_v_a.mean(1)
            r_a_pool  = r_a_a.mean(1)
            r_stack   = torch.cat([r_v_pool, r_a_pool], dim=0)
            mod_score = self.head.modality_head(r_stack).squeeze(-1)
            mod_lbl   = torch.cat([
                torch.zeros(r_v_pool.shape[0], device=r_v_pool.device),
                torch.ones( r_a_pool.shape[0], device=r_a_pool.device),
            ])
            loss_dis  = F.binary_cross_entropy_with_logits(mod_score, mod_lbl)

            loss_orth = 0.5 * (
                self.head._orth_loss(u_v_a, r_v_a)
              + self.head._orth_loss(u_a_a, r_a_a)
            )

            # Spurious probe (diagnostic only — never propagates to encoder)
            score_spu      = self.head._cls_score(
                self.head.redundant_head, s_repr_av.detach()
            )
            loss_spu_probe = self.head._ce_loss(score_spu, lbl_av)
        else:
            zero = torch.tensor(0.0, device=video_feats.device, requires_grad=True)
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth = loss_spu_probe = zero

        # SVD regularisation every 50 steps
        if self.global_step % 50 == 0:
            self.head.urd_v.apply_svd_regularization()
            self.head.urd_a.apply_svd_regularization()

        loss = (
            loss_task
            + self.lambda_mi   * loss_mi
            + self.lambda_sda  * loss_sda
            + self.lambda_dis  * loss_dis
            + self.lambda_orth * loss_orth
            + self.lambda_dadv * loss_dadv
            + self.lambda_ddis * loss_ddis
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_mi",        loss_mi,        **log_kw)
        self.log("train_loss_sda",       loss_sda,       **log_kw)
        self.log("train_loss_dis",       loss_dis,       **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_dadv",      loss_dadv,      **log_kw)
        self.log("train_loss_ddis",      loss_ddis,      **log_kw)
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
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
    batch_size  = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 4))

    # ── AV1M train (domain=0, task-labelled) ─────────────────────────────────
    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    print(f"[load_data] AV1M train: {len(av1m_base)} clips", flush=True)

    # ── FAVC real (domain=1, no task labels — cls_label=-1) ──────────────────
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
            print(f"[load_data] FAVC oversampled {favc_oversample}× → {len(favc_dom)} clips", flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — "
              "training without FAVC domain signal.", flush=True)
        train_ds = av1m_train

    # ── AV1M val (domain=0, only AV1M labelled clips) ────────────────────────
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 43))
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader = load_data(config)
    model = E2EModelA6(config=config)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger", {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A6_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A6_e2e/ckpts"),
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
    accum  = int(config.get("accumulate_grad_batches", 8))
    clip   = float(config.get("gradient_clip_val", 1.0))
    prec   = config.get("precision", "16-mixed")

    trainer = L.Trainer(
        max_epochs               = int(config.get("epochs", 30)),
        accelerator              = "gpu",
        devices                  = 1,
        precision                = prec,
        accumulate_grad_batches  = accum,
        gradient_clip_val        = clip,
        callbacks                = [ckpt_cb, es_cb],
        logger                   = logger,
        log_every_n_steps        = 10,
        enable_progress_bar      = True,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
