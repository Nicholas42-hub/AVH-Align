"""
E2E Training — A34: A33 (IPW) + Light GRL on Z_c for domain invariance
=======================================================================

A33 achieved cross-domain AUC=0.803 (epoch 3) but domain probe Z_c=0.983.
A34 adds a LIGHT GRL domain classifier on Z_c (lambda=0.2, warm-up 2 epochs)
to nudge Z_c toward domain invariance WITHOUT destroying task signal.

Total loss:
  L = L_task_ipw                   (IPW-reweighted CE, AV1M only)
    + lambda_ipw  * L_domain_pred  (non-adversarial domain predictor for IPW)
    + lambda_grl  * L_grl_zc       (adversarial GRL on Z_c for invariance)

Key design choices vs A20-A25:
  - lambda_grl = 0.2 (vs 10-15 before) — gentle nudge, not a sledgehammer
  - GRL on mean-pooled Z_c only (not encoder features directly)
  - Separate domain_predictor (detached, for IPW) and domain_clf_grl (GRL)
  - Both have warm-up (ipw_warmup=2, grl_warmup=2 epochs)
  - lr_encoder = 5e-6 (slow, same as A33)
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Sampler
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_causal_ablation import AVH_Causal_Ablation
from mlp_sad_a8 import grad_reverse, _make_adv_head
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


class BalancedDomainSampler(Sampler):
    """
    Yield indices so each batch contains an equal number of AV1M/FAVC clips.

    This keeps the domain signal active on every step instead of relying on
    random shuffling to surface enough target-domain samples in a batch.
    """

    def __init__(self, domain_labels, batch_size: int, drop_last: bool = True):
        self.domain_labels = np.array(domain_labels, dtype=np.int32)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.idx0 = np.where(self.domain_labels == 0)[0]
        self.idx1 = np.where(self.domain_labels == 1)[0]
        self._n_per_domain = batch_size // 2
        if self._n_per_domain == 0:
            raise ValueError("batch_size must be at least 2 for BalancedDomainSampler")

    def __iter__(self):
        rng = np.random.default_rng()
        idx0 = rng.permutation(self.idx0)
        idx1 = rng.permutation(self.idx1)

        n = max(len(idx0), len(idx1))
        idx0 = np.resize(idx0, n)
        idx1 = np.resize(idx1, n)

        n_batches = n // self._n_per_domain
        for i in range(n_batches):
            s = i * self._n_per_domain
            e = s + self._n_per_domain
            batch = np.concatenate([idx0[s:e], idx1[s:e]])
            rng.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        n = max(len(self.idx0), len(self.idx1))
        n_batches = n // self._n_per_domain
        return n_batches * self.batch_size


# ── Domain predictor MLP (for IPW — trained on detached Z_c) ─────────────────

def _make_domain_pred(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 64),     nn.ReLU(),
        nn.Linear(64, 1),
    )


# ── E2E Lightning Module ──────────────────────────────────────────────────────

class E2EModelA34(L.LightningModule):
    """
    A33 (IPW backdoor adjustment) + Light GRL on Z_c (lambda=0.2).

    Two domain classifiers:
      1. domain_predictor: trained on DETACHED Z_c → used only for IPW weights
      2. domain_clf_grl:   trained on GRL(Z_c)    → adversarial, pushes Z_c
                           toward domain invariance with small lambda
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = AVHubertWrapper(
            ckpt_path         = hp["avhubert_ckpt"],
            avhubert_root     = hp.get("avhubert_root", ""),
            n_unfreeze_layers = int(hp.get("n_unfreeze_layers", -1)),
        )

        # ── A2 causal head (use_sg=True) ──────────────────────────────────────
        config.setdefault("ablation", {})["use_sg"]     = True
        config.setdefault("ablation", {})["use_domain"] = False
        config.setdefault("ablation", {})["use_adv"]    = False
        self.head = AVH_Causal_Ablation(config=config)

        # ── Z_c dimensions ────────────────────────────────────────────────────
        proj_dim   = int(hp.get("sync_dim", 512))
        causal_dim = proj_dim * 2  # concat(v_c, a_c)

        # ── IPW domain predictor (detached Z_c, non-adversarial) ──────────────
        self.domain_predictor = _make_domain_pred(causal_dim)

        # ── GRL domain classifier (adversarial on Z_c) ────────────────────────
        self.domain_clf_grl = _make_adv_head(causal_dim)

        # ── Hyperparams ───────────────────────────────────────────────────────
        self.lambda_ipw        = float(hp.get("lambda_ipw",        1.0))
        self.lambda_grl        = float(hp.get("lambda_grl",        0.2))
        self.ipw_clip_min      = float(hp.get("ipw_clip_min",      0.1))
        self.ipw_clip_max      = float(hp.get("ipw_clip_max",     10.0))
        self.ipw_warmup_epochs = int(hp.get("ipw_warmup_epochs",   2))
        self.grl_warmup_epochs = int(hp.get("grl_warmup_epochs",   2))

        self._val_scores: list = []
        self._val_labels: list = []

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder     = float(hp.get("lr_encoder",    5e-6))
        lr_head        = float(hp.get("lr",            1e-4))
        lr_domain      = float(hp.get("lr_domain_pred", lr_head))
        weight_decay   = float(hp.get("weight_decay",  1e-4))

        encoder_params     = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params        = list(self.head.parameters())
        domain_pred_params = list(self.domain_predictor.parameters())
        grl_params         = list(self.domain_clf_grl.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params,     "lr": lr_encoder},
            {"params": head_params,        "lr": lr_head},
            {"params": domain_pred_params, "lr": lr_domain},
            {"params": grl_params,         "lr": lr_domain},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc_causal", "interval": "epoch"},
        }

    def _encode_batch(self, batch):
        video_raw, audio_raw, cls_labels, domain_labels, paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        return video_feats, audio_feats, cls_labels, domain_labels

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        video_feats, audio_feats, cls_labels, domain_labels = self._encode_batch(batch)
        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.long()
        B             = video_feats.shape[0]

        # ── Encode → Z_c ──────────────────────────────────────────────────────
        causal_repr, _, _ = self.head._encode(video_feats, audio_feats)
        z_c_pool = causal_repr.mean(1)  # [B, causal_dim]

        # ── 1. IPW domain predictor (detached — no grad to encoder) ───────────
        z_c_detached  = z_c_pool.detach()
        domain_logits = self.domain_predictor(z_c_detached).squeeze(-1)
        loss_domain_pred = F.binary_cross_entropy_with_logits(
            domain_logits, domain_labels.float()
        )

        # Compute IPW weights from detached propensity scores
        p1 = torch.sigmoid(domain_logits.detach())
        p0 = 1.0 - p1
        p_assigned = torch.where(domain_labels == 1, p1, p0)
        ipw_weights = (0.5 / p_assigned.clamp(min=1e-4)).clamp(
            self.ipw_clip_min, self.ipw_clip_max
        )
        if self.current_epoch < self.ipw_warmup_epochs:
            ipw_weights = torch.ones(B, device=video_feats.device)

        # ── 2. GRL domain classifier (adversarial — gradient DOES flow to Z_c) -
        if self.current_epoch >= self.grl_warmup_epochs:
            z_c_grl = grad_reverse(z_c_pool, alpha=1.0)
            grl_logits = self.domain_clf_grl(z_c_grl).squeeze(-1)
            loss_grl_zc = F.binary_cross_entropy_with_logits(
                grl_logits, domain_labels.float()
            )
        else:
            loss_grl_zc = torch.zeros(1, device=video_feats.device).squeeze()

        # ── 3. Task loss with IPW (AV1M only) ─────────────────────────────────
        valid_mask = (cls_labels >= 0)
        if valid_mask.any():
            logits_per_frame = self.head.causal_head(causal_repr[valid_mask])[..., 0]
            score_task = torch.logsumexp(logits_per_frame, dim=-1)
            logits2    = torch.stack([-score_task, score_task], dim=1)
            ce_per     = F.cross_entropy(logits2, cls_labels[valid_mask], reduction="none")
            w          = ipw_weights[valid_mask]
            loss_task  = (w * ce_per).sum() / w.sum()
        else:
            loss_task = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (
            loss_task
            + self.lambda_ipw * loss_domain_pred
            + self.lambda_grl * loss_grl_zc
        )

        self.log("train_loss",             loss.item(),              on_step=True, on_epoch=False)
        self.log("train_loss_task",        loss_task.item(),         on_step=True, on_epoch=False)
        self.log("train_loss_domain_pred", loss_domain_pred.item(),  on_step=True, on_epoch=False)
        self.log("train_loss_grl_zc",      loss_grl_zc.item(),       on_step=True, on_epoch=False)
        self.log("train_ipw_mean",         ipw_weights.mean().item(), on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 500 == 0:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_val = outputs["loss"].item()
            elif outputs is not None:
                loss_val = float(outputs)
            else:
                loss_val = float("nan")
            total = self.trainer.estimated_stepping_batches
            print(
                f"[Step {self.global_step:06d}/{int(total)}] "
                f"loss={loss_val:.4f}  epoch={self.current_epoch}",
                flush=True,
            )

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_feats, audio_feats, cls_labels, _ = self._encode_batch(batch)
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
        print(f"\n[Epoch {self.current_epoch:02d}] val_auc_causal={auc:.4f}", flush=True)


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
    favc_real_only = bool(config.get("favc_real_only", True))
    use_balanced_sampler = bool(config.get("balanced_domain_sampler", False))

    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    print(f"[load_data] AV1M train clips: {len(av1m_base)}", flush=True)

    favc_oversample = int(config.get("favc_oversample", 1))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(
            favc_root,
            max_frames=max_frames,
            real_only=favc_real_only,
        )
        favc_dom = DomainLabeledDataset(favc_ds, domain_label=1, cls_label_override=-1)
        favc_desc = "real-only" if favc_real_only else "all clips (real+fake)"
        print(f"[load_data] FAVC {favc_desc}: {len(favc_ds)} clips", flush=True)
        if favc_oversample > 1:
            favc_dom = ConcatDataset([favc_dom] * favc_oversample)
            print(f"[load_data] FAVC oversampled {favc_oversample}x → {len(favc_dom)} clips", flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — no domain signal.", flush=True)
        train_ds = av1m_train

    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv,  max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val clips: {len(av1m_val_base)}", flush=True)

    if use_balanced_sampler:
        def _collect_domain_labels(ds):
            if isinstance(ds, ConcatDataset):
                labels = []
                for sub in ds.datasets:
                    labels.extend(_collect_domain_labels(sub))
                return labels
            if isinstance(ds, DomainLabeledDataset):
                return [ds.domain_label] * len(ds)
            return [0] * len(ds)

        domain_labels = _collect_domain_labels(train_ds)
        n0 = sum(1 for label in domain_labels if label == 0)
        n1 = sum(1 for label in domain_labels if label == 1)
        print(
            f"[load_data] BalancedDomainSampler: {n0} domain-0 (AV1M), {n1} domain-1 (FAVC)",
            flush=True,
        )
        sampler = BalancedDomainSampler(domain_labels, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=e2e_collate_fn,
            pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0),
        )
    else:
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
    model = E2EModelA34(config=config)

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger", {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A34_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A34_e2e/ckpts"),
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

    trainer = L.Trainer(
        max_epochs              = int(config.get("epochs", 30)),
        accelerator             = "gpu",
        devices                 = 1,
        precision               = config.get("precision", "16-mixed"),
        accumulate_grad_batches = int(config.get("accumulate_grad_batches", 4)),
        gradient_clip_val       = float(config.get("gradient_clip_val", 1.0)),
        callbacks               = [ckpt_cb, es_cb],
        logger                  = logger,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}", flush=True)


if __name__ == "__main__":
    main()
