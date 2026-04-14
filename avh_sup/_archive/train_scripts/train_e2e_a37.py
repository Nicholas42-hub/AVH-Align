"""
E2E Training — A37: Label-Aware Domain Adversarial on z_sync (No Subspace Split)
==================================================================================

**Motivation (lessons from A35/A36)**

  A35 attacked the FULL z_sync distribution (SWD + variance) but without
  conditioning on labels — meaning it forced real-AV1M ≈ real-FAVC AND
  fake-AV1M ≈ fake-FAVC, which is correct, BUT also forced real-AV1M ≈ fake-FAVC
  along the "domain collapses into class" manifold, destroying label signal.

  A36 addressed this with a conditional GRL + subspace split:
    - z_sync → z_c_inv (task) + z_c_aux (domain attractor)
    - Conditional GRL on z_c_inv only
  The subspace split gave the model a "legal slot" for domain info.

**A37 strategy: label-aware adversarial on full z_sync (no subspace split)**

  Simpler than A36. Apply a label-aware (class-conditional) domain adversarial
  loss DIRECTLY on z_sync_pool (512d) — no subspace projection, no domain attractor.

  The label-aware adversarial conditions the domain discriminator on the
  model's current predicted class (y_hat_prob), asking:
    "Given the predicted class, can you still tell the domain?"

  Input to cond_domain_head: concat(GRL(z_sync_pool), y_hat_prob.detach())
  GRL reverses the gradient from z_sync_pool, pushing sync_enc to remove
  within-class domain signal WITHOUT erasing between-class signal.

  If the simpler approach (no legal slot) works: domain AUC drops while
  task AUC stays high. If it doesn't work (task collapses), it confirms
  that the subspace split in A36 is load-bearing.

**Architecture (changes from A35):**
  Same AVH_SAD_A9 base (z_sync 512d, z_app 256d×2).
  NEW: cond_domain_head: MLP(512 + 1 → 128 → 64 → 1)
  REMOVED: lambda_swd, lambda_var_align (replaced by conditional GRL)

**Losses:**
  L = L_task
    + λ_cond_grl(ep)  * L_cond_grl    [label-aware GRL on z_sync_pool]
    + λ_ddis          * L_ddis         [domain BCE on z_app_pool — routing]
    + λ_orth          * L_orth         [z_sync ⊥ z_app]

**Curriculum λ_cond_grl:**
  ep  0-2 : λ = 0.0   (task warm-up with finetune checkpoint)
  ep  3-8 : λ = 0.0 → 0.5  (linear ramp)
  ep  9+  : λ = 0.5   (held at max)

**Success criteria:**
  domain_auc_zsync_raw < 0.65   (true domain invariance in z_sync)
  domain_auc_zsync_centered < 0.65 (covariance residual also gone)
  val_auc_causal > 0.95         (task not killed)

  python train_e2e_a37.py --config_path configs/A37_e2e.yaml
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Sampler, Subset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_sad_a9 import AVH_SAD_A9
from mlp_sad_a8 import grad_reverse, _make_adv_head


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dataset utilities (identical to A35/A36) ─────────────────────────────────

class BalancedDomainSampler(Sampler):
    """Yields interleaved batches of domain-0 and domain-1 samples."""
    def __init__(self, domain_labels, batch_size: int, drop_last: bool = True):
        self.domain_labels = np.array(domain_labels, dtype=np.int32)
        self.batch_size    = batch_size
        self.drop_last     = drop_last
        self.idx0 = np.where(self.domain_labels == 0)[0]
        self.idx1 = np.where(self.domain_labels == 1)[0]
        self._n_per_domain = batch_size // 2

    def __iter__(self):
        rng  = np.random.default_rng()
        idx0 = rng.permutation(self.idx0)
        idx1 = rng.permutation(self.idx1)
        n    = max(len(idx0), len(idx1))
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
        return (n // self._n_per_domain) * self.batch_size


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


class AugmentedDataset(Dataset):
    def __init__(self, dataset,
                 aug_video_noise: float = 0.3,
                 aug_brightness:  float = 0.5,
                 aug_flip_p:      float = 0.5,
                 aug_audio_noise: float = 0.05,
                 p: float = 0.9):
        self.dataset         = dataset
        self.aug_video_noise = aug_video_noise
        self.aug_brightness  = aug_brightness
        self.aug_flip_p      = aug_flip_p
        self.aug_audio_noise = aug_audio_noise
        self.p               = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video, audio = item[0], item[1]
        rest = item[2:]
        if random.random() > self.p:
            return item
        sigma_v = random.uniform(0.0, self.aug_video_noise)
        if sigma_v > 0:
            video = video + torch.randn_like(video) * sigma_v
        if self.aug_brightness > 0:
            delta = random.uniform(-self.aug_brightness, self.aug_brightness)
            video = video + delta
        if random.random() < self.aug_flip_p:
            video = torch.flip(video, dims=[-1])
        sigma_a = random.uniform(0.0, self.aug_audio_noise)
        if sigma_a > 0:
            audio = audio + torch.randn_like(audio) * sigma_a
        return (video, audio) + rest


# ── E2E Lightning Module ───────────────────────────────────────────────────────

class E2EModelA37(L.LightningModule):
    """
    A37: A9_e2e + label-aware domain adversarial on z_sync (no subspace split).

    Key difference from A35: replaces SWD+variance alignment with conditional GRL.
    Key difference from A36: NO subspace split (z_sync_pool used directly).

    Label-aware domain adversarial:
      cond_domain_head input = concat(GRL(z_sync_pool), y_hat_prob.detach())
      y_hat_prob = sigmoid(cls_score(causal_head, z_sync)).detach()
      For FAVC clips (cls=-1): y_hat_prob from the model prediction (no true label)
      BCE(cond_domain_head(cond_input), domain_labels) — asks "given predicted class,
      can you still tell the domain?" → GRL pushes z_sync toward domain-invariance
      WITHIN each predicted class.

    Architecture:
      Encoder: AVHubertWrapper (fine-tuned from A9_e2e checkpoint)
      Head:    AVH_SAD_A9 (unchanged — z_sync 512d, z_app 512d)
      NEW:     cond_domain_head: Linear(513 → 128) → LN → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 1)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        self.encoder = AVHubertWrapper(
            ckpt_path         = hp["avhubert_ckpt"],
            avhubert_root     = hp.get("avhubert_root", ""),
            n_unfreeze_layers = int(hp.get("n_unfreeze_layers", -1)),
        )
        self.head = AVH_SAD_A9(config=config)

        # ── Conditional GRL domain head (NEW in A37) ──────────────────────────
        # Input dim: sync_dim + 1 (for y_hat_prob scalar)
        sync_dim = int(hp.get("sync_dim", 512))
        self.cond_domain_head = nn.Sequential(
            nn.Linear(sync_dim + 1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_cond_grl_max = float(hp.get("lambda_cond_grl",  0.5))
        self.grl_warmup_start    = int(hp.get("grl_warmup_start",     3))
        self.grl_ramp_epochs     = int(hp.get("grl_ramp_epochs",      6))
        self.lambda_ddis         = float(hp.get("lambda_ddis",       1.0))
        self.lambda_orth         = float(hp.get("lambda_orth",       0.1))
        self.grl_alpha           = float(hp.get("grl_alpha",         1.0))

        self._val_scores: list = []
        self._val_labels: list = []
        self.domain_probe_loader = None

    # ── Curriculum λ for conditional GRL ─────────────────────────────────────

    def _grl_lambda(self) -> float:
        """
        ep < warmup_start                  → 0.0
        warmup_start ≤ ep < warmup+ramp    → linear ramp
        ep ≥ warmup_start + ramp_epochs    → lambda_cond_grl_max
        """
        ep = self.current_epoch
        if ep < self.grl_warmup_start:
            return 0.0
        progress = min(1.0, (ep - self.grl_warmup_start) / max(1, self.grl_ramp_epochs))
        return self.lambda_cond_grl_max * progress

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp           = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder", 5e-6))
        lr_head      = float(hp.get("lr",         1e-4))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = (
            list(self.head.parameters())
            + list(self.cond_domain_head.parameters())
        )

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

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        video_raw, audio_raw, cls_labels, domain_labels, _ = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)

        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        z_sync, z_app, _ = self.head._encode(video_feats, audio_feats)
        z_sync_pool = z_sync.mean(1)   # [B, sync_dim]
        z_app_pool  = z_app.mean(1)    # [B, spurious_dim]

        n0 = (domain_labels < 0.5).sum().item()
        n1 = (domain_labels >= 0.5).sum().item()

        # ── Task score for ALL clips (needed for y_hat conditioning) ──────────
        score_all  = self.head._cls_score(self.head.causal_head, z_sync)  # [B]
        y_hat_prob = torch.sigmoid(score_all).detach()                     # stop-grad

        # ── Task loss (AV1M clips only, cls_labels >= 0) ──────────────────────
        av1m_mask = (cls_labels >= 0)
        if av1m_mask.any():
            loss_task = AVH_SAD_A9._ce_loss(score_all[av1m_mask], cls_labels[av1m_mask])
        else:
            loss_task = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Label-aware domain adversarial on z_sync_pool ────────────────────
        # Conditional GRL: asks "given predicted class, can you still tell domain?"
        # GRL pushes sync_enc to remove within-class domain signal from z_sync.
        # y_hat_prob is stop-graded: label signal does NOT flow back to the encoder
        # through this path, preventing domain classification from "using" label info.
        lambda_grl = self._grl_lambda()
        if lambda_grl > 0:
            z_sync_pool_grl = grad_reverse(z_sync_pool, self.grl_alpha)
            cond_input = torch.cat(
                [z_sync_pool_grl, y_hat_prob.unsqueeze(-1)], dim=-1
            )  # [B, sync_dim+1]
            cond_logit    = self.cond_domain_head(cond_input).squeeze(-1)   # [B]
            loss_cond_grl = F.binary_cross_entropy_with_logits(cond_logit, domain_labels)
        else:
            loss_cond_grl = torch.zeros(1, device=video_feats.device).squeeze()

        # ── z_app domain discriminative loss (routes domain to spurious stream)
        d_score_app = self.head.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Orthogonality: z_sync ⊥ z_app (AV1M clips only) ──────────────────
        if av1m_mask.any():
            loss_orth = self.head._orth_loss(
                z_sync[av1m_mask], z_app[av1m_mask]
            )
        else:
            loss_orth = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Spurious probe (monitoring only — no gradient to encoder) ─────────
        if av1m_mask.any():
            score_spu = AVH_SAD_A9._cls_score(
                self.head.spurious_head, z_app[av1m_mask].detach()
            )
            loss_spu_probe = AVH_SAD_A9._ce_loss(score_spu, cls_labels[av1m_mask])
        else:
            loss_spu_probe = torch.zeros(1, device=video_feats.device).squeeze()

        loss = (
              loss_task
            + lambda_grl            * loss_cond_grl
            + self.lambda_ddis      * loss_ddis
            + self.lambda_orth      * loss_orth
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",              loss,              **log_kw)
        self.log("train_loss_task",         loss_task,         **log_kw)
        self.log("train_loss_cond_grl",     loss_cond_grl,     **log_kw)
        self.log("train_loss_ddis",         loss_ddis,         **log_kw)
        self.log("train_loss_orth",         loss_orth,         **log_kw)
        self.log("train_loss_spu_probe",    loss_spu_probe,    **log_kw)
        self.log("train_grl_lambda",        float(lambda_grl), **log_kw)
        self.log("train_n_domain0",         float(n0),         **log_kw)
        self.log("train_n_domain1",         float(n1),         **log_kw)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _domain, _paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        scores = self.head(video_feats, audio_feats, mode="causal")
        self._val_scores.append(scores.detach().cpu())
        self._val_labels.append(cls_labels.cpu())

    # ── Domain probe ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_domain_probe(self):
        """
        Returns (auc_zsync_raw, auc_zsync_centered).

        PRIMARY:   raw z_sync_pool logistic regression (TRUE domain invariance).
        SECONDARY: centered (after per-domain mean subtraction → covariance residual).

        Want: RAW < 0.65 AND CENTERED < 0.65.
        """
        self.eval()
        zs_list, dom_list = [], []
        device = next(self.parameters()).device

        for batch in self.domain_probe_loader:
            if batch is None:
                continue
            video_raw, audio_raw, _, domain_labels, _ = batch
            video_raw = video_raw.to(device)
            audio_raw = audio_raw.to(device)
            video_feats, audio_feats = self.encoder(video_raw, audio_raw)
            z_sync, _, _ = self.head._encode(video_feats, audio_feats)
            zs_list.append(z_sync.mean(1).cpu().numpy())
            dom_list.extend(domain_labels.tolist())

        zs_all  = np.concatenate(zs_list)
        dom_all = np.array(dom_list)

        def _probe(X, y, n_folds=5):
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
            )
            skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
            try:
                return roc_auc_score(y, probs)
            except Exception:
                return 0.5

        auc_zsync_raw = _probe(zs_all, dom_all)

        # Centered: remove per-domain mean → probes covariance residual
        zs_centered = zs_all.copy()
        for d in [0, 1]:
            idx_d = (dom_all == d)
            if idx_d.sum() > 1:
                zs_centered[idx_d] -= zs_centered[idx_d].mean(0)
        auc_zsync_centered = _probe(zs_centered, dom_all)

        self.train()
        return auc_zsync_raw, auc_zsync_centered

    # ── Validation epoch end ──────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        scores_local = torch.cat(self._val_scores)
        labels_local = torch.cat(self._val_labels)
        self._val_scores.clear()
        self._val_labels.clear()

        if self.trainer.world_size > 1:
            scores_all = self.all_gather(scores_local)
            labels_all = self.all_gather(labels_local)
            scores = scores_all.reshape(-1).cpu().numpy()
            labels = labels_all.reshape(-1).cpu().numpy()
        else:
            scores = scores_local.cpu().numpy()
            labels = labels_local.cpu().numpy()

        mask = labels >= 0
        if mask.sum() < 2:
            return
        try:
            auc = roc_auc_score(labels[mask], scores[mask])
        except Exception:
            auc = 0.5
        self.log("val_auc_causal", auc, prog_bar=True, sync_dist=False)

        if self.domain_probe_loader is not None and self.trainer.is_global_zero:
            zsync_raw, zsync_centered = self._run_domain_probe()
            ep  = self.current_epoch
            lam = self._grl_lambda()
            print(
                f"\n[Epoch {ep:02d}|λ_grl={lam:.3f}] A37 probe report:"
                f"\n  z_sync domain RAW      = {zsync_raw:.4f}  ← PRIMARY (want <0.65)"
                f"\n  z_sync domain CENTERED = {zsync_centered:.4f}  (covariance residual)"
                f"\n  task AUC               = {auc:.4f}",
                flush=True,
            )
            self.log("domain_auc_zsync_raw",      zsync_raw,      on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("domain_auc_zsync_centered", zsync_centered, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("train_grl_lambda_ep",        lam,            on_epoch=True, prog_bar=False, sync_dist=False)


# ── Data loading (identical to A35/A36) ──────────────────────────────────────

def load_data(config: dict):
    dp          = config["data_paths"]
    train_csv   = dp["e2e_train_csv"]
    val_csv     = dp["e2e_val_csv"]
    favc_root   = dp.get("favc_raw_root", "")
    hp          = config.get("model_hparams", {})
    max_frames  = int(hp.get("max_frames", 150))
    batch_size  = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 6))

    aug_cfg         = config.get("augmentation", {})
    do_augment      = bool(aug_cfg.get("enabled", True))
    aug_video_noise = float(aug_cfg.get("video_noise",  0.3))
    aug_brightness  = float(aug_cfg.get("brightness",   0.5))
    aug_flip_p      = float(aug_cfg.get("flip_p",       0.5))
    aug_audio_noise = float(aug_cfg.get("audio_noise",  0.05))
    aug_p           = float(aug_cfg.get("p",            0.9))

    from datasets_e2e import (
        AV1M_E2E_FullPathDataset,
        FakeAVCeleb_E2E_Dataset,
        e2e_collate_fn,
    )

    # AV1M train (domain=0)
    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    if do_augment:
        av1m_train = AugmentedDataset(
            av1m_train,
            aug_video_noise=aug_video_noise,
            aug_brightness=aug_brightness,
            aug_flip_p=aug_flip_p,
            aug_audio_noise=aug_audio_noise,
            p=aug_p,
        )
    print(f"[load_data] AV1M train: {len(av1m_base)} clips", flush=True)

    # FAVC ALL clips (domain=1, real+fake; cls_label_override=-1 → no task loss)
    favc_oversample = int(config.get("favc_oversample", 3))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
        favc_dom = DomainLabeledDataset(favc_ds, domain_label=1, cls_label_override=-1)
        print(f"[load_data] FAVC all clips (real+fake): {len(favc_ds)}", flush=True)
        if do_augment:
            favc_dom = AugmentedDataset(
                favc_dom,
                aug_video_noise=aug_video_noise,
                aug_brightness=aug_brightness,
                aug_flip_p=aug_flip_p,
                aug_audio_noise=aug_audio_noise,
                p=aug_p,
            )
        if favc_oversample > 1:
            favc_dom = ConcatDataset([favc_dom] * favc_oversample)
            print(f"[load_data] FAVC oversampled {favc_oversample}× → {len(favc_dom)} clips", flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — training without FAVC.", flush=True)
        train_ds = av1m_train

    # AV1M val (domain=0, for validation AUC)
    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    def _collect_domain_labels(ds):
        if isinstance(ds, ConcatDataset):
            labels = []
            for sub in ds.datasets:
                labels.extend(_collect_domain_labels(sub))
            return labels
        elif isinstance(ds, AugmentedDataset):
            return _collect_domain_labels(ds.dataset)
        elif isinstance(ds, DomainLabeledDataset):
            return [ds.domain_label] * len(ds)
        else:
            return [0] * len(ds)

    domain_labels_list = _collect_domain_labels(train_ds)
    n0 = sum(1 for l in domain_labels_list if l == 0)
    n1 = sum(1 for l in domain_labels_list if l == 1)
    print(f"[load_data] BalancedDomainSampler: {n0} domain-0 (AV1M), {n1} domain-1 (FAVC)", flush=True)

    sampler = BalancedDomainSampler(domain_labels_list, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    # Domain probe loader (200 AV1M val + 200 FAVC, no augmentation)
    n_probe   = 200
    rng_probe = np.random.default_rng(0)
    av1m_probe_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    av1m_probe = DomainLabeledDataset(
        Subset(av1m_probe_base, list(range(min(n_probe, len(av1m_probe_base))))),
        domain_label=0,
    )
    domain_probe_loader = None
    if favc_root and os.path.isdir(favc_root):
        favc_probe_base = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
        favc_probe_idx  = rng_probe.choice(
            len(favc_probe_base), size=min(n_probe, len(favc_probe_base)), replace=False
        ).tolist()
        favc_probe = DomainLabeledDataset(
            Subset(favc_probe_base, favc_probe_idx),
            domain_label=1, cls_label_override=-1,
        )
        probe_ds = ConcatDataset([av1m_probe, favc_probe])
        domain_probe_loader = DataLoader(
            probe_ds, batch_size=8, shuffle=False,
            num_workers=2, collate_fn=e2e_collate_fn,
        )
        print(
            f"[load_data] Domain probe loader: {len(av1m_probe)} AV1M + "
            f"{len(favc_probe)} FAVC (no augmentation)", flush=True
        )

    return train_loader, val_loader, domain_probe_loader


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 43))
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader, domain_probe_loader = load_data(config)
    model = E2EModelA37(config=config)
    model.domain_probe_loader = domain_probe_loader

    # ── Load from A9_e2e checkpoint ───────────────────────────────────────────
    # Keys: "encoder.*", "head.*" — map cleanly to E2EModelA37.
    # New A37 module (cond_domain_head) is missing → strict=False, random init.
    hp = config.get("model_hparams", {})
    finetune_ckpt = hp.get("finetune_ckpt", "")
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        print(f"[A37] Loading weights from {finetune_ckpt} (strict=False)", flush=True)
        ckpt_data  = torch.load(finetune_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_miss = len(missing)
        n_unex = len(unexpected)
        print(f"[A37] Missing keys (new A37 modules, expected): {n_miss}", flush=True)
        if missing and n_miss <= 20:
            for k in missing:
                print(f"       missing: {k}", flush=True)
        print(f"[A37] Unexpected keys (removed modules, expected): {n_unex}", flush=True)
    else:
        print("[A37] WARNING: no finetune_ckpt found — starting from scratch.", flush=True)

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",    {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A37_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A37_e2e/ckpts"),
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

    accum = int(config.get("accumulate_grad_batches", 4))
    clip  = float(config.get("gradient_clip_val", 1.0))
    prec  = config.get("precision", "16-mixed")

    num_gpus  = int(config.get("num_gpus",  1))
    num_nodes = int(config.get("num_nodes", 1))
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if (num_gpus > 1 or num_nodes > 1)
        else "auto"
    )
    trainer = L.Trainer(
        max_epochs              = int(config.get("epochs", 30)),
        accelerator             = "gpu",
        devices                 = num_gpus,
        num_nodes               = num_nodes,
        strategy                = strategy,
        precision               = prec,
        accumulate_grad_batches = accum,
        gradient_clip_val       = clip,
        callbacks               = [ckpt_cb, es_cb],
        logger                  = logger,
        log_every_n_steps       = 1,
        enable_progress_bar     = False,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Training done: {__import__('datetime').datetime.now().strftime('%a %d %b %Y %H:%M:%S')}", flush=True)
    print(f"  Best ckpt in:  {ckpt_cfg.get('ckpt_dir', 'outputs_A37_e2e/ckpts')}/", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
