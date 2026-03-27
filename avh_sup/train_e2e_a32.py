"""
E2E Training — A32: Null-space gradient projection on z_sync
=============================================================

**Motivation (lessons from A29–A31)**:

  A29 (MSE, λ=500): stalled — gradient ∝ gap, self-diminishing.
  A30 (d², λ=1):    task collapsed ep0 — gradient 25× too large.
  A31 (L1-d, λ=0.05, ladv=0): running — gradient constant, task-safe,
      but task and alignment gradients still can CONFLICT in parameter space.

**A32: Null-space gradient projection**:

  Core idea: instead of adding alignment loss to the total loss, we
  PREVENT the task gradient from reinforcing the domain direction.

  At the start of each epoch, compute the unit domain direction:
    d_unit = normalize(μ_FAVC - μ_AV1M)  from probe batch (200+200, stable)

  Before feeding z_sync to the task head, project out the domain direction:
    z_sync_for_task = z_sync - (z_sync @ d_unit) * d_unit   [null-space of d]

  The task head can ONLY use dimensions ⊥ d_unit.
  Task gradient flows back through z_sync_for_task, so:
  → encoder is only rewarded for encoding task info in directions ⊥ d_unit
  → over time, encoder abandons the d_unit direction entirely
  → mean gap shrinks naturally (encoder has no incentive to maintain it)

  Mean alignment (L1-d, λ=0.02) on raw z_sync_pool: actively nudges the gap
  toward zero to accelerate convergence and reward the transition.

  KEY PROPERTY: task and alignment gradients are ORTHOGONAL BY CONSTRUCTION.
  They cannot conflict. No risk of task collapse (unlike A30).

**Expected dynamics**:
  ep0: task AUC may dip slightly (head must use reduced representation initially)
       but will quickly recover as encoder restructures within null-space
  ep1+: mean gap shrinks steadily, domain AUC drops
  ep5+: target domain AUC <0.65, task AUC >0.95

"""

import argparse
import os
import random
import sys

import numpy as np
import torch
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


# ── Domain label wrapper ──────────────────────────────────────────────────────

class BalancedDomainSampler(Sampler):
    """
    Yields indices such that each batch of `batch_size` contains exactly
    batch_size//2 samples from domain=0 (AV1M) and batch_size//2 from
    domain=1 (FAVC).
    """

    def __init__(self, domain_labels, batch_size: int, drop_last: bool = True):
        self.domain_labels = np.array(domain_labels, dtype=np.int32)
        self.batch_size    = batch_size
        self.drop_last     = drop_last
        self.idx0 = np.where(self.domain_labels == 0)[0]
        self.idx1 = np.where(self.domain_labels == 1)[0]
        self._n_per_domain = batch_size // 2

    def __iter__(self):
        rng = np.random.default_rng()
        idx0 = rng.permutation(self.idx0)
        idx1 = rng.permutation(self.idx1)
        h = self._n_per_domain
        n_batches = min(len(idx0), len(idx1)) // h
        for i in range(n_batches):
            batch = np.concatenate([idx0[i*h:(i+1)*h], idx1[i*h:(i+1)*h]])
            rng.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        return (min(len(self.idx0), len(self.idx1)) // self._n_per_domain) * self.batch_size


class DomainLabeledDataset(Dataset):
    def __init__(self, dataset, domain_label: int, cls_label_override: int = None):
        self.dataset           = dataset
        self.domain_label      = domain_label
        self.cls_label_override = cls_label_override

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video, audio, cls_label, *rest = item
        if self.cls_label_override is not None:
            cls_label = self.cls_label_override
        return video, audio, cls_label, self.domain_label, *rest


class AugmentedDataset(Dataset):
    def __init__(self, dataset, aug_video_noise=0.3, aug_brightness=0.5,
                 aug_flip_p=0.5, aug_audio_noise=0.05, p=0.9):
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
        video, audio, *rest = item
        if random.random() < self.p:
            video = video + torch.randn_like(video) * self.aug_video_noise
            b = random.uniform(1.0 - self.aug_brightness, 1.0 + self.aug_brightness)
            video = (video * b).clamp(-1.0, 1.0)
        if random.random() < self.aug_flip_p:
            video = torch.flip(video, dims=[-1])
        sigma_a = random.uniform(0.0, self.aug_audio_noise)
        if sigma_a > 0:
            audio = audio + torch.randn_like(audio) * sigma_a

        return (video, audio) + tuple(rest)


# ── E2E Lightning Module ───────────────────────────────────────────────────────

class E2EModelA32(L.LightningModule):
    """
    A32: Null-space gradient projection.
    Fine-tune from A9_e2e with:
      - d_unit computed per-epoch from probe batch (200+200 → stable)
      - z_sync_for_task = z_sync projected onto null-space of d_unit
      - Task head only sees dims ⊥ domain direction → gradient can't reinforce gap
      - L1-d mean alignment (λ=0.02) on raw z_sync_pool for active acceleration
      - lambda_ddis=1.0 routes domain signal through z_app
      - lambda_ladv=0, lambda_zsync_dann=0, lambda_orth=0.1
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

        # Light DANN on z_sync (after centering) — lambda 0 by default
        zsync_dim = int(hp.get("sync_dim", 512))
        self.domain_clf_zsync = _make_adv_head(zsync_dim)
        self.lambda_zsync_dann = float(hp.get("lambda_zsync_dann", 0.0))
        self.grl_alpha_zsync   = float(hp.get("grl_alpha_zsync",   1.0))

        # L1-d mean alignment: λ*mean(|mu0-mu1|/std) — constant gradient
        self.lambda_mean_align = float(hp.get("lambda_mean_align", 0.02))

        self.lambda_ddis     = float(hp.get("lambda_ddis",       1.0))
        self.lambda_ladv     = float(hp.get("lambda_ladv",       0.0))
        self.lambda_orth     = float(hp.get("lambda_orth",       0.1))
        self.grl_alpha       = float(hp.get("grl_alpha",         1.0))

        self._val_scores: list = []
        self._val_labels: list = []
        self.domain_probe_loader = None  # set in main() after init

        # Domain direction for null-space projection (updated each epoch)
        self.d_unit = None   # [512] unit vector on correct device, or None

    # ── Per-epoch d_unit computation ──────────────────────────────────────────

    def on_train_epoch_start(self):
        """
        Compute the domain direction d_unit = normalize(mu_FAVC - mu_AV1M)
        from the probe batch at the start of each training epoch.
        This gives a stable 200+200 sample estimate, updated as the encoder evolves.
        """
        if self.domain_probe_loader is None:
            self.d_unit = None
            return

        self.eval()
        device = next(self.parameters()).device
        zs_list, dom_list = [], []

        with torch.no_grad():
            for batch in self.domain_probe_loader:
                if batch is None:
                    continue
                video_raw, audio_raw, _, domain_labels, _ = batch
                video_raw = video_raw.to(device)
                audio_raw = audio_raw.to(device)

                video_feats, audio_feats = self.encoder(video_raw, audio_raw)
                z_sync, _, _ = self.head._encode(video_feats, audio_feats)
                zs_list.append(z_sync.mean(1).cpu())
                dom_list.extend(domain_labels.tolist())

        self.train()

        zs_all  = torch.cat(zs_list)          # [400, 512]
        dom_all = torch.tensor(dom_list)

        mu0    = zs_all[dom_all == 0].mean(0)  # [512]
        mu1    = zs_all[dom_all == 1].mean(0)  # [512]
        gap    = mu1 - mu0
        gap_norm = gap.norm().item()

        if gap_norm > 1e-6:
            self.d_unit = (gap / gap_norm).to(device)   # unit vector [512]
        else:
            self.d_unit = None  # gap ≈ 0 — no projection needed!

        print(
            f"\n[A32] Epoch {self.current_epoch}: "
            f"||mu1-mu0||={gap_norm:.4f}  "
            f"d_unit={'computed' if self.d_unit is not None else 'NULL (gap≈0!)'}",
            flush=True,
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        video_raw, audio_raw, cls_labels, domain_labels, _ = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)

        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        z_sync, z_app, _ = self.head._encode(video_feats, audio_feats)
        z_sync_pool = z_sync.mean(1)   # [B, 512]
        z_app_pool  = z_app.mean(1)    # [B, 256]

        mask0 = (domain_labels < 0.5)
        mask1 = (domain_labels >= 0.5)
        n0, n1 = mask0.sum().item(), mask1.sum().item()

        # ── A32 core: NULL-SPACE PROJECTION ───────────────────────────────────
        #
        # d_unit (updated each epoch) = normalize(mu_FAVC - mu_AV1M)
        # This is the direction in z_sync space that a linear classifier uses
        # to separate the two domains.
        #
        # Project z_sync onto the complementary subspace:
        #   z_sync_for_task[b,t,:] = z_sync[b,t,:] - (z_sync[b,t,:] · d_unit) * d_unit
        #
        # Task gradient can ONLY flow through dims ⊥ d_unit.
        # → encoder is trained to encode fake/real in the null-space of d_unit
        # → encoder abandons d_unit dimension → mean gap naturally collapses
        #
        # L1-d alignment on raw z_sync_pool (λ=0.02) directly closes the gap,
        # accelerating what the projection is already steering toward.
        #
        if self.d_unit is not None and n0 >= 1 and n1 >= 1:
            # Null-space projection: [B, T, 512] → [B, T, 512]
            proj_coeff    = z_sync.matmul(self.d_unit)            # [B, T]
            z_sync_for_task = z_sync - proj_coeff.unsqueeze(-1) * self.d_unit  # [B, T, 512]
        else:
            z_sync_for_task = z_sync  # no projection (gap already ≈ 0)

        # ── Mean alignment on raw z_sync_pool (L1-d, constant gradient) ──────
        if n0 >= 1 and n1 >= 1:
            mu0 = z_sync_pool[mask0].mean(0)   # [512]
            mu1 = z_sync_pool[mask1].mean(0)   # [512]
            with torch.no_grad():
                std_combined = z_sync_pool.std(0).clamp(min=1e-6)
            loss_mean_align = (mu0 - mu1).abs().div(std_combined).mean()
        else:
            loss_mean_align = torch.zeros(1, device=z_sync_pool.device).squeeze()

        # ── Light DANN on z_sync (lambda=0 — disabled but kept for logging) ──
        z_sync_pool_c_grl  = grad_reverse(z_sync_pool.clone(), alpha=self.grl_alpha_zsync)
        domain_logit_zsync = self.domain_clf_zsync(z_sync_pool_c_grl).squeeze(-1)
        loss_zsync_dann    = F.binary_cross_entropy_with_logits(
            domain_logit_zsync, domain_labels
        )
        with torch.no_grad():
            domain_acc_zsync = (
                (domain_logit_zsync.detach() > 0).float() == domain_labels
            ).float().mean()

        # ── z_app domain discriminative loss (routes domain to spurious stream)
        d_score_app = self.head.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Task + orth (AV1M samples only, using projected z_sync) ─────────
        av1m_mask = (cls_labels >= 0)
        if av1m_mask.any():
            z_sync_av1m_proj = z_sync_for_task[av1m_mask]   # projected
            z_app_av1m       = z_app[av1m_mask]
            lbl_av1m         = cls_labels[av1m_mask]
            z_app_pool_av1m  = z_app_pool[av1m_mask]

            score_task = self.head._cls_score(self.head.causal_head, z_sync_av1m_proj)
            loss_task  = self.head._ce_loss(score_task, lbl_av1m)
            loss_orth  = self.head._orth_loss(z_sync_av1m_proj, z_app_av1m)

            z_app_rev   = grad_reverse(z_app_pool_av1m, self.grl_alpha)
            l_score_app = self.head.label_head_s(z_app_rev).squeeze(-1)
            loss_ladv   = F.binary_cross_entropy_with_logits(
                l_score_app, lbl_av1m.float()
            )
        else:
            zero = torch.zeros(1, device=video_feats.device).squeeze()
            loss_task = loss_orth = loss_ladv = zero

        loss = (
              loss_task
            + self.lambda_mean_align  * loss_mean_align
            + self.lambda_zsync_dann  * loss_zsync_dann
            + self.lambda_ddis        * loss_ddis
            + self.lambda_ladv        * loss_ladv
            + self.lambda_orth        * loss_orth
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",              loss,              **log_kw)
        self.log("train_loss_task",         loss_task,         **log_kw)
        self.log("train_loss_mean_align",   loss_mean_align,   **log_kw)
        self.log("train_loss_zsync_dann",   loss_zsync_dann,   **log_kw)
        self.log("train_domain_acc_zsync",  domain_acc_zsync,  **log_kw)
        self.log("train_loss_ddis",         loss_ddis,         **log_kw)
        self.log("train_loss_ladv",         loss_ladv,         **log_kw)
        self.log("train_loss_orth",         loss_orth,         **log_kw)
        self.log("train_n_domain0",         float(n0),         **log_kw)
        self.log("train_n_domain1",         float(n1),         **log_kw)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _domain, _paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        cls_labels = cls_labels.long()
        z_sync, _, _ = self.head._encode(video_feats, audio_feats)

        av1m_mask = (cls_labels >= 0)
        if not av1m_mask.any():
            return

        # Validation uses full z_sync (not projected) — measures raw task AUC
        score = self.head._cls_score(self.head.causal_head, z_sync[av1m_mask])
        score = score[:, 1] if score.ndim == 2 else score
        self._val_scores.append(score.detach().cpu())
        self._val_labels.append(cls_labels[av1m_mask].cpu())

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

        # ── Per-epoch domain probe ─────────────────────────────────────────────
        if self.domain_probe_loader is not None:
            enc_auc_raw, zsync_auc_raw, zsync_auc_centered = self._run_domain_probe()
            epoch = self.current_epoch
            print(
                f"\n[Epoch {epoch:02d}] Domain probe RAW (true invariance): "
                f"enc AUC={enc_auc_raw:.4f}  z_sync AUC={zsync_auc_raw:.4f}"
                f"  (target <0.65)  task AUC={auc:.4f}",
                flush=True,
            )
            print(
                f"[Epoch {epoch:02d}] Domain probe CENTERED (residual): "
                f"z_sync AUC={zsync_auc_centered:.4f}",
                flush=True,
            )
            self.log("domain_auc_enc_raw",        enc_auc_raw,        on_epoch=True, prog_bar=False)
            self.log("domain_auc_zsync_raw",      zsync_auc_raw,      on_epoch=True, prog_bar=False)
            self.log("domain_auc_zsync_centered", zsync_auc_centered, on_epoch=True, prog_bar=False)

    def _run_domain_probe(self):
        """
        Domain probe: 5-fold stratified CV on 400+400 samples.
        Returns (auc_enc_raw, auc_zsync_raw, auc_zsync_centered).
        """
        self.eval()
        enc_list, zs_list, dom_list = [], [], []
        device = next(self.parameters()).device

        for batch in self.domain_probe_loader:
            if batch is None:
                continue
            video_raw, audio_raw, _, domain_labels, _ = batch
            video_raw = video_raw.to(device)
            audio_raw = audio_raw.to(device)

            video_feats, audio_feats = self.encoder(video_raw, audio_raw)
            z_sync, _, _ = self.head._encode(video_feats, audio_feats)

            enc_feats = torch.cat([video_feats.mean(1), audio_feats.mean(1)], dim=-1)
            enc_list.append(enc_feats.cpu().numpy())
            zs_list.append(z_sync.mean(1).cpu().numpy())
            dom_list.extend(domain_labels.tolist())

        enc_all = np.concatenate(enc_list)
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

        auc_enc_raw   = _probe(enc_all, dom_all)
        auc_zsync_raw = _probe(zs_all,  dom_all)

        zs_centered = zs_all.copy()
        for d in [0, 1]:
            idx_d = (dom_all == d)
            if idx_d.sum() > 1:
                zs_centered[idx_d] = zs_centered[idx_d] - zs_centered[idx_d].mean(0)
        auc_zsync_centered = _probe(zs_centered, dom_all)

        self.train()
        return auc_enc_raw, auc_zsync_raw, auc_zsync_centered

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder",   5e-6))
        lr_head      = float(hp.get("lr",           1e-4))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = (list(self.head.parameters())
                          + list(self.domain_clf_zsync.parameters()))

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

    aug_cfg     = config.get("augmentation", {})
    do_augment  = bool(aug_cfg.get("enabled", True))
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

    favc_oversample = int(config.get("favc_oversample", 3))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(
            favc_root, max_frames=max_frames, real_only=False
        )
        favc_dom = DomainLabeledDataset(
            favc_ds, domain_label=1, cls_label_override=-1
        )
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
            print(f"[load_data] FAVC oversampled {favc_oversample}× → {len(favc_dom)} clips",
                  flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — "
              "training without FAVC domain signal.", flush=True)
        train_ds = av1m_train

    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    def _collect_domain_labels(ds):
        if isinstance(ds, ConcatDataset):
            labels = []
            for sub in ds.datasets:
                labels.extend(_collect_domain_labels(sub))
            return labels
        elif isinstance(ds, (DomainLabeledDataset, AugmentedDataset)):
            inner = ds.dataset if isinstance(ds, AugmentedDataset) else ds
            if isinstance(inner, DomainLabeledDataset):
                return [inner.domain_label] * len(inner)
            return _collect_domain_labels(ds.dataset)
        else:
            return [0] * len(ds)

    domain_labels = _collect_domain_labels(train_ds)
    n0 = sum(1 for l in domain_labels if l == 0)
    n1 = sum(1 for l in domain_labels if l == 1)
    print(f"[load_data] BalancedDomainSampler: {n0} domain-0 (AV1M), {n1} domain-1 (FAVC)",
          flush=True)

    sampler = BalancedDomainSampler(domain_labels, batch_size=batch_size, drop_last=True)

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

    n_probe = 200
    rng_probe = np.random.default_rng(0)
    av1m_probe_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    av1m_probe = DomainLabeledDataset(
        Subset(av1m_probe_base, list(range(min(n_probe, len(av1m_probe_base))))),
        domain_label=0
    )
    domain_probe_loader = None
    if favc_root and os.path.isdir(favc_root):
        favc_probe_base = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
        favc_probe_idx  = rng_probe.choice(len(favc_probe_base),
                                           size=min(n_probe, len(favc_probe_base)),
                                           replace=False).tolist()
        favc_probe = DomainLabeledDataset(
            Subset(favc_probe_base, favc_probe_idx),
            domain_label=1, cls_label_override=-1
        )
        probe_ds = ConcatDataset([av1m_probe, favc_probe])
        domain_probe_loader = DataLoader(
            probe_ds, batch_size=8, shuffle=False,
            num_workers=2, collate_fn=e2e_collate_fn,
        )
        print(f"[load_data] Domain probe loader: {len(av1m_probe)} AV1M + "
              f"{len(favc_probe)} FAVC (no augmentation)", flush=True)

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
    model = E2EModelA32(config=config)
    model.domain_probe_loader = domain_probe_loader

    hp = config.get("model_hparams", {})
    finetune_ckpt = hp.get("finetune_ckpt", "")
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        print(f"[A32] Loading weights from {finetune_ckpt} (strict=False)", flush=True)
        ckpt_data  = torch.load(finetune_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[A32] Missing keys (new modules, OK): {missing[:10]}", flush=True)
        print(f"[A32] Unexpected keys (removed modules, OK): {unexpected[:5]}", flush=True)
        print(
            f"[A32] lambda_mean_align={model.lambda_mean_align}  "
            f"lambda_ladv={model.lambda_ladv}  "
            f"lambda_ddis={model.lambda_ddis}  (null-space projection + L1-d)",
            flush=True,
        )
    else:
        print("[A32] WARNING: no finetune_ckpt found — starting from scratch.", flush=True)

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",   {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A32_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A32_e2e/ckpts"),
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

    accum  = int(config.get("accumulate_grad_batches", 4))
    clip   = float(config.get("gradient_clip_val", 2.0))
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
    print(">>> A32_e2e training complete.", flush=True)


if __name__ == "__main__":
    main()
