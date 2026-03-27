"""
E2E Training — A26: Fine-tune from A9_e2e + Direct Mean Alignment on z_sync_pool
===================================================================================

**Motivation (lessons from A20-A25)**:

  A20-A25 ALL collapsed to task AUC = 0.50.  Root cause: lambda_enc_dann = 10-15
  produces adversarial gradients 10-15x larger than the task gradient, driving the
  encoder toward constant (domain+task uninformative) features.

  A25's z_sync AUC = 0.50 is a PROBE ARTIFACT: the probe applies per-domain
  centering before logistic regression, so ANY representation reports 0.50
  after that transform, regardless of whether the encoder learned invariance.

  A24 (CDAN) was the only genuine signal: z_sync AUC = 0.9187, task = 0.5188.
  But starting from random weights, task can't recover while DANN dominates.

**Key design changes in A26**:

  1.  START FROM A9_e2e CHECKPOINT (val_auc_causal = 1.0000 at epoch 8).
      The encoder is already task-optimised.  We fine-tune GENTLY for domain
      invariance without destroying what's already learned.

  2.  DIRECT MEAN ALIGNMENT instead of DANN (no adversarial arms race):
        loss_mean_align = MSE(z_sync_pool[domain=0].mean(0),
                              z_sync_pool[domain=1].mean(0))
      This directly minimises the mean-shift signal that's driving domain AUC=1.
      No domain classifier, no GRL, no training instability.
      Gradient: 2*(mu_0 - mu_1)/n_d per sample → targeted, stable signal.

  3.  LIGHT DANN on z_sync (lambda=2) to remove RESIDUAL covariance signal
      AFTER mean-alignment has reduced the mean shift.

  4.  NO enc_dann: encoder is already good; don't touch it with adversarial.

  5.  Re-enable lambda_ddis = 1.0:  routes domain information THROUGH z_app
      (the spurious stream), creating incentive to separate domain signal
      from z_sync without any adversarial pressure on z_sync.

  6.  Per-domain centering of z_sync_pool (A25 mechanism):  prevents the task
      gradient from using mean shift — so the task never "re-learns" the domain
      mean as a sync proxy, which would undo the mean alignment.

  7.  Very low lr_encoder = 2e-7 (encoder already at task optimum; we want
      minimal disturbance — only enough to shift the domain means).

  8.  DUAL PROBE:
      - PRIMARY  (uncentered): raw z_sync_pool logistic regression
        → measures TRUE encoder invariance; must drop below 0.65
      - SECONDARY (centered): same after per-domain centering
        → measures residual signal beyond mean shift (A25-style)

**Expected dynamics**:
  epoch 0:  task AUC ~1.0 (loaded from A9_e2e), z_sync_raw AUC ~1.0 (unchanged)
  epoch 1+: mean alignment pushes mu_0 → mu_1; z_sync_raw AUC drops
            task AUC stays high because lr_enc is tiny and ddis routes domain
            signal away from z_sync into z_app

  python train_e2e_a26.py --config_path configs/A26_e2e.yaml
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

        n     = max(len(idx0), len(idx1))
        idx0  = np.resize(idx0, n)
        idx1  = np.resize(idx1, n)

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

class E2EModelA26(L.LightningModule):
    """
    A26: Fine-tune from A9_e2e (task AUC=1.0) with:
      - Direct mean alignment loss on z_sync_pool (no DANN at enc level)
      - Light DANN on z_sync (lambda=2) for residual covariance
      - Per-domain centering to block task gradient from re-learning mean shift
      - lambda_ddis=1.0 to route domain info through z_app
      - Very low lr_encoder=2e-7 to protect task AUC
      - Dual probe: raw (primary, true invariance) + centered (secondary)
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

        # Light DANN on z_sync (after centering) — handles residual covariance
        zsync_dim = int(hp.get("sync_dim", 512))
        self.domain_clf_zsync = _make_adv_head(zsync_dim)
        self.lambda_zsync_dann = float(hp.get("lambda_zsync_dann", 2.0))
        self.grl_alpha_zsync   = float(hp.get("grl_alpha_zsync",   1.0))

        # Mean alignment lambda — direct MSE on domain means of z_sync_pool
        self.lambda_mean_align = float(hp.get("lambda_mean_align", 10.0))

        self.lambda_ddis     = float(hp.get("lambda_ddis",       1.0))
        self.lambda_ladv     = float(hp.get("lambda_ladv",       1.0))
        self.lambda_orth     = float(hp.get("lambda_orth",       0.1))
        self.grl_alpha       = float(hp.get("grl_alpha",         1.0))

        self._val_scores: list = []
        self._val_labels: list = []
        self.domain_probe_loader = None  # set in main() after init

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder",   2e-7))
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

        # ── A26 core: DIRECT MEAN ALIGNMENT ──────────────────────────────────
        # Explicitly minimise ||mean(z_sync_pool[d=0]) - mean(z_sync_pool[d=1])||^2
        # Gradient per sample: 2*(mu_0 - mu_1)/n_d  →  stable, targeted signal
        # Avoids the adversarial arms-race that destroyed task AUC in A20-A25.
        if n0 >= 1 and n1 >= 1:
            mu0 = z_sync_pool[mask0].mean(0)   # [512]
            mu1 = z_sync_pool[mask1].mean(0)   # [512]
            loss_mean_align = F.mse_loss(mu0, mu1)
        else:
            loss_mean_align = torch.zeros(1, device=z_sync_pool.device).squeeze()

        # ── Per-domain centering (A25): prevent task gradient from re-learning
        #    the mean shift that mean_align is trying to erase ─────────────────
        z_sync_pool_c = z_sync_pool.clone()
        if n0 > 1:
            z_sync_pool_c[mask0] = z_sync_pool[mask0] - z_sync_pool[mask0].mean(0)
        if n1 > 1:
            z_sync_pool_c[mask1] = z_sync_pool[mask1] - z_sync_pool[mask1].mean(0)

        # ── Light DANN on CENTERED z_sync (handles residual covariance) ──────
        z_sync_pool_c_grl  = grad_reverse(z_sync_pool_c, alpha=self.grl_alpha_zsync)
        domain_logit_zsync = self.domain_clf_zsync(z_sync_pool_c_grl).squeeze(-1)  # [B]
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

        # ── Task + label-adv + orth (AV1M samples only) ─────────────────────
        av1m_mask = (cls_labels >= 0)
        if av1m_mask.any():
            z_sync_av1m     = z_sync[av1m_mask]
            z_app_av1m      = z_app[av1m_mask]
            lbl_av1m        = cls_labels[av1m_mask]
            z_app_pool_av1m = z_app_pool[av1m_mask]

            score_task = self.head._cls_score(self.head.causal_head, z_sync_av1m)
            loss_task  = self.head._ce_loss(score_task, lbl_av1m)
            loss_orth  = self.head._orth_loss(z_sync_av1m, z_app_av1m)

            z_app_rev   = grad_reverse(z_app_pool_av1m, self.grl_alpha)
            l_score_app = self.head.label_head_s(z_app_rev).squeeze(-1)
            loss_ladv   = F.binary_cross_entropy_with_logits(
                l_score_app, lbl_av1m.float()
            )
            score_spu      = self.head._cls_score(
                self.head.spurious_head, z_app_av1m.detach()
            )
            loss_spu_probe = self.head._ce_loss(score_spu, lbl_av1m)
        else:
            zero = torch.zeros(1, device=video_feats.device).squeeze()
            loss_task = loss_orth = loss_ladv = loss_spu_probe = zero

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
        self.log("train_loss_spu_probe",    loss_spu_probe,    **log_kw)
        self.log("train_n_domain0",         float(n0),         **log_kw)
        self.log("train_n_domain1",         float(n1),         **log_kw)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _domain, _paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        scores = self.head(video_feats, audio_feats, mode="causal")
        self._val_scores.append(scores.detach().cpu())
        self._val_labels.append(cls_labels.cpu())

    @torch.no_grad()
    def _run_domain_probe(self):
        """
        Returns (enc_auc_raw, zsync_auc_raw, zsync_auc_centered).

        - enc_auc_raw:       probe on raw enc_feats     (measures enc invariance)
        - zsync_auc_raw:     probe on raw z_sync_pool   (PRIMARY — TRUE invariance)
        - zsync_auc_centered: probe on centered z_sync  (secondary — residual beyond mean)
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

        def _probe(X, y):
            sc  = StandardScaler().fit(X)
            Xs  = sc.transform(X)
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs").fit(Xs, y)
            prob = clf.predict_proba(Xs)[:, 1]
            try:
                return roc_auc_score(y, prob)
            except Exception:
                return 0.5

        # ── PRIMARY: raw probe (no centering) ─────────────────────────────────
        auc_enc_raw   = _probe(enc_all, dom_all)
        auc_zsync_raw = _probe(zs_all,  dom_all)

        # ── SECONDARY: centered probe (A25-style) ─────────────────────────────
        zs_centered = zs_all.copy()
        for d in [0, 1]:
            idx_d = (dom_all == d)
            if idx_d.sum() > 1:
                zs_centered[idx_d] = zs_centered[idx_d] - zs_centered[idx_d].mean(0)
        auc_zsync_centered = _probe(zs_centered, dom_all)

        self.train()
        return auc_enc_raw, auc_zsync_raw, auc_zsync_centered

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

        # ── Per-epoch domain probe ────────────────────────────────────────────
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
            self.log("domain_auc_enc_raw",       enc_auc_raw,        on_epoch=True, prog_bar=False)
            self.log("domain_auc_zsync_raw",     zsync_auc_raw,      on_epoch=True, prog_bar=False)
            self.log("domain_auc_zsync_centered",zsync_auc_centered, on_epoch=True, prog_bar=False)


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

    # ── AV1M train (domain=0) ─────────────────────────────────────────────────
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

    # ── FAVC ALL clips (domain=1, real+fake) ──────────────────────────────────
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

    # ── AV1M val ──────────────────────────────────────────────────────────────
    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    # ── BalancedDomainSampler ─────────────────────────────────────────────────
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

    # ── Domain probe loader ───────────────────────────────────────────────────
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
    model = E2EModelA26(config=config)
    model.domain_probe_loader = domain_probe_loader

    # ── Load weights from A9_e2e checkpoint (strict=False: new domain_clf_zsync
    #    will be randomly initialised; encoder + head loaded from A9_e2e) ──────
    hp = config.get("model_hparams", {})
    finetune_ckpt = hp.get("finetune_ckpt", "")
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        print(f"[A26] Loading weights from {finetune_ckpt} (strict=False)", flush=True)
        ckpt_data  = torch.load(finetune_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[A26] Missing keys (new modules, OK): {missing[:10]}", flush=True)
        print(f"[A26] Unexpected keys (removed modules, OK): {unexpected[:5]}", flush=True)
    else:
        print("[A26] WARNING: no finetune_ckpt found — starting from scratch.", flush=True)

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",   {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A26_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A26_e2e/ckpts"),
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
    print(">>> A26_e2e training complete.", flush=True)


if __name__ == "__main__":
    main()
