"""
E2E Training — A23: RFF-MMD Directly on z_sync_pool + Enc-level DANN
================================================================================

**Motivation**:
  A15 applies RFF-MMD on enc_feats [B,2048] (raw AV-HuBERT outputs).  The SAD
  head then projects enc_feats → z_sync.  If this projection is not itself
  domain-invariant, domain signal can leak back INTO z_sync even if enc_feats
  are aligned.  A23 targets z_sync_pool [B,512] — the EXACT representation
  whose domain AUC we measure — with the MMD loss directly.

**Key differences from earlier experiments**:
  - A10-A14 (CORAL):  CORAL computes FULL covariance. With n=4 per domain from
    R^2048 or R^512, the sample covariance is rank≤3. Loss collapses to ~1e-3
    while 2045+ directions remain completely unconstrained. Domain AUC stays 1.0.

  - A15 (RFF-MMD on enc_feats): No rank issue; applies at encoder level only.
    If the SAD projection re-introduces domain cues, A15 cannot fix z_sync.

  - A23 (RFF-MMD on z_sync_pool): Applies the kernel test DIRECTLY to z_sync,
    with no rank requirement. The RBF kernel implicitly matches ALL moments of
    the distribution, not just covariance. When MMD → 0, the z_sync distribution
    is indistinguishable across domains → linear probe AUC → 0.5.

**RFF-MMD details** (same as A15, adapted for z_sync):
  W ∈ R^{512×256}: fixed Gaussian random matrix (frozen buffer)
  b ∈ R^{256}: fixed uniform random phase offsets
  φ(x) = √(2/D) · cos(x · W/σ + b)  where σ = per-batch median pairwise dist
  L_mmd = MSE(φ(Z₀).mean(0), φ(Z₁).mean(0))   (Z₀=AV1M, Z₁=FAVC), lambda=25.0
  BalancedDomainSampler ensures n0==n1==4 every step → MMD never skipped.

**Enc-level DANN** (lambda=15.0, updated from 10.0): stronger upstream pressure
  after A18 showed lambda=1 DANN (epoch 0: z_sync 0.9928, epoch 1 regression to 0.9984)
  is insufficient. lambda=15 matches A21/A22 configs.

**lambda_ddis=0.0**: removes the conflicting gradient that trains domain_head_s
  to predict domain from z_app (would push domain info INTO the encoder).

  python train_e2e_a23.py --config_path configs/A23_e2e.yaml
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


# ── Random Fourier Feature MMD ─────────────────────────────────────────────────────
#
# RFF approximates the RBF kernel MMD, which measures the FULL distributional
# discrepancy between two feature sets (not just mean or covariance).
# No rank requirements: works with n=2 samples per domain (unlike CORAL which
# needs n > d to produce a full-rank covariance estimate).
#
# W [d, D] and b [D] are Gaussian random weights stored as frozen buffers.
# They define a fixed RKHS feature map phi(x) = sqrt(2/D)*cos(Wx/sigma + b).
# sigma is set per-batch via the median pairwise distance (data-adaptive).
# L_mmd = ||E[phi(Z_0)] - E[phi(Z_1)]||^2_2  -> 0 iff distributions match.

def _rff_mmd(X: torch.Tensor, Y: torch.Tensor,
             W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Random Fourier Feature MMD between source X [N,d] and target Y [M,d].
    phi(x) = sqrt(2/D) * cos(x @ W_scaled + b)  where W_scaled = W / sigma.
    sigma = per-batch median pairwise distance (median heuristic, data-adaptive).
    W [d,D] and b [D] are fixed random buffers. Returns scalar >= 0.
    """
    XY = torch.cat([X, Y], dim=0)
    pairwise = torch.cdist(XY, XY, p=2)
    n = XY.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    sigma = pairwise[idx[0], idx[1]].median().clamp(min=1e-3)

    W_scaled = W / sigma
    phi_X = torch.cos(X @ W_scaled + b) * math.sqrt(2.0 / W.shape[1])  # [N, D]
    phi_Y = torch.cos(Y @ W_scaled + b) * math.sqrt(2.0 / W.shape[1])  # [M, D]
    return F.mse_loss(phi_X.mean(0), phi_Y.mean(0))


# ── Simple CORAL domain alignment loss ───────────────────────────────────────
#
# Directly attacks what a linear probe exploits: per-feature means and
# second-order covariance structure.  No domain head, no adversarial game.
#
# L_coral = MSE(mean_0, mean_1) + ||Cov_0 - Cov_1||²_F / (4d²)
#
# When L_coral → 0, Z_sync has identical mean and covariance across both
# domains → linear probe AUC → 0.5 (random chance).

def _coral(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss between source X [N,d] and target Y [M,d].
    Always non-negative, zero iff X and Y have identical mean+covariance.
    """
    d = X.shape[1]

    # Mean alignment (most important term for linear separability)
    loss_mean = F.mse_loss(X.mean(0), Y.mean(0))

    # Covariance alignment
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Cs = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    Ct = (Yc.T @ Yc) / max(Y.shape[0] - 1, 1)
    loss_cov = ((Cs - Ct) ** 2).sum() / (4.0 * d * d)

    return loss_mean + loss_cov
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

# ── Balanced domain sampler ───────────────────────────────────────────────────
#
# Guarantees EXACTLY ceil(batch_size/2) samples from each domain per batch.
# This ensures _coral() is always called with sufficient examples from both
# domains — fixing the root cause of A9/A9_e2e where FAVC appeared in only
# ~11% of batches and MMD was silently skipped 89% of the time.

class BalancedDomainSampler(Sampler):
    """
    Yields indices such that each batch of `batch_size` contains exactly
    batch_size//2 samples from domain=0 (AV1M) and batch_size//2 from
    domain=1 (FAVC).  The last (odd) slot goes to domain=0 if batch_size is odd.

    Parameters
    ----------
    domain_labels : list[int]  — domain label (0 or 1) for every dataset item
    batch_size     : int        — total batch size (must be ≥ 2)
    drop_last      : bool       — drop the last incomplete batch (default True)
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

        # Repeat shorter list to match length of longer
        n     = max(len(idx0), len(idx1))
        idx0  = np.resize(idx0, n)
        idx1  = np.resize(idx1, n)

        n_batches = n // self._n_per_domain
        if self.drop_last:
            n_batches = n_batches  # already floor division

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


# ── Input augmentation wrapper ────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """
    Wraps any dataset returning (video [T,H,W], audio [T_a,104], ...)
    and applies random video/audio augmentation.

    Augmentations (each applied independently per clip):
      - Gaussian noise on video frames (σ ~ Uniform(0, aug_video_noise))
      - Brightness shift on video frames (δ ~ Uniform(-aug_brightness, +aug_brightness))
      - Horizontal flip on all frames consistently (p=aug_flip_p)
      - Gaussian noise on audio feats (σ ~ Uniform(0, aug_audio_noise))

    Applied to BOTH AV1M and FAVC clips (equally) so the encoder cannot
    use augmentation presence/absence as a domain signal.

    Video frames are AV-HuBERT normalized float32 [T, H, W], zero-centred,
    so adding N(0, 0.3) ≈ adding ±7 units in [0-255] pixel space.
    Audio feats are layer-normalised float32 [T_a, 104].
    """

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
        self.p               = p          # probability of augmenting a clip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video, audio = item[0], item[1]
        rest = item[2:]

        if random.random() > self.p:
            return item

        # ── Video augmentation ────────────────────────────────────────────────
        # Gaussian noise
        sigma_v = random.uniform(0.0, self.aug_video_noise)
        if sigma_v > 0:
            video = video + torch.randn_like(video) * sigma_v

        # Brightness (additive shift in normalised space)
        if self.aug_brightness > 0:
            delta = random.uniform(-self.aug_brightness, self.aug_brightness)
            video = video + delta

        # Horizontal flip (consistent across all frames)
        if random.random() < self.aug_flip_p:
            video = torch.flip(video, dims=[-1])

        # ── Audio augmentation ────────────────────────────────────────────────
        sigma_a = random.uniform(0.0, self.aug_audio_noise)
        if sigma_a > 0:
            audio = audio + torch.randn_like(audio) * sigma_a

        return (video, audio) + rest


# ── E2E Lightning Module (same as A9_e2e — no change to losses) ───────────────

class E2EModelA23(L.LightningModule):
    """
    End-to-end: AVHubertWrapper (fully unfrozen) + AVH_SAD_A9 head.
    A23: RFF-MMD DIRECTLY on z_sync_pool [B,512] + enc-level DANN (lambda=10).

    Key design:
      - RFF-MMD targets z_sync_pool directly (the rep whose domain AUC we track).
        This is the most direct intervention: aligns the EXACT distribution we
        care about, with no rank issues (works with n=4 per domain from R^512).
      - Enc-level DANN (GRL, lambda=10) applies upstream pressure to discourage
        the encoder from producing domain-discriminative features before the head.
      - No z_sync-level DANN (to keep mechanisms distinct from A20/A21).
      - lambda_ddis=0.0: no conflicting gradient.
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

        # A20: domain classifier at enc_feats [2048] level with GRL (from A17)
        enc_dim = 2 * int(hp.get("feat_dim", 1024))  # 2048
        self.domain_clf_enc = _make_adv_head(enc_dim)
        self.lambda_enc_dann = float(hp.get("lambda_enc_dann", 1.0))
        self.grl_alpha_enc   = float(hp.get("grl_alpha_enc",   1.0))

        # A23 new: RFF buffers for z_sync_pool MMD (dim=512, D=256 random features)
        # W ~ N(0,I), b ~ Uniform(0, 2pi) — frozen throughout training.
        # The RBF kernel approximated by these features measures ALL moments of
        # the z_sync distribution, not just mean/covariance.
        n_rff_zsync = int(hp.get("n_rff_zsync", 256))
        self.register_buffer("rff_W_zsync", torch.randn(zsync_dim, n_rff_zsync))
        self.register_buffer("rff_b_zsync", torch.rand(n_rff_zsync) * 2 * math.pi)
        self.lambda_zsync_mmd = float(hp.get("lambda_zsync_mmd", 10.0))

        # No z_sync-level DANN in A23 (mechanism isolation: only MMD targets z_sync)
        # Enc-level DANN is kept at lambda=10 for upstream adversarial pressure.
        self.lambda_zsync_dann = 0.0   # disabled
        self.grl_alpha_zsync   = 0.0

        self.lambda_enc_coral = float(hp.get("lambda_enc_coral", 0.0))  # disabled in A23
        self.lambda_ddis     = float(hp.get("lambda_ddis",       1.0))
        self.lambda_ladv     = float(hp.get("lambda_ladv",       1.0))
        self.lambda_orth     = float(hp.get("lambda_orth",       0.1))
        self.grl_alpha       = float(hp.get("grl_alpha",         1.0))
        self.mmd_min_samples = int  (hp.get("mmd_min_samples",     2))

        self._val_scores: list = []
        self._val_labels: list = []
        self.domain_probe_loader = None  # set in main() after init

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder",   1e-6))
        lr_head      = float(hp.get("lr",           1e-4))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = (list(self.head.parameters())
                          + list(self.domain_clf_enc.parameters()))
        # Note: no domain_clf_zsync in A23 (MMD has no learnable params)

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
        z_sync_pool = z_sync.mean(1)
        z_app_pool  = z_app.mean(1)

        # ── Encoder-level DANN (GRL on enc_feats [2048], lambda=10) ────────────
        # Upstream adversarial pressure: prevents AV-HuBERT from encoding domain
        # into enc_feats in the first place. Gradient: d(task)/dW - alpha * d(dom)/dW.
        mask0 = (domain_labels < 0.5)
        mask1 = (domain_labels >= 0.5)
        n0, n1 = mask0.sum().item(), mask1.sum().item()
        enc_feats = torch.cat([video_feats.mean(1), audio_feats.mean(1)], dim=-1)  # [B, 2048]
        enc_feats_grl    = grad_reverse(enc_feats, alpha=self.grl_alpha_enc)
        domain_logit_enc = self.domain_clf_enc(enc_feats_grl).squeeze(-1)  # [B]
        loss_enc_dann    = F.binary_cross_entropy_with_logits(
            domain_logit_enc, domain_labels
        )
        with torch.no_grad():
            domain_acc_enc = ((domain_logit_enc.detach() > 0).float() == domain_labels).float().mean()

        # ── RFF-MMD directly on z_sync_pool (A23 core contribution) ────────
        # Aligns the FULL distribution of z_sync across domains using kernel MMD.
        # No rank issue: works with n0=n1=4. Gradient flows directly to encoder.
        # BalancedDomainSampler guarantees n0==n1 every batch.
        if n0 >= 2 and n1 >= 2 and self.lambda_zsync_mmd > 0:
            loss_zsync_mmd = _rff_mmd(
                z_sync_pool[mask0], z_sync_pool[mask1],
                self.rff_W_zsync, self.rff_b_zsync
            )
        else:
            loss_zsync_mmd = torch.zeros(1, device=z_sync_pool.device).squeeze()

        # Optional CORAL/enc-coral (disabled, lambda=0 in A23 config)
        if n0 >= 2 and n1 >= 2 and self.lambda_enc_coral > 0:
            loss_enc_coral = _coral(enc_feats[mask0], enc_feats[mask1])
        else:
            loss_enc_coral = torch.zeros(1, device=enc_feats.device).squeeze()

        # Z_app discriminative domain loss
        d_score_app = self.head.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # Task + label-adv + orth (AV1M only)
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
            + self.lambda_enc_dann  * loss_enc_dann
            + self.lambda_zsync_mmd * loss_zsync_mmd
            + self.lambda_enc_coral * loss_enc_coral
            + self.lambda_ddis      * loss_ddis
            + self.lambda_ladv      * loss_ladv
            + self.lambda_orth      * loss_orth
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",                loss,             **log_kw)
        self.log("train_loss_task",           loss_task,        **log_kw)
        self.log("train_loss_enc_dann",       loss_enc_dann,    **log_kw)
        self.log("train_loss_zsync_mmd",      loss_zsync_mmd,   **log_kw)
        self.log("train_loss_enc_coral",      loss_enc_coral,   **log_kw)
        self.log("train_domain_acc_enc",      domain_acc_enc,   **log_kw)
        self.log("train_loss_ddis",           loss_ddis,        **log_kw)
        self.log("train_loss_ladv",           loss_ladv,        **log_kw)
        self.log("train_loss_orth",           loss_orth,        **log_kw)
        self.log("train_loss_spu_probe",      loss_spu_probe,   **log_kw)
        self.log("train_n_domain0",           float(n0),        **log_kw)
        self.log("train_n_domain1",           float(n1),        **log_kw)
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
        Extracts encoder-output features (enc_feats [B,2048]) and Z_sync [B,512]
        for the probe dataset (small balanced AV1M+FAVC, no augmentation).
        Runs a quick logistic regression and returns (enc_auc, zsync_auc).
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

        auc_enc   = _probe(enc_all, dom_all)
        auc_zsync = _probe(zs_all,  dom_all)
        self.train()
        return auc_enc, auc_zsync

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
            enc_auc, zsync_auc = self._run_domain_probe()
            epoch = self.current_epoch
            print(f"\n[Epoch {epoch:02d}] Domain probe: "
                  f"enc_feats AUC={enc_auc:.4f}  Z_sync AUC={zsync_auc:.4f}  "
                  f"(target <0.65) [RFF-MMD on z_sync + enc DANN lambda=10, ddis=0]  task AUC={auc:.4f}",
                  flush=True)
            self.log("domain_auc_enc",   enc_auc,   on_epoch=True, prog_bar=False)
            self.log("domain_auc_zsync", zsync_auc, on_epoch=True, prog_bar=False)


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

    # Augmentation params from config
    aug_cfg     = config.get("augmentation", {})
    do_augment  = bool(aug_cfg.get("enabled", True))
    aug_video_noise = float(aug_cfg.get("video_noise",  0.3))
    aug_brightness  = float(aug_cfg.get("brightness",   0.5))
    aug_flip_p      = float(aug_cfg.get("flip_p",       0.5))
    aug_audio_noise = float(aug_cfg.get("audio_noise",  0.05))
    aug_p           = float(aug_cfg.get("p",            0.9))

    # ── AV1M train (domain=0) ─────────────────────────────────────────────────
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
    # KEY FIX vs A9_e2e: real_only=False, so FAVC has matched real+fake distribution
    # with AV1M.  MMD now targets recording-condition gap, not manipulation content.
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
    # No augmentation at val time
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    # ── BalancedDomainSampler: guarantees 50/50 domain split every batch ──────
    # Build domain label list aligned with train_ds indices.
    # DomainLabeledDataset wraps expose .domain_label; ConcatDataset chains them.
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
            # AugmentedDataset wrapping DomainLabeledDataset
            return _collect_domain_labels(ds.dataset)
        else:
            return [0] * len(ds)  # fallback: treat as domain 0

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

    # ── Domain probe loader: small balanced AV1M+FAVC, NO augmentation ────────
    # Used for per-epoch domain AUC printing.  Quick logistic regression on
    # enc_feats and Z_sync to track whether CORAL is reducing domain signal.
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
    model = E2EModelA23(config=config)
    model.domain_probe_loader = domain_probe_loader

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",   {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A23_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A23_e2e/ckpts"),
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
    print(">>> A23_e2e training complete.", flush=True)


if __name__ == "__main__":
    main()
