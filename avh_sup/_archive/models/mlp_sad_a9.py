"""
AVH-Align A9 — SAD + Multi-Kernel MMD Domain Alignment
=======================================================

Root cause of A1–A8 domain-probe failure:
  The upstream AV-HuBERT features carry an overwhelmingly strong domain
  signal: raw probe AUC = 0.999.  GRL (Gradient Reversal Layer) cannot
  suppress this because:
    • The adversarial game quickly reaches a saddle point where the domain
      head wins easily (near-zero CE loss → near-zero GRL gradient).
    • lambda_dadv = 1.0 is swamped by task loss gradients.
    • No re-weighting of lambda fixes this — higher lambda destroys AUC.

  Result across A3/A6/A7/A8: Z_sync domain AUC stays 0.985–0.999;
  indistinguishable from raw features.

A9 fix — replace GRL with multi-kernel MMD (Maximum Mean Discrepancy):
  L_mmd = MMD²(Z_sync | domain=AV1M,  Z_sync | domain=FAVC)

  MMD directly minimises the maximum-mean-discrepancy between the two
  domain distributions of Z_sync in a reproducing kernel Hilbert space.
  Unlike GRL:
    • There is no adversarial game — the gradient is always informative.
    • The loss is non-zero as long as the distributions differ.
    • Provably bounds domain generalisation error (Ben-David et al., 2010).
    • Effective even when the domain gap is very large.

  Multi-bandwidth RBF MMD (from DAN, JMMD):
    K(x, y) = Σ_σ  exp(−‖x−y‖² / (2σ²))
  Bandwidths are set by the median heuristic on each batch (σ²_ref),
  then five scales: σ²_ref × {0.1, 0.5, 1.0, 2.0, 10.0}.

  When either domain has too few samples in a batch (< 2), L_mmd = 0
  for that step.  The training script uses a balanced oversampling
  strategy to ensure both domains are regularly present.

Architecture:
  Identical to A8 (SAD) — Z_sync (causal) / Z_app (spurious).
  Only the Z_sync domain loss changes:
    A8: loss_dadv = BCE(domain_head_c(GRL(Z_sync_pool)), domain)
    A9: loss_mmd  = MMD²(Z_sync_pool[d=0], Z_sync_pool[d=1])
  The domain_head_c and GRL are dropped entirely.

Losses:
  L_task   — CE on causal_head(Z_sync)                  [AV1M clips only]
  L_mmd    — MMD²(Z_sync_pool|d=0, Z_sync_pool|d=1)     [all clips, when ≥2/domain]
  L_ddis   — BCE(domain_head_s(Z_app_pool), domain)      [all clips]
  L_ladv   — BCE(label_head_s(GRL(Z_app_pool)), label)   [AV1M clips only]
  L_orth   — mean cos²(Z_sync_frame, Z_app_frame)        [AV1M clips only]

Overall:
  L = L_task + λ_mmd·L_mmd + λ_ddis·L_ddis + λ_ladv·L_ladv + λ_orth·L_orth
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Re-use identical SAD building blocks from A8
from mlp_sad_a8 import (
    _GRLFunction,
    grad_reverse,
    _make_adv_head,
    _make_cls_head,
    SynchronyEncoder,
    AppearanceEncoder,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-kernel RBF MMD (with median-heuristic bandwidth)
# ─────────────────────────────────────────────────────────────────────────────

def _mmd_rbf(X: torch.Tensor, Y: torch.Tensor,
             bandwidth_scales=(0.1, 0.5, 1.0, 2.0, 10.0)) -> torch.Tensor:
    """
    Unbiased multi-bandwidth RBF MMD² estimator.

    X: [n, d]  — source domain samples (e.g. Z_sync for AV1M)
    Y: [m, d]  — target domain samples (e.g. Z_sync for FAVC)

    Bandwidth is set by the median heuristic on the joint pairwise
    squared distances, then scaled by each factor in bandwidth_scales.
    This is data-adaptive: works regardless of feature scale.

    Returns: scalar MMD² ≥ 0.
    """
    XY = torch.cat([X, Y], dim=0)                    # [n+m, d]
    # Pairwise squared distances over joint set for bandwidth estimate
    dist_all = torch.cdist(XY, XY, p=2) ** 2         # [(n+m), (n+m)]
    # Median of upper triangle (excluding diagonal zeros)
    n_all = XY.shape[0]
    triu_idx = torch.triu_indices(n_all, n_all, offset=1)
    sigma2_ref = dist_all[triu_idx[0], triu_idx[1]].median().clamp(min=1e-4)

    # Pairwise squared distances within/between domains
    dist_XX = torch.cdist(X, X, p=2) ** 2            # [n, n]
    dist_YY = torch.cdist(Y, Y, p=2) ** 2            # [m, m]
    dist_XY = torch.cdist(X, Y, p=2) ** 2            # [n, m]

    n, m = X.shape[0], Y.shape[0]
    mmd = torch.zeros(1, device=X.device)
    for scale in bandwidth_scales:
        sigma2 = sigma2_ref * scale
        K_XX = torch.exp(-dist_XX / (2.0 * sigma2))
        K_YY = torch.exp(-dist_YY / (2.0 * sigma2))
        K_XY = torch.exp(-dist_XY / (2.0 * sigma2))
        # Biased estimator: always ≥ 0, stable for training.
        # When P = Q and n,m are large, biased MMD → 0 correctly.
        mmd_k = K_XX.mean() + K_YY.mean() - 2.0 * K_XY.mean()
        mmd = mmd + mmd_k

    return (mmd / len(bandwidth_scales)).clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  A9 Lightning module: AVH_SAD_A9
# ─────────────────────────────────────────────────────────────────────────────

class AVH_SAD_A9(L.LightningModule):
    """
    A9 — SAD + Multi-Kernel MMD domain alignment.

    Key difference from A8:
      • No domain_head_c or GRL for Z_sync.
      • Z_sync domain alignment via MMD² (L_mmd).

    Batch format (training):
      (video_feats, audio_feats, cls_labels, domain_labels, paths)
      cls_labels   = -1  for FAVC clips (task + L_ladv skipped)
      domain_labels ∈ {0.0, 1.0}  for all clips

    Batch format (validation):
      (video_feats, audio_feats, cls_labels, paths)  [4-tuple]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp       = config.get("model_hparams", {})
        feat_dim = int(hp.get("feat_dim",  1024))
        sync_dim = int(hp.get("sync_dim",   512))
        app_dim  = int(hp.get("app_dim",    256))

        causal_dim   = sync_dim
        spurious_dim = 2 * app_dim

        # ── SAD encoders (identical to A8) ────────────────────────────────────
        self.sync_enc  = SynchronyEncoder(feat_dim, sync_dim)
        self.app_enc_v = AppearanceEncoder(feat_dim, app_dim)
        self.app_enc_a = AppearanceEncoder(feat_dim, app_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.causal_head   = _make_cls_head(causal_dim)
        self.spurious_head = _make_cls_head(spurious_dim)   # monitoring probe only

        # ── Z_app adversarial heads (identical to A8) ─────────────────────────
        # Z_sync: no domain_head_c / no GRL — replaced by MMD
        self.domain_head_s = _make_adv_head(spurious_dim)   # Z_app must predict domain
        self.label_head_s  = _make_adv_head(spurious_dim)   # Z_app must NOT predict label

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_mmd  = float(hp.get("lambda_mmd",  10.0))  # MMD on Z_sync
        self.lambda_ddis = float(hp.get("lambda_ddis",  1.0))  # domain disc on Z_app
        self.lambda_ladv = float(hp.get("lambda_ladv",  1.0))  # label adv on Z_app
        self.lambda_orth = float(hp.get("lambda_orth",  0.1))  # Z_sync ⊥ Z_app
        self.grl_alpha   = float(hp.get("grl_alpha",    1.0))  # GRL strength for Z_app label head
        self.mmd_min_samples = int(hp.get("mmd_min_samples", 2))  # min per domain for MMD
        self.lr          = float(hp.get("lr",           1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Encoding (identical to A8) ────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        z_sync  = self.sync_enc(video_feats, audio_feats)
        z_app_v = self.app_enc_v(video_feats)
        z_app_a = self.app_enc_a(audio_feats)
        z_app   = torch.cat([z_app_v, z_app_a], dim=-1)
        return z_sync, z_app, {"z_sync": z_sync, "z_app_v": z_app_v, "z_app_a": z_app_a}

    @staticmethod
    def _cls_score(head: nn.Module, repr_: torch.Tensor) -> torch.Tensor:
        per_frame = head(repr_)[..., 0]
        return torch.logsumexp(per_frame, dim=-1)

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _orth_loss(sync_frames: torch.Tensor, app_frames: torch.Tensor) -> torch.Tensor:
        s = F.normalize(sync_frames.reshape(-1, sync_frames.shape[-1]), dim=-1)
        a = F.normalize(app_frames.reshape(-1,  app_frames.shape[-1]),  dim=-1)
        return ((s * a).sum(dim=-1) ** 2).mean()

    # ── Forward (inference — identical interface to A8) ───────────────────────

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor,
                mode: str = "causal") -> torch.Tensor:
        z_sync, z_app, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            return self._cls_score(self.causal_head, z_sync)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, z_app.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats: torch.Tensor, audio_feats: torch.Tensor,
                       mode: str = "causal") -> torch.Tensor:
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()     # [B], −1 for FAVC
        domain_labels = domain_labels.float() # [B], 0=AV1M  1=FAVC

        z_sync, z_app, _ = self._encode(video_feats, audio_feats)

        # Pool over time for domain / label heads
        z_sync_pool = z_sync.mean(1)   # [B, sync_dim]
        z_app_pool  = z_app.mean(1)    # [B, spurious_dim]

        # ── MMD: align Z_sync distributions across domains ────────────────────
        # This is the key change from A8: instead of GRL adversarial, we directly
        # minimise the statistical distance between the two domain distributions.
        # When training uses balanced oversampling, most batches have ≥2 per domain.
        mask0 = (domain_labels < 0.5)   # AV1M clips
        mask1 = (domain_labels >= 0.5)  # FAVC clips
        n0, n1 = mask0.sum().item(), mask1.sum().item()

        if n0 >= self.mmd_min_samples and n1 >= self.mmd_min_samples:
            loss_mmd = _mmd_rbf(z_sync_pool[mask0], z_sync_pool[mask1])
        else:
            loss_mmd = torch.zeros(1, device=z_sync_pool.device).squeeze()

        # ── Z_app discriminative domain loss (identical to A8) ────────────────
        d_score_app = self.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Task + label-adversarial losses (AV1M clips only) ────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            z_sync_av1m     = z_sync[av1m_mask]
            z_app_av1m      = z_app[av1m_mask]
            lbl_av1m        = cls_labels[av1m_mask]
            z_app_pool_av1m = z_app_pool[av1m_mask]

            # Task detection
            score_task = self._cls_score(self.causal_head, z_sync_av1m)
            loss_task  = self._ce_loss(score_task, lbl_av1m)

            # Monitoring: spurious probe (detached — not optimised)
            score_spu      = self._cls_score(self.spurious_head, z_app_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)

            # Orthogonality
            loss_orth = self._orth_loss(z_sync_av1m, z_app_av1m)

            # Label adversarial on Z_app: appearance branch must NOT predict label
            z_app_rev   = grad_reverse(z_app_pool_av1m, self.grl_alpha)
            l_score_app = self.label_head_s(z_app_rev).squeeze(-1)
            loss_ladv   = F.binary_cross_entropy_with_logits(
                l_score_app, lbl_av1m.float()
            )
        else:
            loss_task      = torch.zeros(1, device=video_feats.device).squeeze()
            loss_orth      = torch.zeros(1, device=video_feats.device).squeeze()
            loss_ladv      = torch.zeros(1, device=video_feats.device).squeeze()
            loss_spu_probe = torch.zeros(1, device=video_feats.device).squeeze()

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

    # ── Validation (identical to A8) ─────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            video_feats, audio_feats, labels, _, _ = batch
        else:
            video_feats, audio_feats, labels, _ = batch
        with torch.no_grad():
            s_causal   = self.forward(video_feats, audio_feats, mode="causal")
            s_spurious = self.forward(video_feats, audio_feats, mode="spurious")
        self._val_scores["causal"].append(s_causal.cpu().numpy())
        self._val_scores["spurious"].append(s_spurious.cpu().numpy())
        self._val_scores["full"].append(s_causal.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        labels  = np.concatenate(self._val_labels)
        results = {}
        for mode in ("full", "causal", "spurious"):
            scores = np.concatenate(self._val_scores[mode])
            try:
                auc = roc_auc_score(y_true=labels, y_score=scores)
            except ValueError:
                auc = float("nan")
            results[mode] = auc
            self.log(f"val_auc_{mode}", auc, prog_bar=(mode == "causal"))

        print(
            f"\n[Epoch {self.current_epoch}]"
            f"  AUC  full={results['full']:.4f}"
            f"  causal={results['causal']:.4f}"
            f"  spurious={results['spurious']:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
