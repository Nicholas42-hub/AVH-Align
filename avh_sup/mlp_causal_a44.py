"""
AVH-Align A44 — Causal + Domain Adversarial + Cross-Modal Sync Alignment
=========================================================================
Extends A42 with a Sinkhorn Distribution Alignment (SDA) loss on the
causal branch that forces v_c and a_c into the same distribution:

  L_SDA = Sinkhorn(sda_proj(mean_pool(v_c)), sda_proj(mean_pool(a_c)))

Motivation (from domain probe analysis):
  A42's Z_c raw AUC = 0.95 — large mean-shift between AV1M and FAVC.
  The GRL cannot fix mean-shift when encoder is frozen.
  SDA explicitly forces v_c ≈ a_c in distribution → since AV sync is
  a domain-agnostic physical signal, this acts as an implicit
  cross-domain alignment (borrowed from FCD's synergistic alignment).

Architecture (same as A42 + SDA module):
  visual_feats [T×1024], audio_feats [T×1024]   (frozen encoder)
      │
      ├── visual_proj_causal   → v_c [T×proj_dim]
      ├── visual_proj_spurious → v_s [T×proj_dim]
      ├── audio_proj_causal    → a_c [T×proj_dim]
      └── audio_proj_spurious  → a_s [T×proj_dim]
      │
      Z_c = concat(v_c, a_c) [T×1024]
      Z_s = concat(v_s, a_s) [T×1024]
      │
      SDA: sda_proj(mean v_c) ≈ sda_proj(mean a_c)  ← NEW in A44
      │
      ├── causal_head(Z_c)               → task classification
      ├── spurious_head(Z_s.detach())    → probe only
      ├── adv_head(GRL(Z_s))             → label adversarial on Z_s
      ├── domain_head_c(GRL(Z_c_pool))   → domain adversarial on Z_c
      └── domain_head_s(Z_s_pool)        → domain discriminative on Z_s

Losses:
  L = L_task
      + λ_adv  · L_adv   (GRL label on Z_s)
      + λ_dadv · L_dadv  (GRL domain on Z_c)
      + λ_ddis · L_ddis  (domain pull on Z_s)
      + λ_orth · L_orth  (Z_c ⊥ Z_s)
      + λ_sda  · L_SDA   (Sinkhorn: v_c ≈ a_c)  ← NEW

At batch_size=1 (most clips), Sinkhorn degenerates to L2:
  L_SDA = || sda_proj(v_c) - sda_proj(a_c) ||²
"""

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ── Gradient Reversal Layer ────────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)


# ── Head factories ─────────────────────────────────────────────────────────────

def _make_cls_head(input_dim: int = 1024) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
        nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(),
        nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 1),
    )


def _make_domain_head(input_dim: int = 1024) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 64),        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ── Sinkhorn Distribution Alignment (from FCD, pure PyTorch) ──────────────────

def _sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 5,
) -> torch.Tensor:
    """
    Debiased Sinkhorn divergence S_ε(x, y) ≥ 0.
    For N=1 (batch_size=1), degenerates to ||x - y||² / D.

    x, y: [N, D] — L2-normalised embeddings.
    """
    def _weps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        N, M = a.shape[0], b.shape[0]
        D    = a.shape[1]
        C    = torch.cdist(a, b).pow(2) / D
        lp   = torch.full((N,), -math.log(N), device=a.device)
        lq   = torch.full((M,), -math.log(M), device=b.device)
        f    = torch.zeros(N, device=a.device)
        g    = torch.zeros(M, device=b.device)
        for _ in range(n_iter):
            f = eps * (lp - torch.logsumexp((-C   + g.unsqueeze(0)) / eps, dim=1))
            g = eps * (lq - torch.logsumexp((-C.T + f.unsqueeze(0)) / eps, dim=1))
        P = torch.exp((-C + f.unsqueeze(1) + g.unsqueeze(0)) / eps)
        return (P * C).sum()

    return _weps(x, y) - 0.5 * _weps(x, x) - 0.5 * _weps(y, y)


# ── SDA projection module (shared by both modalities) ─────────────────────────

class SDA(nn.Module):
    """Shared projection → L2-normalised space for Sinkhorn alignment."""
    def __init__(self, proj_dim: int, sda_dim: int):
        super().__init__()
        self.proj = nn.Linear(proj_dim, sda_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, proj_dim] → [B, sda_dim] L2-normalised."""
        return F.normalize(self.proj(x), dim=-1)


# ── Model ─────────────────────────────────────────────────────────────────────

class AVH_Causal_A44(L.LightningModule):
    """
    A44: A42 + Sinkhorn cross-modal sync alignment on Z_c.

    Config keys (under model_hparams):
        feat_dim          : int   (default 1024)
        proj_dim          : int   (default 512)
        sda_dim           : int   SDA projection dim (default 128)
        sda_eps           : float Sinkhorn ε (default 0.05)
        sda_n_iter        : int   Sinkhorn iterations (default 5)
        lambda_adv        : float label adversarial on Z_s (default 10.0)
        grl_alpha         : float GRL α for label head (default 5.0)
        lambda_dadv       : float domain adversarial on Z_c (default 1.0)
        grl_domain_alpha  : float GRL α for domain head (default 1.0)
        lambda_ddis       : float domain discriminative on Z_s (default 1.0)
        lambda_orth       : float orthogonality (default 0.1)
        lambda_sda        : float Sinkhorn alignment (default 1.0)
        lr                : float (default 1e-3)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})
        feat_dim              = int(hp.get("feat_dim",         1024))
        proj_dim              = int(hp.get("proj_dim",          512))
        sda_dim               = int(hp.get("sda_dim",           128))
        self.sda_eps          = float(hp.get("sda_eps",         0.05))
        self.sda_n_iter       = int(hp.get("sda_n_iter",          5))
        self.lambda_adv       = float(hp.get("lambda_adv",      10.0))
        grl_alpha             = float(hp.get("grl_alpha",        5.0))
        self.lambda_dadv      = float(hp.get("lambda_dadv",      1.0))
        grl_domain_alpha      = float(hp.get("grl_domain_alpha", 1.0))
        self.lambda_ddis      = float(hp.get("lambda_ddis",      1.0))
        self.lambda_orth      = float(hp.get("lambda_orth",      0.1))
        self.lambda_sda       = float(hp.get("lambda_sda",       1.0))
        self.lr               = float(hp.get("lr",              1e-3))

        self.proj_dim = proj_dim
        fused_dim = proj_dim * 2

        # ── Projections ───────────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── SDA: shared projection for cross-modal Sinkhorn alignment ────────
        self.sda = SDA(proj_dim, sda_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.causal_head   = _make_cls_head(fused_dim)
        self.spurious_head = _make_cls_head(fused_dim)  # probe (detached)

        # ── Label adversarial on Z_s ──────────────────────────────────────────
        self.adv_head  = _make_cls_head(fused_dim)
        self.grl_label = GradientReversal(alpha=grl_alpha)

        # ── Domain adversarial on Z_c ─────────────────────────────────────────
        self.domain_head_c = _make_domain_head(fused_dim)
        self.grl_domain    = GradientReversal(alpha=grl_domain_alpha)

        # ── Domain discriminative on Z_s ──────────────────────────────────────
        self.domain_head_s = _make_domain_head(fused_dim)

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """Returns (z_c [B,T,2P], z_s [B,T,2P], v_c [B,T,P], a_c [B,T,P])."""
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)
        return (
            torch.cat([v_c, a_c], dim=-1),
            torch.cat([v_s, a_s], dim=-1),
            v_c,
            a_c,
        )

    @staticmethod
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(head(repr_)[..., 0], dim=-1)

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(torch.stack([-score, score], dim=1), labels)

    @staticmethod
    def _orth_loss(z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        B, T, D = z_c.shape
        c = F.normalize(z_c.reshape(-1, D), dim=-1)
        s = F.normalize(z_s.reshape(-1, D), dim=-1)
        return ((c * s).sum(dim=-1) ** 2).mean()

    # ── Forward (inference) ───────────────────────────────────────────────────

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        z_c, z_s, _, _ = self._encode(video_feats, audio_feats)
        if mode in ("full", "causal"):
            return self._cls_score(self.causal_head, z_c)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, z_s)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        z_c, z_s, v_c, a_c = self._encode(video_feats, audio_feats)

        z_c_pool = z_c.mean(1)  # [B, fused_dim]
        z_s_pool = z_s.mean(1)  # [B, fused_dim]

        # ── Domain losses (ALL clips) ─────────────────────────────────────────
        d_logit_c = self.domain_head_c(self.grl_domain(z_c_pool)).squeeze(-1)
        loss_dadv = F.binary_cross_entropy_with_logits(d_logit_c, domain_labels)

        d_logit_s = self.domain_head_s(z_s_pool).squeeze(-1)
        loss_ddis = F.binary_cross_entropy_with_logits(d_logit_s, domain_labels)

        # ── SDA: cross-modal sync alignment on v_c vs a_c  (ALL clips) ───────
        # Mean-pool over time, project to shared SDA space, then Sinkhorn.
        v_c_sda = self.sda(v_c.mean(1))  # [B, sda_dim], L2-normalised
        a_c_sda = self.sda(a_c.mean(1))  # [B, sda_dim], L2-normalised
        loss_sda = _sinkhorn_divergence(
            v_c_sda, a_c_sda, eps=self.sda_eps, n_iter=self.sda_n_iter
        )

        # ── Task losses (AV1M only) ───────────────────────────────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            z_c_av = z_c[av1m_mask]
            z_s_av = z_s[av1m_mask]
            lbl_av = cls_labels[av1m_mask]

            score_causal = self._cls_score(self.causal_head, z_c_av)
            loss_task    = self._ce_loss(score_causal, lbl_av)

            # Spurious probe (monitor only — not in total loss)
            _s_spu = self._cls_score(self.spurious_head, z_s_av.detach())

            # Label adversarial on Z_s
            adv_score = self._cls_score(self.adv_head, self.grl_label(z_s_av))
            loss_adv  = self._ce_loss(adv_score, lbl_av)

            loss_orth = self._orth_loss(z_c_av, z_s_av)
        else:
            loss_task = loss_adv = loss_orth = \
                torch.tensor(0.0, device=video_feats.device, requires_grad=True)

        loss = (
            loss_task
            + self.lambda_adv  * loss_adv
            + self.lambda_dadv * loss_dadv
            + self.lambda_ddis * loss_ddis
            + self.lambda_orth * loss_orth
            + self.lambda_sda  * loss_sda
        )

        self.log("train_loss",      loss,      on_step=False, on_epoch=True)
        self.log("train_loss_task", loss_task, on_step=False, on_epoch=True)
        self.log("train_loss_adv",  loss_adv,  on_step=False, on_epoch=True)
        self.log("train_loss_dadv", loss_dadv, on_step=False, on_epoch=True)
        self.log("train_loss_ddis", loss_ddis, on_step=False, on_epoch=True)
        self.log("train_loss_orth", loss_orth, on_step=False, on_epoch=True)
        self.log("train_loss_sda",  loss_sda,  on_step=False, on_epoch=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            video_feats, audio_feats, labels, _, _ = batch
        else:
            video_feats, audio_feats, labels, _ = batch

        with torch.no_grad():
            s_c = self.forward(video_feats, audio_feats, mode="causal")
            s_s = self.forward(video_feats, audio_feats, mode="spurious")

        self._val_scores["causal"].append(s_c.cpu().numpy())
        self._val_scores["spurious"].append(s_s.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        labels = np.concatenate(self._val_labels)
        results = {}
        for mode in ("causal", "spurious"):
            scores = np.concatenate(self._val_scores[mode])
            try:
                auc = roc_auc_score(y_true=labels, y_score=scores)
            except ValueError:
                auc = float("nan")
            results[mode] = auc
            self.log(f"val_auc_{mode}", auc, prog_bar=True)

        print(
            f"\n[Epoch {self.current_epoch}] "
            f"val_auc  causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}",
            flush=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
