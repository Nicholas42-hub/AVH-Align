"""
AVH-Align A43 — Causal + Domain Adversarial (Frozen Encoder)
=============================================================
Extends A2 (Z_c/Z_s causal disentanglement) with explicit domain
push-pull objective from A6, applied to the causal/spurious branches:

  Push: GRL(domain_head_c(Z_c_pool)) → domain info EXPELLED from Z_c
  Pull: domain_head_s(Z_s_pool)       → domain info CONCENTRATED in Z_s

Architecture
------------
visual_feats [T×1024], audio_feats [T×1024]   (frozen AV-HuBERT encoder)
    │
    ├── visual_proj_causal   → v_c [T×512]
    ├── visual_proj_spurious → v_s [T×512]
    ├── audio_proj_causal    → a_c [T×512]
    └── audio_proj_spurious  → a_s [T×512]
    │
    Z_c = concat(v_c, a_c)  [T×1024]
    Z_s = concat(v_s, a_s)  [T×1024]
    │
    ├── causal_head(Z_c)               → task classification (main)
    ├── spurious_head(Z_s.detach())    → probe only
    ├── adv_head(GRL(Z_s))             → label adversarial on Z_s  [A2]
    ├── domain_head_c(GRL(Z_c_pool))   → domain adversarial on Z_c [A6]
    └── domain_head_s(Z_s_pool)        → domain discriminative on Z_s [A6]

Training losses
---------------
  L = L_task
      + λ_adv   · L_adv_label   (GRL on Z_s, expel label from spurious)
      + λ_dadv  · L_dadv        (GRL on Z_c, expel domain from causal)
      + λ_ddis  · L_ddis        (Z_s attracts domain signal)
      + λ_orth  · L_orth        (Z_c ⊥ Z_s orthogonality)

Training data
-------------
  AV1M train  (domain=0, cls_label ≥ 0)
  FAVC val-real (domain=1, cls_label=-1 → task losses masked out)

Batch format
------------
  (video_feats, audio_feats, cls_labels, domain_labels, paths)
"""

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
    """Classification MLP: input → 512 → 256 → 128 → 1."""
    return nn.Sequential(
        nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
        nn.Linear(512, 256),       nn.LayerNorm(256), nn.ReLU(),
        nn.Linear(256, 128),       nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 1),
    )


def _make_domain_head(input_dim: int = 1024) -> nn.Sequential:
    """Lightweight domain MLP: input → 256 → 64 → 1."""
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 64),        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class AVH_Causal_A43(L.LightningModule):
    """
    A43: frozen-encoder Z_c/Z_s causal model with domain adversarial training.

    Config keys (under model_hparams):
        feat_dim      : int   (default 1024)
        proj_dim      : int   (default 512)
        lambda_adv    : float adversarial label loss on Z_s (default 10.0)
        grl_alpha     : float GRL alpha for label adversarial (default 5.0)
        lambda_dadv   : float adversarial domain loss on Z_c (default 1.0)
        grl_domain_alpha : float GRL alpha for domain adversarial (default 1.0)
        lambda_ddis   : float discriminative domain loss on Z_s (default 1.0)
        lambda_orth   : float orthogonality loss (default 0.1)
        lr            : float (default 1e-3)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})
        feat_dim          = hp.get("feat_dim",          1024)
        proj_dim          = hp.get("proj_dim",           512)
        self.lambda_adv        = float(hp.get("lambda_adv",        10.0))
        grl_alpha              = float(hp.get("grl_alpha",          5.0))
        self.lambda_dadv       = float(hp.get("lambda_dadv",        1.0))
        grl_domain_alpha       = float(hp.get("grl_domain_alpha",   1.0))
        self.lambda_ddis       = float(hp.get("lambda_ddis",        1.0))
        self.lambda_orth       = float(hp.get("lambda_orth",        0.1))
        self.lr                = float(hp.get("lr",                 1e-3))

        fused_dim = proj_dim * 2  # 1024 after concat(visual, audio)

        # ── Projection heads ──────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.causal_head   = _make_cls_head(fused_dim)
        self.spurious_head = _make_cls_head(fused_dim)  # probe only (detached)

        # ── Label adversarial on Z_s (A2) ─────────────────────────────────────
        self.adv_head = _make_cls_head(fused_dim)
        self.grl_label = GradientReversal(alpha=grl_alpha)

        # ── Domain adversarial on Z_c (A6-style) ──────────────────────────────
        self.domain_head_c = _make_domain_head(fused_dim)
        self.grl_domain    = GradientReversal(alpha=grl_domain_alpha)

        # ── Domain discriminative on Z_s (A6-style) ───────────────────────────
        self.domain_head_s = _make_domain_head(fused_dim)

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)
        return torch.cat([v_c, a_c], dim=-1), torch.cat([v_s, a_s], dim=-1)

    @staticmethod
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        """MLP head → LogSumExp over time → clip score [B]."""
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
        z_c, z_s = self._encode(video_feats, audio_feats)
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
        cls_labels    = cls_labels.long()     # [B], -1 for FAVC
        domain_labels = domain_labels.float() # [B], 0=AV1M 1=FAVC

        z_c, z_s = self._encode(video_feats, audio_feats)

        # Pool over time for domain heads
        z_c_pool = z_c.mean(1)   # [B, fused_dim]
        z_s_pool = z_s.mean(1)   # [B, fused_dim]

        # ── Domain losses (ALL clips — both AV1M and FAVC) ───────────────────
        # Adversarial on Z_c: GRL reverses gradients so Z_c forgets domain
        d_logit_c = self.domain_head_c(self.grl_domain(z_c_pool)).squeeze(-1)
        loss_dadv = F.binary_cross_entropy_with_logits(d_logit_c, domain_labels)

        # Discriminative on Z_s: pull domain signal into Z_s
        d_logit_s = self.domain_head_s(z_s_pool).squeeze(-1)
        loss_ddis = F.binary_cross_entropy_with_logits(d_logit_s, domain_labels)

        # ── Task losses (AV1M only, cls_label >= 0) ───────────────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            z_c_av  = z_c[av1m_mask]
            z_s_av  = z_s[av1m_mask]
            lbl_av  = cls_labels[av1m_mask]

            # Classification: only from Z_c (use_sg style — Z_s gets NO task grad)
            score_causal   = self._cls_score(self.causal_head, z_c_av)
            loss_task      = self._ce_loss(score_causal, lbl_av)

            # Probe: spurious head on detached Z_s (monitor only)
            score_spurious = self._cls_score(self.spurious_head, z_s_av.detach())
            _loss_spu_probe = self._ce_loss(score_spurious, lbl_av)  # not in total loss

            # Label adversarial on Z_s: GRL keeps Z_s task-irrelevant
            adv_score = self._cls_score(self.adv_head, self.grl_label(z_s_av))
            loss_adv  = self._ce_loss(adv_score, lbl_av)

            # Orthogonality
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
        )

        self.log("train_loss",       loss,       on_step=False, on_epoch=True)
        self.log("train_loss_task",  loss_task,  on_step=False, on_epoch=True)
        self.log("train_loss_adv",   loss_adv,   on_step=False, on_epoch=True)
        self.log("train_loss_dadv",  loss_dadv,  on_step=False, on_epoch=True)
        self.log("train_loss_ddis",  loss_ddis,  on_step=False, on_epoch=True)
        self.log("train_loss_orth",  loss_orth,  on_step=False, on_epoch=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        # Val batches are AV1M 4-tuple (no domain label)
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

        self.log("val_auc_causal", results["causal"], prog_bar=True)
        print(
            f"\n[Epoch {self.current_epoch}] "
            f"val_auc  causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}",
            flush=True,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
