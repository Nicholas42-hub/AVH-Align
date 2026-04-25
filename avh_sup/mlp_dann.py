"""
DANN-UDA Baseline
=================

Standard Domain-Adversarial Neural Network (Ganin et al., 2016) applied
directly to frozen AV-HuBERT features — no factorization.

Architecture
------------
  z_v, z_a  [B, T, 1024]
      │
      ├─ proj_v: 1024 → proj_dim        [B, T, proj_dim]
      ├─ proj_a: 1024 → proj_dim        [B, T, proj_dim]
      │
      ├─ fuse:  cat(proj_v, proj_a)     [B, T, 2*proj_dim]
      │
      ├─ task_mlp: 2*proj_dim → … → 1  [B, T, 1]
      │   → logsumexp(T) → [B]          (fake score)
      │
      └─ domain head (GRL):
            mean-pool fused [B, 2*proj_dim]
            → GRL(α)
            → dom_mlp → 1               (domain logit)

Losses
------
  L_task = BCE(task_score, y)             [AV1M clips only, cls_label >= 0]
  L_adv  = BCE(dom_logit, domain_label)   [all clips]
  L      = L_task + λ_adv * L_adv

Training data (5-tuple batches)
--------------------------------
  (video [B,T,1024], audio [B,T,1024],
   cls_label [B]   (int, -1 = skip task loss for target clips),
   domain_label [B] (float, 0=source, 1=target),
   paths)

Eval interface
--------------
  model.predict_scores(video, audio) → [B] fake scores
  (compatible with run_baseline in eval_favc.py)
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ─── Gradient Reversal Layer ─────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GRLFunction.apply(x, alpha)


# ─── Building blocks ─────────────────────────────────────────────────────────

def _make_proj(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


def _make_task_mlp(in_dim: int) -> nn.Sequential:
    """4-layer task MLP matching two-way baseline capacity."""
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def _make_domain_head(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ─── Main model ──────────────────────────────────────────────────────────────

class DANN_UDA(L.LightningModule):
    """Standard DANN-UDA baseline on frozen AV-HuBERT features."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        hp = config["model_hparams"]

        feat_dim  = hp.get("feat_dim",  1024)
        proj_dim  = hp.get("proj_dim",  512)
        fused_dim = 2 * proj_dim

        self.lambda_adv = hp.get("lambda_adv", 1.0)
        self.grl_alpha  = hp.get("grl_alpha",  1.0)
        self.lr         = hp.get("lr", 1e-3)

        self.proj_v    = _make_proj(feat_dim, proj_dim)
        self.proj_a    = _make_proj(feat_dim, proj_dim)
        self.task_mlp  = _make_task_mlp(fused_dim)
        self.domain_head = _make_domain_head(fused_dim)

        self._val_scores: list[float] = []
        self._val_labels: list[int]   = []

    # ── Forward ──────────────────────────────────────────────────────────────

    def _encode(self, video: torch.Tensor, audio: torch.Tensor):
        """
        video, audio: [B, T, 1024]
        Returns:
          task_logit: [B]  (logsumexp of frame-level predictions)
          fused_pool: [B, 2*proj_dim]  (mean-pooled for domain head)
        """
        pv = self.proj_v(video)          # [B, T, proj_dim]
        pa = self.proj_a(audio)          # [B, T, proj_dim]
        fused = torch.cat([pv, pa], dim=-1)  # [B, T, fused_dim]

        frame_logits = self.task_mlp(fused).squeeze(-1)  # [B, T]
        task_logit   = torch.logsumexp(frame_logits, dim=-1)  # [B]

        fused_pool = fused.mean(dim=1)  # [B, fused_dim]
        return task_logit, fused_pool

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video, audio, cls_labels, domain_labels, _ = batch
        task_logit, fused_pool = self._encode(video, audio)

        # Task loss — source-only (cls_label >= 0)
        src_mask = cls_labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        if src_mask.any():
            src_logit = task_logit[src_mask]
            src_label = cls_labels[src_mask].float()
            loss = F.binary_cross_entropy_with_logits(src_logit, src_label)
        self.log("train_task_loss", loss, prog_bar=False)

        # Adversarial domain loss — all clips
        dom_input = grad_reverse(fused_pool, self.grl_alpha)
        dom_logit = self.domain_head(dom_input).squeeze(-1)  # [B]
        adv_loss  = F.binary_cross_entropy_with_logits(
            dom_logit, domain_labels.float())
        self.log("train_adv_loss", adv_loss, prog_bar=False)

        total = loss + self.lambda_adv * adv_loss
        self.log("train_loss", total, prog_bar=True)
        return total

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        video, audio, cls_labels, domain_labels, _ = batch
        task_logit, fused_pool = self._encode(video, audio)

        # Task val loss (source only)
        src_mask = cls_labels >= 0
        if src_mask.any():
            val_loss = F.binary_cross_entropy_with_logits(
                task_logit[src_mask], cls_labels[src_mask].float())
            self.log("val_loss", val_loss, on_epoch=True, on_step=False)

            for score, lbl in zip(task_logit[src_mask].cpu().tolist(),
                                   cls_labels[src_mask].cpu().tolist()):
                self._val_scores.append(float(score))
                self._val_labels.append(int(lbl))

    def on_validation_epoch_end(self):
        if len(set(self._val_labels)) == 2 and len(self._val_scores) > 0:
            auc = roc_auc_score(self._val_labels, self._val_scores)
            self.log("val_auc", auc, prog_bar=True)
        self._val_scores.clear()
        self._val_labels.clear()

    # ── Eval interface ────────────────────────────────────────────────────────

    def predict_scores(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Returns [B] fake scores (higher = more likely fake)."""
        task_logit, _ = self._encode(video, audio)
        return task_logit

    # ── Optimizer ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
