"""
CORAL-UDA Baseline
==================

CORAL (Correlation ALignment, Sun & Saenko 2016) applied to frozen
AV-HuBERT features — no factorization.  Aligns second-order statistics
(covariance) of projected source and target features.

Architecture
------------
  z_v, z_a  [B, T, 1024]
      │   mean-pool over T
      ├─ z_v_pool, z_a_pool  [B, 1024]
      │   concat
      ├─ z_pool  [B, 2048]
      │
      ├─ proj → coral_dim  [B, coral_dim]   ← CORAL alignment in this space
      │
      ├─ task_mlp: coral_dim → … → 1  [B]   (fake score, BCE on source only)
      │
      └─ CORAL loss: ||C_src - C_tgt||_F^2 / (4 * coral_dim^2)

Batch construction
------------------
  A collate_fn in the DataLoader pre-pools variable-length clips to fixed
  [1024] per modality, enabling standard batch_size > 1.  Source and target
  samples coexist in each batch (domain_label 0/1); the CORAL loss is
  computed on the projected pool representation, splitting by domain label.

  Minimum batch requirement: ≥ 2 source AND ≥ 2 target samples to form
  non-trivial covariances.  Batches with a missing domain are skipped for
  the CORAL term.

Losses
------
  L_task  = BCE(score, y)       [source clips, cls_label >= 0]
  L_coral = CORAL(Z_src, Z_tgt) [when both domains present in batch]
  L       = L_task + λ_coral * L_coral

Training data (5-tuple batches, mean-pooled)
---------------------------------------------
  (video [B, 1024], audio [B, 1024],
   cls_label [B], domain_label [B], paths)

Eval interface
--------------
  model.predict_scores(video, audio)
    video, audio: [B, T, 1024]  OR  [B, 1024]
    Returns [B] fake scores (compatible with run_baseline in eval_favc.py).
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ─── CORAL loss ──────────────────────────────────────────────────────────────

def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Batch CORAL loss (Sun & Saenko 2016).
    source, target: [N, d] — already projected representations.
    Returns scalar loss.
    """
    ns, d = source.shape
    nt    = target.shape[0]

    # Centre each set
    cs = source - source.mean(dim=0, keepdim=True)
    ct = target - target.mean(dim=0, keepdim=True)

    # Unbiased sample covariances (/ (n-1) where n>1)
    cov_s = (cs.T @ cs) / (ns - 1) if ns > 1 else cs.T @ cs
    cov_t = (ct.T @ ct) / (nt - 1) if nt > 1 else ct.T @ ct

    diff = cov_s - cov_t
    loss = (diff * diff).sum() / (4.0 * d * d)
    return loss


# ─── Building blocks ─────────────────────────────────────────────────────────

def _make_task_mlp(in_dim: int) -> nn.Sequential:
    """Same depth/width as two-way baseline."""
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


# ─── Main model ──────────────────────────────────────────────────────────────

class CORAL_UDA(L.LightningModule):
    """CORAL-UDA baseline on frozen AV-HuBERT features."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        hp = config["model_hparams"]

        feat_dim   = hp.get("feat_dim",   1024)
        fused_dim  = 2 * feat_dim          # cat(z_v_pool, z_a_pool)
        coral_dim  = hp.get("coral_dim",  256)

        self.lambda_coral = hp.get("lambda_coral", 1.0)
        self.lr           = hp.get("lr", 1e-3)

        # Projection to lower-dim alignment space
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, coral_dim),
            nn.LayerNorm(coral_dim),
            nn.ReLU(),
        )
        self.task_mlp = _make_task_mlp(coral_dim)

        self._val_scores: list[float] = []
        self._val_labels: list[int]   = []

    # ── Forward ──────────────────────────────────────────────────────────────

    def _pool_and_encode(self,
                         video: torch.Tensor,
                         audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Accepts either:
          • temporal inputs  [B, T, 1024]  → mean-pool over T
          • pre-pooled       [B, 1024]
        Returns:
          z_proj  [B, coral_dim]
          score   [B]
        """
        if video.dim() == 3:
            video = video.mean(dim=1)   # [B, 1024]
            audio = audio.mean(dim=1)
        z = torch.cat([video, audio], dim=-1)  # [B, 2048]
        z_proj = self.proj(z)                  # [B, coral_dim]
        score  = self.task_mlp(z_proj).squeeze(-1)  # [B]
        return z_proj, score

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video, audio, cls_labels, domain_labels, _ = batch
        z_proj, score = self._pool_and_encode(video, audio)

        # Task loss — source-only
        src_mask = cls_labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        if src_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                score[src_mask], cls_labels[src_mask].float())
        self.log("train_task_loss", loss, prog_bar=False)

        # CORAL loss — needs ≥ 2 samples per domain in this batch
        tgt_mask = domain_labels > 0.5
        if src_mask.sum() >= 2 and tgt_mask.sum() >= 2:
            c_loss = coral_loss(z_proj[src_mask], z_proj[tgt_mask])
            self.log("train_coral_loss", c_loss, prog_bar=False)
            loss = loss + self.lambda_coral * c_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        video, audio, cls_labels, domain_labels, _ = batch
        _, score = self._pool_and_encode(video, audio)

        src_mask = cls_labels >= 0
        if src_mask.any():
            val_loss = F.binary_cross_entropy_with_logits(
                score[src_mask], cls_labels[src_mask].float())
            self.log("val_loss", val_loss, on_epoch=True, on_step=False)
            for s, l in zip(score[src_mask].cpu().tolist(),
                             cls_labels[src_mask].cpu().tolist()):
                self._val_scores.append(float(s))
                self._val_labels.append(int(l))

    def on_validation_epoch_end(self):
        if len(set(self._val_labels)) == 2 and len(self._val_scores) > 0:
            auc = roc_auc_score(self._val_labels, self._val_scores)
            self.log("val_auc", auc, prog_bar=True)
        self._val_scores.clear()
        self._val_labels.clear()

    # ── Eval interface ────────────────────────────────────────────────────────

    def predict_scores(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """Returns [B] fake scores. Accepts temporal [B,T,1024] or pooled [B,1024]."""
        _, score = self._pool_and_encode(video, audio)
        return score

    # ── Optimizer ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
