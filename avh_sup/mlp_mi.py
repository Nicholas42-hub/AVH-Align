"""
AVH-Align Mutual Information (MINE) Model
==========================================
Replaces the standard cross-entropy loss with a MINE-based Mutual Information
loss that explicitly encourages high MI between audio and visual representations
for real (aligned) pairs, and low MI for fake (misaligned) pairs.

Architecture
------------
visual_feats [T×1024], audio_feats [T×1024]
    │
    ├── visual_enc  : Linear(1024→512) → ReLU → LayerNorm → mean-pool → z_v [B×512]
    └── audio_enc   : Linear(1024→512) → ReLU → LayerNorm → mean-pool → z_a [B×512]
    │
    └── critic (statistics net T) : Linear(1024→512) → ReLU → Linear(512→1)
                                    takes concat(z_v, z_a) → scalar score

Training losses
---------------
  MINE lower bound (DV representation):
      MI(V;A) ≈ E_{P(V,A)}[T(v,a)] – log E_{P(V)P(A)}[exp(T(v,a))]
  where marginals are approximated by shuffling z_a within the batch.

  Supervised MI loss:
    L_mi_real   = –MI_lb(z_v_real, z_a_real)     # maximise MI for real (aligned) pairs
    L_cls       = BCE(critic_score, real_label)    # binary classification supervision
    L_total     = L_cls + lambda_mi * L_mi_real

  lambda_mi controls the strength of the MI regularisation.

Evaluation
----------
  predict_scores returns –T(z_v, z_a), so that higher score ↔ more likely fake.
  This is consistent with the existing roc_auc_score evaluation (y_true: 0=real, 1=fake).
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_encoder(in_dim: int = 1024, out_dim: int = 512) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


def _make_critic(in_dim: int = 1024, hidden_dim: int = 512) -> nn.Sequential:
    """Statistics network T(z_v, z_a) → scalar."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


# ── MINE loss ──────────────────────────────────────────────────────────────────

def mine_loss(critic: nn.Module, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative MINE lower bound for (z_v, z_a).

    Returns a scalar loss whose minimum corresponds to maximum MI(V;A).
    Gradient descent on this loss → maximise MI.

    Args:
        critic  : statistics network T: R^{2d} → R
        z_v     : visual embeddings   [N, d]
        z_a     : audio  embeddings   [N, d]

    Returns:
        scalar tensor  (-MI lower bound)
    """
    N = z_v.size(0)

    # Joint: matched pairs
    T_joint = critic(torch.cat([z_v, z_a], dim=-1))           # [N, 1]

    # Marginal: shuffle z_a to break audio-visual pairing
    idx = torch.randperm(N, device=z_a.device)
    T_marginal = critic(torch.cat([z_v, z_a[idx]], dim=-1))   # [N, 1]

    # DV lower bound: E_{P}[T] - log E_{Q}[exp(T)]
    mi_lb = T_joint.mean() - torch.log(torch.exp(T_marginal).mean().clamp(min=1e-8))

    return -mi_lb   # return *negative* so we minimise → maximise MI


# ── Lightning Module ───────────────────────────────────────────────────────────

class AVH_MI(L.LightningModule):
    """
    Audio-Visual Deepfake Detector using Mutual Information (MINE) loss.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        hp = config.get("model_hparams", {})
        feat_dim   = hp.get("feat_dim",   1024)
        proj_dim   = hp.get("proj_dim",   512)
        hidden_dim = hp.get("hidden_dim", 512)
        self.lambda_mi = hp.get("lambda_mi", 1.0)
        self.lr        = hp.get("lr", 1e-3)

        # Encoders
        self.visual_enc = _make_encoder(feat_dim, proj_dim)
        self.audio_enc  = _make_encoder(feat_dim, proj_dim)

        # MINE statistics / critic network
        self.critic = _make_critic(proj_dim * 2, hidden_dim)

        self.save_hyperparameters()

        # Buffers for epoch-level AUC
        self._val_scores: list = []
        self._val_labels: list = []

    # ── forward ────────────────────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """
        Encode variable-length feature sequences to fixed-size embeddings.

        Args:
            video_feats : [B, T, feat_dim]
            audio_feats : [B, T, feat_dim]
        Returns:
            z_v, z_a    : [B, proj_dim]
        """
        # Mean-pool over time, then project
        z_v = self.visual_enc(video_feats.mean(dim=1))   # [B, proj_dim]
        z_a = self.audio_enc(audio_feats.mean(dim=1))    # [B, proj_dim]
        return z_v, z_a

    def forward(self, input_feats):
        """Returns the raw critic score (alignment score)."""
        video_feats, audio_feats = input_feats
        z_v, z_a = self._encode(video_feats, audio_feats)
        score = self.critic(torch.cat([z_v, z_a], dim=-1)).squeeze(-1)  # [B]
        return score

    def predict_scores(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """
        Returns: –T(z_v, z_a)  (higher ↔ more likely fake / misaligned).
        Consistent with roc_auc_score where y_true 1 = fake.
        """
        score = self.forward((video_feats, audio_feats))
        return -score

    # ── shared step ────────────────────────────────────────────────────────────

    def _shared_step(self, batch, split: str = "train"):
        video_feats, audio_feats, labels, _ = batch
        labels = labels.to(self.device)

        z_v, z_a = self._encode(video_feats, audio_feats)

        # ── Classification loss ──────────────────────────────────────────────
        # critic score: high ↔ aligned ↔ real (label=0)
        score = self.critic(torch.cat([z_v, z_a], dim=-1)).squeeze(-1)   # [B]
        # BCE: target = 1 for real, 0 for fake  (i.e. 1 – original_label)
        bce_targets = (1.0 - labels.float())
        cls_loss = F.binary_cross_entropy_with_logits(score, bce_targets)

        # ── MINE MI loss on real pairs only ──────────────────────────────────
        real_mask = (labels == 0)
        mi_loss_val = torch.tensor(0.0, device=self.device)
        if real_mask.sum() > 1:
            mi_loss_val = mine_loss(self.critic, z_v[real_mask], z_a[real_mask])

        # ── Total loss ───────────────────────────────────────────────────────
        total_loss = cls_loss + self.lambda_mi * mi_loss_val

        self.log(f"{split}_loss",     total_loss,  on_epoch=True, prog_bar=True)
        self.log(f"{split}_cls_loss", cls_loss,    on_epoch=True)
        self.log(f"{split}_mi_loss",  mi_loss_val, on_epoch=True)

        return total_loss, score, labels

    # ── training / validation steps ────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def on_validation_epoch_start(self):
        self._val_scores = []
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        loss, score, labels = self._shared_step(batch, "val")
        # collect for epoch-level AUC
        # –score → higher = more fake, to align with AUC convention
        self._val_scores.append((-score).detach().cpu().numpy())
        self._val_labels.append(labels.detach().cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        all_scores = np.concatenate(self._val_scores)
        all_labels = np.concatenate(self._val_labels)
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(y_score=all_scores, y_true=all_labels)
            ap  = average_precision_score(y_score=all_scores, y_true=all_labels)
            self.log("val_auc", auc, prog_bar=True)
            self.log("val_ap",  ap)

    # ── optimiser ──────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
