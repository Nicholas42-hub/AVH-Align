"""
AVH-Align Causal Model
======================
Extends the base AVH-Align supervised head with explicit causal / spurious
representation disentanglement.

Architecture
------------
visual_feats [T×1024], audio_feats [T×1024]
    │
    ├── visual_proj_causal   → v_c [T×512]
    ├── visual_proj_spurious → v_s [T×512]
    ├── audio_proj_causal    → a_c [T×512]
    └── audio_proj_spurious  → a_s [T×512]
    │
    causal_repr   = concat(v_c, a_c)  [T×1024]  ← A/V sync signal
    spurious_repr = concat(v_s, a_s)  [T×1024]  ← modality-specific residuals
    │
    ├── full_head(causal_repr + spurious_repr) → LogSumExp → clip score
    ├── causal_head(causal_repr)               → LogSumExp → clip score
    └── spurious_head(spurious_repr)           → LogSumExp → clip score

Training losses
---------------
  L = CE(full) + λ_causal * CE(causal) + λ_spurious * CE(spurious)
      + λ_orth * orth_loss(causal_repr, spurious_repr)

  orth_loss: mean squared cosine similarity per frame — forces the two streams
             to encode genuinely different information.

Evaluation
----------
  v_full     : full_head    with both streams          → should be highest AUC
  v_causal   : causal_head  with causal stream only    → hypothesis: ≈ v_full
  v_spurious : spurious_head with spurious stream only → hypothesis: ≈ 0.5
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


def _make_mlp(input_dim: int = 1024) -> nn.Sequential:
    """Shared MLP architecture: 1024→512→256→128→1 (matches base AVH_Sup)."""
    return nn.Sequential(
        nn.Linear(input_dim, 512),
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


class AVH_Causal(L.LightningModule):
    """
    Causal-disentangled supervised head for AVH-Align.

    Config keys (under model_hparams):
        feat_dim        : int   (default 1024) — AV-HuBERT output dimension
        proj_dim        : int   (default 512)  — per-modality projection size
        lambda_causal   : float (default 1.0)  — weight for causal cls loss
        lambda_spurious : float (default 0.5)  — weight for spurious cls loss
        lambda_orth     : float (default 0.1)  — weight for orthogonality loss
        lr              : float (default 1e-3)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})
        feat_dim        = hp.get("feat_dim",        1024)
        proj_dim        = hp.get("proj_dim",         512)
        self.lambda_causal   = hp.get("lambda_causal",   1.0)
        self.lambda_spurious = hp.get("lambda_spurious",  0.5)
        self.lambda_orth     = hp.get("lambda_orth",      0.1)
        self.lr              = hp.get("lr",              1e-3)

        fused_dim = proj_dim * 2  # concat of visual + audio projections = 1024

        # ── Projections ───────────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── Classification heads ───────────────────────────────────────────────
        # full_head: additive fusion keeps same dimension as either stream alone
        self.full_head     = _make_mlp(fused_dim)
        self.causal_head   = _make_mlp(fused_dim)
        self.spurious_head = _make_mlp(fused_dim)

        # ── Validation accumulators ────────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Representation helpers ─────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """Return (causal_repr, spurious_repr) each [B×T×fused_dim]."""
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)

        causal_repr   = torch.cat([v_c, a_c], dim=-1)   # [B×T×1024]
        spurious_repr = torch.cat([v_s, a_s], dim=-1)   # [B×T×1024]
        return causal_repr, spurious_repr

    @staticmethod
    def _orth_loss(causal_repr: torch.Tensor, spurious_repr: torch.Tensor) -> torch.Tensor:
        """
        Orthogonality loss: mean squared cosine similarity per frame token.
        Forces causal and spurious streams to encode different information.
        """
        B, T, D = causal_repr.shape
        c = F.normalize(causal_repr.reshape(-1, D), dim=-1)   # [B*T × D]
        s = F.normalize(spurious_repr.reshape(-1, D), dim=-1) # [B*T × D]
        cos_sim = (c * s).sum(dim=-1)  # [B*T]
        return (cos_sim ** 2).mean()

    @staticmethod
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        """Apply MLP head then LogSumExp pooling over time → clip score."""
        per_frame = head(repr_)[..., 0]   # [B×T]
        return torch.logsumexp(per_frame, dim=-1)  # [B]

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)  # [B×2]
        return F.cross_entropy(logits, labels)

    # ── Forward (used at inference) ────────────────────────────────────────────

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mode: str = "full",
    ) -> torch.Tensor:
        """
        Args:
            video_feats : [B×T×1024]
            audio_feats : [B×T×1024]
            mode        : "full" | "causal" | "spurious"
        Returns:
            clip score  : [B]
        """
        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)

        if mode == "full":
            return self._cls_score(self.full_head, causal_repr + spurious_repr)
        elif mode == "causal":
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, spurious_repr)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'full', 'causal', or 'spurious'.")

    def predict_scores(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mode: str = "full",
    ) -> torch.Tensor:
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        labels = labels.long()

        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)

        score_full     = self._cls_score(self.full_head,     causal_repr + spurious_repr)
        score_causal   = self._cls_score(self.causal_head,   causal_repr)
        score_spurious = self._cls_score(self.spurious_head, spurious_repr)

        loss_cls      = self._ce_loss(score_full,     labels)
        loss_causal   = self._ce_loss(score_causal,   labels)
        loss_spurious = self._ce_loss(score_spurious, labels)
        loss_orth     = self._orth_loss(causal_repr, spurious_repr)

        loss = (
            loss_cls
            + self.lambda_causal   * loss_causal
            + self.lambda_spurious * loss_spurious
            + self.lambda_orth     * loss_orth
        )

        self.log("train_loss",         loss,         on_step=False, on_epoch=True)
        self.log("train_loss_cls",      loss_cls,     on_step=False, on_epoch=True)
        self.log("train_loss_causal",   loss_causal,  on_step=False, on_epoch=True)
        self.log("train_loss_spurious", loss_spurious,on_step=False, on_epoch=True)
        self.log("train_loss_orth",     loss_orth,    on_step=False, on_epoch=True)
        return loss

    # ── Validation ─────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch

        with torch.no_grad():
            s_full     = self.forward(video_feats, audio_feats, mode="full")
            s_causal   = self.forward(video_feats, audio_feats, mode="causal")
            s_spurious = self.forward(video_feats, audio_feats, mode="spurious")

        self._val_scores["full"].append(s_full.cpu().numpy())
        self._val_scores["causal"].append(s_causal.cpu().numpy())
        self._val_scores["spurious"].append(s_spurious.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        labels = np.concatenate(self._val_labels)

        results = {}
        for mode in ("full", "causal", "spurious"):
            scores = np.concatenate(self._val_scores[mode])
            try:
                auc = roc_auc_score(y_true=labels, y_score=scores)
            except ValueError:
                auc = float("nan")
            results[mode] = auc
            self.log(f"val_auc_{mode}", auc, prog_bar=(mode != "spurious"))

        # Primary monitor metric — causal AUC (core hypothesis)
        self.log("val_auc_causal", results["causal"], prog_bar=True)

        print(
            f"\n[Epoch {self.current_epoch}] "
            f"AUC  full={results['full']:.4f}  "
            f"causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}"
        )

    # ── Optimizer ──────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
