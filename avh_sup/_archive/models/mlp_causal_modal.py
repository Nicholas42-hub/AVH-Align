"""
AVH-Align Causal Model with Modal Balance Constraints
======================================================
Extends the ablation causal model (A2 base: adversarial + stop-gradient) with
positive constraints on the causal features to fix the audio-dominant problem.

Problem
-------
Previous experiments (A0–A3) show causal features detect audio deepfake well
but perform poorly on visual deepfake.  The causal branch is effectively
audio-dominant.

Three complementary modal balance constraints
---------------------------------------------
Branch 1 - Per-modality classification loss (use_modal_cls)
    Each modality's causal projection is individually supervised:
        modal_head_visual(v_c) → CE(labels)   ← learns to detect visual deepfake
        modal_head_audio(a_c)  → CE(labels)   ← learns to detect audio deepfake
    This applies a positive gradient directly to both projectors so neither
    modality can free-ride on the other.

    Branch 1b - Modal balance penalty (use_modal_balance, requires use_modal_cls)
    Additionally penalises the squared difference between per-modality CE losses:
        L_balance = (CE_audio - CE_visual)²
    Forces both modalities to contribute equally to the causal branch.
    Without this, audio can dominate even when B1 heads are present because
    gradients naturally flow where loss descends fastest.

Branch 2 - Cross-modal alignment loss (use_cross_modal)
    Minimises mean square cosine distance between v_c and a_c per frame:
        L_cross = 1 - mean_cosine_similarity(v_c, a_c)
    Forces the causal features from both modalities to encode the same
    semantic information (genuine audio-visual causal dependency).

Branch 3 - Modal discriminator adversarial on causal features (use_modal_disc)
    A small discriminator is trained to predict whether a token came from
    v_c (0) or a_c (1).  A GRL between the projectors and the discriminator
    reverses the gradient → forcing v_c and a_c to become modality-agnostic.
    This prevents the causal encoder from collapsing to a single modality.

Combined framework
------------------
  Input (video_feats [B×T×1024], audio_feats [B×T×1024])
      │
      ├── visual_proj_causal   → v_c [B×T×512]
      ├── visual_proj_spurious → v_s [B×T×512]
      ├── audio_proj_causal    → a_c [B×T×512]
      └── audio_proj_spurious  → a_s [B×T×512]
      │
      causal_repr   = concat(v_c, a_c)  [B×T×1024]
      spurious_repr = concat(v_s, a_s)  [B×T×1024]
      │
      ├── causal_head(Z_c)                      → CE(labels)   main cls
      ├── [Branch 1] modal_head_visual(v_c)         → CE(labels)   visual cls
      ├── [Branch 1] modal_head_audio(a_c)          → CE(labels)   audio cls
      ├── [Branch 1b] (CE_audio - CE_visual)^2      → L_balance   modal balance
      ├── [Branch 2] cosine_align(v_c, a_c)         → L_cross      alignment
      ├── [Branch 3] modal_disc(GRL_modal(v_c||a_c))→ BCE(modality)adversarial
      ├── adv_head(GRL(Z_s))                    → CE(labels)   spurious adv
      └── spurious_head(Z_s.detach())           → CE(labels)   probe only

Config flags (under config['modal_balance'])
--------------------------------------------
  use_modal_cls    : bool  - Branch 1 per-modality classification loss
  use_modal_balance: bool  - Branch 1b modal balance penalty (requires use_modal_cls)
  use_cross_modal  : bool  - Branch 2 cross-modal alignment loss
  use_modal_disc   : bool  - Branch 3 modal discriminator adversarial

Hyperparameters (under config['model_hparams'])
-----------------------------------------------
  feat_dim           : int   (default 1024)
  proj_dim           : int   (default 512)
  lambda_orth        : float (default 0.5)
  lambda_adv         : float (default 10.0)  weight for spurious adversarial
  lambda_modal_cls     : float (default 1.0)   weight for Branch 1
  lambda_modal_balance : float (default 1.0)   weight for Branch 1b balance penalty
  lambda_cross_modal   : float (default 0.5)   weight for Branch 2
  lambda_modal_disc    : float (default 1.0)   weight for Branch 3
  grl_alpha          : float (default 5.0)   GRL scale for spurious adversarial
  grl_modal_alpha    : float (default 1.0)   GRL scale for modal discriminator
  lr                 : float (default 1e-3)

Evaluation modes (same interface as AVH_Causal_Ablation)
---------------------------------------------------------
  "causal"   — causal_head(Z_c)       primary inference head
  "spurious" — spurious_head(Z_s)     probe head (should be ~0.5)
  "full"     — same as "causal" (Z_s  stop-gradient throughout)
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


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


# ── MLP factories ──────────────────────────────────────────────────────────────

def _make_mlp(input_dim: int = 1024) -> nn.Sequential:
    """4-layer MLP: input→512→256→128→1 (matches ablation base)."""
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


def _make_half_mlp(input_dim: int = 512) -> nn.Sequential:
    """Lighter 3-layer MLP for single-modality heads: input→256→128→1."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


def _make_modal_disc(input_dim: int = 512) -> nn.Sequential:
    """Modal discriminator MLP: input→256→1 (binary: audio vs visual)."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


class AVH_Causal_Modal(L.LightningModule):
    """
    Causal model with modal balance constraints.

    Inherits the A2 base (adversarial + stop-gradient) and adds three optional
    modal balance losses to address causal feature audio-dominance.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp  = config.get("model_hparams", {})
        mb  = config.get("modal_balance", {})

        feat_dim  = hp.get("feat_dim",  1024)
        proj_dim  = hp.get("proj_dim",   512)
        fused_dim = proj_dim * 2          # concat(v_c, a_c) = 1024

        # ── Modal balance flags ─────────────────────────────────────────────────
        self.use_modal_cls     = bool(mb.get("use_modal_cls",     False))  # Branch 1
        self.use_modal_balance = bool(mb.get("use_modal_balance", False))  # Branch 1b
        self.use_cross_modal   = bool(mb.get("use_cross_modal",   False))  # Branch 2
        self.use_modal_disc    = bool(mb.get("use_modal_disc",    False))  # Branch 3

        # ── Loss weights ────────────────────────────────────────────────────────
        self.lambda_orth          = hp.get("lambda_orth",           0.5)
        self.lambda_adv           = hp.get("lambda_adv",           10.0)
        self.lambda_modal_cls     = hp.get("lambda_modal_cls",      1.0)
        self.lambda_modal_balance = hp.get("lambda_modal_balance",  1.0)
        self.lambda_cross_modal   = hp.get("lambda_cross_modal",    0.5)
        self.lambda_modal_disc    = hp.get("lambda_modal_disc",     1.0)
        grl_alpha               = hp.get("grl_alpha",           5.0)
        grl_modal_alpha         = hp.get("grl_modal_alpha",     1.0)
        self.lr                 = hp.get("lr",                  1e-3)

        # ── Shared projections ──────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── Main classification heads ───────────────────────────────────────────
        # causal_head: primary — operates on Z_c = concat(v_c, a_c)
        # spurious_head: diagnostic probe only (encoder always detached)
        self.causal_head   = _make_mlp(fused_dim)
        self.spurious_head = _make_mlp(fused_dim)

        # ── Spurious adversarial head + GRL (A2 base) ──────────────────────────
        # adv_head tries to predict labels from Z_s;
        # GRL reverses gradients → forces Z_s non-informative for labels.
        self.adv_head = _make_mlp(fused_dim)
        self.grl      = GradientReversal(alpha=grl_alpha)

        # ── Branch 1: Per-modality classification heads ────────────────────────
        # Operate on individual proj_dim features before concatenation.
        # Forces each projector to independently detect deepfakes.
        if self.use_modal_cls:
            self.modal_head_visual = _make_half_mlp(proj_dim)  # v_c → score
            self.modal_head_audio  = _make_half_mlp(proj_dim)  # a_c → score

        # ── Branch 3: Modal discriminator + GRL ─────────────────────────────────
        # Trained on tokens pooled from v_c and a_c with binary labels.
        # GRL reverses gradient to projectors → v_c and a_c become indistinguishable.
        if self.use_modal_disc:
            self.modal_disc     = _make_modal_disc(proj_dim)
            self.grl_modal      = GradientReversal(alpha=grl_modal_alpha)

        # ── Validation accumulators ─────────────────────────────────────────────
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    # ── Representation helpers ─────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """
        Returns (v_c, a_c, v_s, a_s, causal_repr, spurious_repr).
          v_c, a_c, v_s, a_s  : [B×T×proj_dim]
          causal_repr          : [B×T×fused_dim]  = concat(v_c, a_c)
          spurious_repr        : [B×T×fused_dim]  = concat(v_s, a_s)
        """
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)
        causal_repr   = torch.cat([v_c, a_c], dim=-1)
        spurious_repr = torch.cat([v_s, a_s], dim=-1)
        return v_c, a_c, v_s, a_s, causal_repr, spurious_repr

    @staticmethod
    def _orth_loss(causal_repr: torch.Tensor, spurious_repr: torch.Tensor) -> torch.Tensor:
        """Mean squared cosine similarity — forces Z_c ⊥ Z_s."""
        B, T, D = causal_repr.shape
        c = F.normalize(causal_repr.reshape(-1, D),   dim=-1)
        s = F.normalize(spurious_repr.reshape(-1, D), dim=-1)
        return ((c * s).sum(dim=-1) ** 2).mean()

    @staticmethod
    def _cross_modal_loss(
        v_c: torch.Tensor,
        a_c: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Branch 2: Label-conditioned cross-modal alignment loss.

        Only aligns v_c and a_c for REAL samples (label == 0).
          - Real clips: both modalities are genuine → causal projections should
            encode the same authentic AV content → align them.
          - Fake clips: manipulation may exist in only one modality → forcing
            unconditional alignment would destroy modality-specific fake signal.

        L_cross = mean(1 - cosine_similarity(v_c_t, a_c_t))  over real frames
        Returns 0 if no real samples in the batch.
        """
        real_mask = (labels == 0)   # [B]
        if not real_mask.any():
            return torch.tensor(0.0, device=v_c.device)

        v_real = v_c[real_mask]   # [B_real × T × D]
        a_real = a_c[real_mask]
        B_r, T, D = v_real.shape

        v_norm = F.normalize(v_real.reshape(-1, D), dim=-1)   # [B_real*T × D]
        a_norm = F.normalize(a_real.reshape(-1, D), dim=-1)
        cos_sim = (v_norm * a_norm).sum(dim=-1)                # [B_real*T]
        return (1.0 - cos_sim).mean()

    @staticmethod
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        """MLP head → LogSumExp pooling over time → clip score [B]."""
        per_frame = head(repr_)[..., 0]   # [B×T]
        return torch.logsumexp(per_frame, dim=-1)

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    def _modal_disc_loss(
        self,
        v_c: torch.Tensor,
        a_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Branch 3: Modal discriminator adversarial loss.

        Concatenate all v_c and a_c tokens along the batch dimension, assign
        binary modality labels (0=visual, 1=audio), then:
          1. Apply GRL to v_c tokens and a_c tokens before the discriminator.
          2. Discriminator learns to predict modality → BCE loss.
          3. GRL reverses gradient into the projectors → they cannot be told apart.

        v_c, a_c : [B×T×proj_dim]
        """
        B, T, D = v_c.shape

        # Apply GRL to individual modality projections before discrimination
        v_c_grl = self.grl_modal(v_c.reshape(-1, D))   # [B*T × D]
        a_c_grl = self.grl_modal(a_c.reshape(-1, D))   # [B*T × D]

        tokens = torch.cat([v_c_grl, a_c_grl], dim=0)  # [2*B*T × D]
        modal_labels = torch.cat([
            torch.zeros(B * T, device=v_c.device),
            torch.ones( B * T, device=v_c.device),
        ], dim=0)  # [2*B*T]

        logits = self.modal_disc(tokens).squeeze(-1)    # [2*B*T]
        return F.binary_cross_entropy_with_logits(logits, modal_labels)

    # ── Forward (inference) ────────────────────────────────────────────────────

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mode: str = "causal",
    ) -> torch.Tensor:
        v_c, a_c, v_s, a_s, causal_repr, spurious_repr = self._encode(
            video_feats, audio_feats
        )
        if mode in ("causal", "full"):
            # Stop-gradient throughout: Z_c only, same as A2
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, spurious_repr)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'causal', 'full', or 'spurious'.")

    def predict_scores(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        mode: str = "causal",
    ) -> torch.Tensor:
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        labels = labels.long()

        v_c, a_c, v_s, a_s, causal_repr, spurious_repr = self._encode(
            video_feats, audio_feats
        )

        # ── A2 base: classification restricted to Z_c only (stop-gradient) ──────
        # Z_s receives gradient ONLY from the spurious adversarial (GRL).
        score_main = self._cls_score(self.causal_head, causal_repr)
        loss_cls   = self._ce_loss(score_main, labels)

        # ── A2 base: spurious adversarial (GRL) ─────────────────────────────────
        adv_score = self._cls_score(self.adv_head, self.grl(spurious_repr))
        loss_adv  = self._ce_loss(adv_score, labels)

        # ── Spurious probe — diagnostic only, never updates encoder ─────────────
        score_spurious_probe = self._cls_score(self.spurious_head, spurious_repr.detach())
        loss_spu_probe       = self._ce_loss(score_spurious_probe, labels)

        # ── Orthogonality loss ──────────────────────────────────────────────────
        loss_orth = self._orth_loss(causal_repr, spurious_repr)

        # ── Branch 1: Per-modality classification loss ─────────────────────────
        # Each projector is independently supervised to detect deepfakes.
        # Gradients flow directly through v_c and a_c → both must contribute.
        loss_modal_cls     = torch.tensor(0.0, device=video_feats.device)
        loss_modal_balance = torch.tensor(0.0, device=video_feats.device)
        if self.use_modal_cls:
            score_visual   = self._cls_score(self.modal_head_visual, v_c)
            score_audio    = self._cls_score(self.modal_head_audio,  a_c)
            ce_visual      = self._ce_loss(score_visual, labels)
            ce_audio       = self._ce_loss(score_audio,  labels)
            loss_modal_cls = ce_visual + ce_audio
            if self.use_modal_balance:
                # Penalise squared gap: forces both modalities to contribute equally
                loss_modal_balance = (ce_audio - ce_visual) ** 2

        # ── Branch 2: Label-conditioned cross-modal alignment loss ─────────────
        # Only align real samples: fake clips may have unilateral manipulation,
        # so unconditional alignment would destroy the fake signal.
        loss_cross_modal = torch.tensor(0.0, device=video_feats.device)
        if self.use_cross_modal:
            loss_cross_modal = self._cross_modal_loss(v_c, a_c, labels)

        # ── Branch 3: Modal discriminator adversarial loss ─────────────────────
        # Discriminator tries to tell v_c from a_c tokens.
        # GRL reverses gradient → projectors learn modality-invariant features.
        loss_modal_disc = torch.tensor(0.0, device=video_feats.device)
        if self.use_modal_disc:
            loss_modal_disc = self._modal_disc_loss(v_c, a_c)

        # ── Total loss ──────────────────────────────────────────────────────────
        loss = (
            loss_cls
            + self.lambda_adv           * loss_adv
            + self.lambda_orth          * loss_orth
            + self.lambda_modal_cls     * loss_modal_cls
            + self.lambda_modal_balance * loss_modal_balance
            + self.lambda_cross_modal   * loss_cross_modal
            + self.lambda_modal_disc    * loss_modal_disc
        )

        self.log("train_loss",              loss,              on_step=False, on_epoch=True)
        self.log("train_loss_cls",          loss_cls,          on_step=False, on_epoch=True)
        self.log("train_loss_adv",          loss_adv,          on_step=False, on_epoch=True)
        self.log("train_loss_orth",         loss_orth,         on_step=False, on_epoch=True)
        self.log("train_loss_spu_probe",    loss_spu_probe,    on_step=False, on_epoch=True)
        self.log("train_loss_modal_cls",     loss_modal_cls,     on_step=False, on_epoch=True)
        self.log("train_loss_modal_balance", loss_modal_balance, on_step=False, on_epoch=True)
        self.log("train_loss_cross_modal",  loss_cross_modal,   on_step=False, on_epoch=True)
        self.log("train_loss_modal_disc",   loss_modal_disc,   on_step=False, on_epoch=True)
        return loss

    # ── Validation ─────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        with torch.no_grad():
            s_causal   = self.forward(video_feats, audio_feats, mode="causal")
            s_spurious = self.forward(video_feats, audio_feats, mode="spurious")
        self._val_scores["causal"].append(s_causal.cpu().numpy())
        self._val_scores["spurious"].append(s_spurious.cpu().numpy())
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
            self.log(f"val_auc_{mode}", auc, prog_bar=(mode == "causal"))

        print(
            f"\n[Epoch {self.current_epoch}]  "
            f"AUC causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}"
        )

    # ── Optimizer ──────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
