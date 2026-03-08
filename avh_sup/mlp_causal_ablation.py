"""
AVH-Align Causal Ablation Model
================================
Unified model for the three-step ablation study.

Ablation levels (additive, each incorporates all previous):
─────────────────────────────────────────────────────────────
A1  |  Adversarial loss only
    |  Classification from Z_c + Z_s (no stop-gradient)
    |  GRL adversarial loss on Z_s (push Z_s non-informative for label)

A2  |  + Stop Gradient  (a.k.a. Z_c-only classification)
    |  Classification RESTRICTED to Z_c only
    |  Z_s receives gradient ONLY from GRL adversarial (never from cls)

A3  |  + Domain-aware constraint
    |  + Domain adversarial on Z_s (GRL forces Z_s domain-invariant)
    |  AV1M and FAVC clips mixed; domain head (AV1M=0, FAVC=1) on Z_s
    |  Explicitly models invariance (Z_c) vs variability (Z_s)

Config flag summary
---────────────────
  ablation:
    use_adv:    true/false   # Component 1: adversarial loss on Z_s
    use_sg:     true/false   # Component 2: restrict cls to Z_c only
    use_domain: true/false   # Component 3: domain adversarial on Z_s

Architecture
─────────────
  visual_feats [T×1024], audio_feats [T×1024]
      │
      ├── visual_proj_causal   → v_c [T×proj_dim]
      ├── visual_proj_spurious → v_s [T×proj_dim]
      ├── audio_proj_causal    → a_c [T×proj_dim]
      └── audio_proj_spurious  → a_s [T×proj_dim]
      │
      causal_repr   = concat(v_c, a_c)   ← A/V sync signal (invariant)
      spurious_repr = concat(v_s, a_s)   ← modality-specific residuals (variable)
      │
      A1: full_head(Z_c + Z_s)       + adv_head(GRL(Z_s))
      A2: causal_head(Z_c)           + adv_head(GRL(Z_s))
      A3: causal_head(Z_c)           + adv_head(GRL(Z_s))
                                     + domain_head(GRL_domain(Z_s))
      spurious_head(Z_s.detach())    ← probe (never trains encoder)

Evaluation
──────────
  All three configs share the same inference interface:
    mode="full"     → full_head(Z_c + Z_s)  (not meaningful for A2/A3 except diagnostics)
    mode="causal"   → causal_head(Z_c)       ← primary inference head for A2/A3
    mode="spurious" → spurious_head(Z_s)     ← probe: should be ≈0.5 cross-domain if
                                               domain constraint works
  Hypothesis verified if:
    causal AUC ≈ full AUC   (Z_c is self-sufficient)
    spurious AUC ≈ 0.5      (Z_s non-informative, especially cross-domain)
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


def _make_mlp(input_dim: int = 1024) -> nn.Sequential:
    """1024→512→256→128→1 — shared classification MLP."""
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


def _make_domain_mlp(input_dim: int = 1024) -> nn.Sequential:
    """Lighter domain classifier: 1024→256→1."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


class AVH_Causal_Ablation(L.LightningModule):
    """
    Unified ablation model. Controlled by config['ablation'] flags.

    Batch format:
        A1/A2: (video_feats, audio_feats, cls_labels, paths)
        A3:    (video_feats, audio_feats, cls_labels, domain_labels, paths)
                where cls_labels = -1 for FAVC-only domain samples
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp  = config.get("model_hparams", {})
        abl = config.get("ablation", {})

        feat_dim = hp.get("feat_dim",  1024)
        proj_dim = hp.get("proj_dim",   512)
        fused_dim = proj_dim * 2  # concat(visual, audio) = 1024

        # ── Ablation flags ──────────────────────────────────────────────────────
        self.use_adv    = bool(abl.get("use_adv",    False))  # Component 1
        self.use_sg     = bool(abl.get("use_sg",     False))  # Component 2
        self.use_domain = bool(abl.get("use_domain", False))  # Component 3

        # ── Loss weights ────────────────────────────────────────────────────────
        self.lambda_orth   = hp.get("lambda_orth",   0.5)
        self.lambda_adv    = hp.get("lambda_adv",   10.0)
        self.lambda_domain = hp.get("lambda_domain",  5.0)
        grl_alpha          = hp.get("grl_alpha",      5.0)
        grl_domain_alpha   = hp.get("grl_domain_alpha", 1.0)
        self.lr            = hp.get("lr",            1e-3)

        # ── Projections ─────────────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── Classification heads ─────────────────────────────────────────────────
        # full_head: Z_c + Z_s — used in A1 (and as diagnostic in A2/A3)
        # causal_head: Z_c only — primary inference head for A2/A3
        # spurious_head: Z_s detached probe — never trains encoder, AUC diagnostic
        self.full_head     = _make_mlp(fused_dim)
        self.causal_head   = _make_mlp(fused_dim)
        self.spurious_head = _make_mlp(fused_dim)  # probe (encoder always detached)

        # ── Adversarial head (Component 1) ───────────────────────────────────────
        # adv_head: tries to predict fake/real from Z_s
        # GRL: reverses gradients → forces Z_s ≠ label-informative
        if self.use_adv:
            self.adv_head = _make_mlp(fused_dim)
            self.grl      = GradientReversal(alpha=grl_alpha)

        # ── Domain adversarial head (Component 3) ───────────────────────────────
        # domain_head: tries to predict source domain from Z_s (AV1M=0, FAVC=1)
        # GRL_domain: reverses gradients → forces Z_s ≠ domain-informative
        if self.use_domain:
            self.domain_head = _make_domain_mlp(fused_dim)
            self.grl_domain  = GradientReversal(alpha=grl_domain_alpha)

        # ── Validation accumulators ─────────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Representation helpers ─────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """Return (causal_repr, spurious_repr) each [B×T×fused_dim]."""
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)
        causal_repr   = torch.cat([v_c, a_c], dim=-1)
        spurious_repr = torch.cat([v_s, a_s], dim=-1)
        return causal_repr, spurious_repr

    @staticmethod
    def _orth_loss(causal_repr: torch.Tensor, spurious_repr: torch.Tensor) -> torch.Tensor:
        B, T, D = causal_repr.shape
        c = F.normalize(causal_repr.reshape(-1, D),   dim=-1)
        s = F.normalize(spurious_repr.reshape(-1, D), dim=-1)
        return ((c * s).sum(dim=-1) ** 2).mean()

    @staticmethod
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        """MLP head → LogSumExp pooling over time → clip score [B]."""
        per_frame = head(repr_)[..., 0]
        return torch.logsumexp(per_frame, dim=-1)

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _bce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy for domain head (one logit per clip, pooled mean)."""
        # score: [B] (LogSumExp-pooled); labels: [B] (0/1)
        return F.binary_cross_entropy_with_logits(score, labels.float())

    # ── Forward (inference) ────────────────────────────────────────────────────

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)
        if mode == "full":
            if self.use_sg:
                # A2/A3: full_head is never trained (Z_s has no cls gradient).
                # Classification IS Z_c-only → report causal_head as "full".
                return self._cls_score(self.causal_head, causal_repr)
            else:
                # A1: full_head trained on Z_c + Z_s
                return self._cls_score(self.full_head, causal_repr + spurious_repr)
        elif mode == "causal":
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, spurious_repr)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        # ── Unpack batch (A3 includes domain_labels) ───────────────────────────
        if self.use_domain:
            video_feats, audio_feats, cls_labels, domain_labels, _ = batch
            domain_labels = domain_labels.long()
        else:
            video_feats, audio_feats, cls_labels, _ = batch
            domain_labels = None

        cls_labels = cls_labels.long()
        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)

        # For A3: cls & adv losses applied ONLY on AV1M samples (domain_label == 0)
        if domain_labels is not None:
            av1m_mask = (domain_labels == 0)
        else:
            av1m_mask = torch.ones(cls_labels.shape[0], dtype=torch.bool, device=cls_labels.device)

        # ── Classification loss ─────────────────────────────────────────────────
        # A1 (no stop-grad): main cls uses full_head(Z_c + Z_s) — gradient flows
        #                    to BOTH Z_c and Z_s. Also train causal_head(Z_c) as
        #                    a secondary loss so it can be evaluated at test time.
        # A2/A3 (+stop-grad): classification RESTRICTED to causal_head(Z_c) only
        #                     — Z_s receives NO gradient from cls task.
        if av1m_mask.any():
            if not self.use_sg:
                # A1: train full_head on Z_c + Z_s (no stop-gradient)
                score_full_a1  = self._cls_score(self.full_head,   causal_repr[av1m_mask] + spurious_repr[av1m_mask])
                score_causal_a1 = self._cls_score(self.causal_head, causal_repr[av1m_mask])
                loss_cls = (
                    self._ce_loss(score_full_a1,   cls_labels[av1m_mask])
                    + self._ce_loss(score_causal_a1, cls_labels[av1m_mask])
                )
            else:
                # A2/A3: only Z_c used for classification
                score_main = self._cls_score(self.causal_head, causal_repr[av1m_mask])
                loss_cls   = self._ce_loss(score_main, cls_labels[av1m_mask])
        else:
            loss_cls = torch.tensor(0.0, device=video_feats.device)

        # ── Adversarial loss (Component 1) ──────────────────────────────────────
        # adv_head tries to predict labels from Z_s;
        # GRL reverses gradients → Z_s becomes non-informative for labels.
        # Only applied on labelled (AV1M) samples.
        loss_adv = torch.tensor(0.0, device=video_feats.device)
        if self.use_adv and av1m_mask.any():
            adv_score = self._cls_score(self.adv_head, self.grl(spurious_repr[av1m_mask]))
            loss_adv  = self._ce_loss(adv_score, cls_labels[av1m_mask])

        # ── Domain adversarial loss (Component 3) ──────────────────────────────
        # domain_head tries to predict source domain from Z_s;
        # GRL_domain reverses gradients → Z_s becomes domain-invariant.
        # Applied on ALL samples (both AV1M and FAVC needed for contrast).
        loss_domain = torch.tensor(0.0, device=video_feats.device)
        if self.use_domain and domain_labels is not None:
            domain_score = self._cls_score(self.domain_head, self.grl_domain(spurious_repr))
            loss_domain  = self._bce_loss(domain_score, domain_labels.float())

        # ── Spurious probe (never updates encoder — diagnostic only) ────────────
        score_spurious = self._cls_score(self.spurious_head, spurious_repr.detach())
        if av1m_mask.any():
            loss_spurious_probe = self._ce_loss(score_spurious[av1m_mask], cls_labels[av1m_mask])
        else:
            loss_spurious_probe = torch.tensor(0.0, device=video_feats.device)

        # ── Orthogonality loss ──────────────────────────────────────────────────
        loss_orth = self._orth_loss(causal_repr, spurious_repr)

        # ── Total loss ──────────────────────────────────────────────────────────
        loss = (
            loss_cls
            + self.lambda_adv    * loss_adv
            + self.lambda_domain * loss_domain
            + self.lambda_orth   * loss_orth
        )

        self.log("train_loss",         loss,               on_step=False, on_epoch=True)
        self.log("train_loss_cls",      loss_cls,          on_step=False, on_epoch=True)
        self.log("train_loss_adv",      loss_adv,          on_step=False, on_epoch=True)
        self.log("train_loss_domain",   loss_domain,       on_step=False, on_epoch=True)
        self.log("train_loss_spu_probe",loss_spurious_probe,on_step=False,on_epoch=True)
        self.log("train_loss_orth",     loss_orth,         on_step=False, on_epoch=True)
        return loss

    # ── Validation ─────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        # Validation is always AV1M only (no FAVC domain samples needed)
        if self.use_domain:
            video_feats, audio_feats, labels, domain_labels, _ = batch
        else:
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
            self.log(f"val_auc_{mode}", auc, prog_bar=(mode == "causal"))

        print(
            f"\n[Epoch {self.current_epoch}] "
            f"AUC  full={results['full']:.4f}  "
            f"causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}"
        )

    # ── Optimizer ──────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
