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
                                     + domain_head(GRL_domain(Z_c))  ← push Z_c domain-inv
                                     + domain_head_s(Z_s)            ← pull Z_s absorbs domain
      spurious_head(Z_s.detach())        ← 4-layer MLP probe (never trains encoder)

Evaluation
──────────
  All three configs share the same inference interface:
    mode="full"            → full_head(Z_c + Z_s)  (not meaningful for A2/A3 except diagnostics)
    mode="causal"          → causal_head(Z_c)       ← primary inference head for A2/A3
    mode="spurious"        → spurious_head(Z_s)     ← MLP probe: should be ≈0.5
  Hypothesis verified if:
    causal AUC ≈ full AUC        (Z_c is self-sufficient)
    spurious AUC ≈ 0.5           (MLP probe: non-informative Z_s)
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


def _make_domain_mlp_nclass(input_dim: int, n_classes: int) -> nn.Sequential:
    """N-class domain classifier: input_dim→256→n_classes."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, n_classes),
    )


class MINENetwork(nn.Module):
    """
    Statistics network T(z_c, z_s) for MINE mutual information estimation.

    MINE lower bound:  MI(Z_c; Z_s) >= E[T(z_c, z_s)] - log E[exp(T(z_c, z_s'))]
    where z_s' is drawn from the marginal (shuffled across the batch).

    Training protocol (alternating):
      Step 1 — maximise lower bound w.r.t. T (better MI estimate)
      Step 2 — minimise lower bound w.r.t. encoder (enforce independence)
    """
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """z_c, z_s: [B, D] → scalar per sample [B]."""
        return self.net(torch.cat([z_c, z_s], dim=-1)).squeeze(-1)


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

        feat_dim  = hp.get("feat_dim",  1024)
        proj_dim  = hp.get("proj_dim",   512)
        # proj_dim_spurious: capacity bottleneck for Z_s.
        # Reducing this (e.g. 32) prevents Z_s from physically holding the
        # sync signal — a signal-free geometric separation constraint.
        self.proj_dim_spurious = int(hp.get("proj_dim_spurious", proj_dim))
        fused_dim_causal   = proj_dim * 2                       # [v_c; a_c]
        fused_dim_spurious = self.proj_dim_spurious * 2         # [v_s; a_s]
        fused_dim = fused_dim_causal  # classification heads always use causal dim

        # ── Ablation flags ──────────────────────────────────────────────────────
        self.use_adv    = bool(abl.get("use_adv",    False))  # Component 1
        self.use_sg     = bool(abl.get("use_sg",     False))  # Component 2
        self.use_domain = bool(abl.get("use_domain", False))  # Component 3
        self.use_mine   = bool(abl.get("use_mine",   False))  # Component 4: MINE independence
        # Component 5: manipulation-type adversarial on Z_c
        # Forces Z_c invariant to HOW a video was manipulated (4 manip types),
        # so it captures universal fake detection rather than type-specific artefacts.
        self.use_manip_domain = bool(abl.get("use_manip_domain", False))

        # ── Loss weights ────────────────────────────────────────────────────────
        self.lambda_orth   = hp.get("lambda_orth",   0.5)
        self.lambda_adv    = hp.get("lambda_adv",   10.0)
        self.lambda_domain = hp.get("lambda_domain",  5.0)
        self.lambda_mine   = hp.get("lambda_mine",    1.0)
        self.lambda_manip_domain = hp.get("lambda_manip_domain", 5.0)
        grl_alpha          = hp.get("grl_alpha",      5.0)
        grl_domain_alpha   = hp.get("grl_domain_alpha", 1.0)
        grl_manip_alpha    = hp.get("grl_manip_alpha",  1.0)
        self.lr            = hp.get("lr",            1e-3)
        self.mine_lr       = hp.get("mine_lr",       1e-4)

        # ── Projections ─────────────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, self.proj_dim_spurious)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, self.proj_dim_spurious)

        # ── Classification heads ─────────────────────────────────────────────────
        # full_head: Z_c + Z_s — used in A1 (and as diagnostic in A2/A3)
        # causal_head: Z_c only — primary inference head for A2/A3
        # spurious_head: 4-layer MLP probe — never trains encoder
        self.full_head     = _make_mlp(fused_dim_causal + fused_dim_spurious)
        self.causal_head   = _make_mlp(fused_dim_causal)
        self.spurious_head = _make_mlp(fused_dim_spurious)  # MLP probe (encoder always detached)

        # ── Adversarial head (Component 1) ───────────────────────────────────────
        # adv_head: tries to predict fake/real from Z_s
        # GRL: reverses gradients → forces Z_s ≠ label-informative
        if self.use_adv:
            self.adv_head = _make_mlp(fused_dim_spurious)
            self.grl      = GradientReversal(alpha=grl_alpha)

        # ── Domain adversarial head (Component 3) ───────────────────────────────
        # Push-pull design (mirrors A6):
        #   domain_head   on Z_c  + GRL_domain → forces Z_c domain-invariant (push)
        #   domain_head_s on Z_s  (no GRL)     → concentrates domain info in Z_s (pull)
        # Together: Z_c loses domain signal, Z_s absorbs it.
        if self.use_domain:
            self.lambda_ddis   = hp.get("lambda_ddis", 1.0)   # weight for Z_s pull term
            self.domain_head   = _make_domain_mlp(fused_dim_causal)
            self.domain_head_s = _make_domain_mlp(fused_dim_spurious)
            self.grl_domain    = GradientReversal(alpha=grl_domain_alpha)

        # ── Manipulation-type adversarial head (Component 5) ─────────────────────
        # manip_domain_head: tries to predict manipulation type from Z_c
        # GRL_manip: reverses gradients → forces Z_c invariant to manip type
        # domain_labels here are manip_type ∈ {0,1,2,3}
        if self.use_manip_domain:
            n_manip = int(hp.get("n_manip_classes", 4))
            self.manip_domain_head = _make_domain_mlp_nclass(fused_dim_causal, n_manip)
            self.grl_manip         = GradientReversal(alpha=grl_manip_alpha)

        # ── MINE independence network (Component 4) ─────────────────────────────
        # Replaces the per-sample orth loss with a proper MI lower-bound estimator.
        # Requires alternating optimisation → automatic_optimization disabled.
        if self.use_mine:
            mine_hidden = hp.get("mine_hidden", 512)
            self.mine_net = MINENetwork(fused_dim, hidden_dim=mine_hidden)
            self.automatic_optimization = False
            # Number of MINE network inner updates per encoder step.
            # More inner steps → better MI estimate before the encoder sees it.
            self.mine_inner_steps = int(hp.get("mine_inner_steps", 5))

        # ── β-VAE components (Component 4 — A4) ─────────────────────────────────
        # When use_betavae=True the four deterministic Linear projections above
        # are replaced with variational heads (output 2×proj_dim: μ ∥ logvar).
        # A decoder reconstructs the original [V̄; Ā] clip representation to
        # prevent information collapse.  β_s >> β_c compresses Z_s harder,
        # forcing sync-relevant signal to accumulate exclusively in Z_c.
        self.use_betavae = bool(abl.get("use_betavae", False))
        if self.use_betavae:
            self.beta_c       = hp.get("beta_c",        1.0)
            self.beta_s       = hp.get("beta_s",       10.0)
            self.lambda_recon = hp.get("lambda_recon",   0.0)
            # Linear warm-up: β scales from 0 → target over this many epochs.
            # Lets the encoder learn to classify before KL pressure is applied.
            self.beta_warmup_epochs = int(hp.get("beta_warmup_epochs", 20))
            # Overwrite projections to output 2×dim (μ ∥ logvar).
            # logvar is clamped to [-4, 4] to prevent sigma explosion when
            # KL ≈ 0 during warm-up (unconstrained logvar → infinite noise).
            self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim * 2)
            self.visual_proj_spurious = nn.Linear(feat_dim, self.proj_dim_spurious * 2)
            self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim * 2)
            self.audio_proj_spurious  = nn.Linear(feat_dim, self.proj_dim_spurious * 2)
            # Decoder only instantiated when lambda_recon > 0.
            if self.lambda_recon > 0:
                self.decoder = nn.Sequential(
                    nn.Linear(fused_dim_causal + fused_dim_spurious, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, feat_dim * 2),
                )

        # ── Validation accumulators ─────────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Representation helpers ─────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """Return (causal_repr, spurious_repr, vae_extras|None).

        vae_extras = (mu_c, lv_c, mu_s, lv_s)  each [B, T, fused_dim]
        when use_betavae is True; None otherwise.

        During training the VAE path adds Gaussian noise via reparameterisation.
        At eval/inference time the deterministic posterior mean is used instead.
        """
        if not self.use_betavae:
            v_c = self.visual_proj_causal(video_feats)
            v_s = self.visual_proj_spurious(video_feats)
            a_c = self.audio_proj_causal(audio_feats)
            a_s = self.audio_proj_spurious(audio_feats)
            return (
                torch.cat([v_c, a_c], dim=-1),
                torch.cat([v_s, a_s], dim=-1),
                None,
            )

        # ── Variational path ────────────────────────────────────────────────────
        # Each projection outputs 2×dim; we split into (μ, logvar) and
        # reparameterise.  logvar is clamped to [-4, 4] to prevent sigma
        # explosion when KL ≈ 0 during warm-up.  At eval time, use mean only.
        def _reparam(h: torch.Tensor):
            mu, logvar = h.chunk(2, dim=-1)
            logvar = logvar.clamp(-4.0, 4.0)
            if self.training:
                z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
            else:
                z = mu
            return z, mu, logvar

        v_c, mu_vc, lv_vc = _reparam(self.visual_proj_causal(video_feats))
        v_s, mu_vs, lv_vs = _reparam(self.visual_proj_spurious(video_feats))
        a_c, mu_ac, lv_ac = _reparam(self.audio_proj_causal(audio_feats))
        a_s, mu_as, lv_as = _reparam(self.audio_proj_spurious(audio_feats))

        causal_repr   = torch.cat([v_c, a_c], dim=-1)
        spurious_repr = torch.cat([v_s, a_s], dim=-1)
        mu_c = torch.cat([mu_vc, mu_ac], dim=-1)
        lv_c = torch.cat([lv_vc, lv_ac], dim=-1)
        mu_s = torch.cat([mu_vs, mu_as], dim=-1)
        lv_s = torch.cat([lv_vs, lv_as], dim=-1)

        return causal_repr, spurious_repr, (mu_c, lv_c, mu_s, lv_s)

    @staticmethod
    def _orth_loss(causal_repr: torch.Tensor, spurious_repr: torch.Tensor) -> torch.Tensor:
        D_c = causal_repr.shape[-1]
        D_s = spurious_repr.shape[-1]
        D = min(D_c, D_s)
        c = F.normalize(causal_repr.reshape(-1, D_c)[..., :D],   dim=-1)
        s = F.normalize(spurious_repr.reshape(-1, D_s)[..., :D], dim=-1)
        return ((c * s).sum(dim=-1) ** 2).mean()

    @staticmethod
    def _mine_lower_bound(T_joint: torch.Tensor, T_marginal: torch.Tensor) -> torch.Tensor:
        """
        MINE lower bound on MI (numerically stable via logsumexp):
          MI >= E[T(joint)] - log(E[exp(T(marginal))])
                           = E[T(joint)] - (logsumexp(T(marginal)) - log(N))
        Positive = high MI, near-zero = near-independence.
        """
        n = T_marginal.size(0)
        import math
        return T_joint.mean() - (torch.logsumexp(T_marginal, dim=0) - math.log(n))

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
        causal_repr, spurious_repr, _ = self._encode(video_feats, audio_feats)
        if mode == "full":
            if self.use_sg:
                # A2/A3: full_head is never trained (Z_s has no cls gradient).
                # Classification IS Z_c-only → report causal_head as "full".
                return self._cls_score(self.causal_head, causal_repr)
            else:
                # A1: full_head trained on Z_c + Z_s concatenated
                return self._cls_score(self.full_head, torch.cat([causal_repr, spurious_repr], dim=-1))
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
        # ── Unpack batch (A3/manip includes domain_labels) ────────────────────
        if self.use_domain or self.use_manip_domain:
            video_feats, audio_feats, cls_labels, domain_labels, _ = batch
            domain_labels = domain_labels.long()
        else:
            video_feats, audio_feats, cls_labels, _ = batch
            domain_labels = None

        cls_labels = cls_labels.long()
        causal_repr, spurious_repr, vae_extras = self._encode(video_feats, audio_feats)

        # For A3: cls & adv losses applied ONLY on AV1M samples (domain_label == 0)
        # For manip_domain: all samples are AV1M, so mask is always True
        if domain_labels is not None and self.use_domain and not self.use_manip_domain:
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
                # A0 (no adv, no sg): full_head(Z_c+Z_s) + causal_head(Z_c) +
                # spurious_head(Z_s.detach()) — all three heads trained so that
                # both Z_c and Z_s independently encode label signal. This makes
                # causal/spurious head AUCs meaningful for ablation evaluation.
                # A1 (adv, no sg): same full_head cls, but ALSO train causal_head(Z_c)
                # as secondary so it can be evaluated at test time.
                score_full = self._cls_score(self.full_head, torch.cat([causal_repr[av1m_mask], spurious_repr[av1m_mask]], dim=-1))
                loss_cls = self._ce_loss(score_full, cls_labels[av1m_mask])
                if not self.use_adv:
                    # A0 only: also train causal_head and spurious_head independently.
                    score_causal = self._cls_score(self.causal_head, causal_repr[av1m_mask])
                    score_spurious = self._cls_score(self.spurious_head, spurious_repr[av1m_mask].detach())
                    loss_cls = loss_cls + self._ce_loss(score_causal, cls_labels[av1m_mask]) \
                                       + self._ce_loss(score_spurious, cls_labels[av1m_mask])
                else:
                    # A1 only: secondary causal_head loss for test-time evaluation.
                    score_causal_secondary = self._cls_score(self.causal_head, causal_repr[av1m_mask])
                    loss_cls = loss_cls + self._ce_loss(score_causal_secondary, cls_labels[av1m_mask])
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
        # Push: GRL on mean-pooled Z_c → Z_c becomes domain-invariant.
        # Pull: direct domain supervision on mean-pooled Z_s → Z_s absorbs domain.
        # Applied on ALL samples (both AV1M and FAVC needed for contrast).
        loss_domain = torch.tensor(0.0, device=video_feats.device)
        if self.use_domain and domain_labels is not None:
            # Pass full [B, T, D] tensors; _cls_score applies LogSumExp over T → [B]
            domain_score_c = self._cls_score(self.domain_head, self.grl_domain(causal_repr))
            loss_dadv    = self._bce_loss(domain_score_c, domain_labels.float())     # push Z_c
            domain_score_s = self._cls_score(self.domain_head_s, spurious_repr)
            loss_ddis    = self._bce_loss(domain_score_s, domain_labels.float())     # pull Z_s
            loss_domain  = loss_dadv + self.lambda_ddis * loss_ddis

        # ── Manipulation-type adversarial loss (Component 5) ─────────────────
        # manip_domain_head predicts manip_type (0/1/2/3) from mean-pooled Z_c
        # via GRL → Z_c becomes invariant to manipulation type.
        loss_manip_domain = torch.tensor(0.0, device=video_feats.device)
        if self.use_manip_domain and domain_labels is not None:
            z_c_grl    = self.grl_manip(causal_repr)        # GRL on [B, T, D]
            z_c_pooled = z_c_grl.mean(1)                    # mean pool → [B, D]
            manip_logits = self.manip_domain_head(z_c_pooled)  # [B, n_manip]
            loss_manip_domain = F.cross_entropy(manip_logits, domain_labels)

        # ── Spurious probe (never updates encoder — diagnostic only) ──────────────
        # Trains independently on detached Z_s. Loss NOT added to main loss.
        s_detached = spurious_repr.detach()
        score_spurious = self._cls_score(self.spurious_head, s_detached)
        if av1m_mask.any():
            loss_spurious_probe = self._ce_loss(score_spurious[av1m_mask], cls_labels[av1m_mask])
        else:
            loss_spurious_probe = torch.tensor(0.0, device=video_feats.device)

        # ── β-VAE losses (Component 4 — A4) ────────────────────────────────────
        # KL_c and KL_s encourage both posteriors toward N(0,I).  β_s >> β_c
        # compresses Z_s more aggressively; combined with the adversarial GRL,
        # this leaves no room for spurious shortcuts in Z_s.
        # The reconstruction term ensures Z_c + Z_s retain full information.
        loss_vae = torch.tensor(0.0, device=video_feats.device)
        if self.use_betavae and vae_extras is not None:
            mu_c, lv_c, mu_s, lv_s = vae_extras
            kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).sum(-1).mean()
            kl_s = -0.5 * (1 + lv_s - mu_s.pow(2) - lv_s.exp()).sum(-1).mean()
            # Linear β warm-up: β = 0 at epoch 0, target at beta_warmup_epochs.
            warmup_scale = min(1.0, self.current_epoch / max(1, self.beta_warmup_epochs))
            loss_vae = warmup_scale * self.beta_c * kl_c + warmup_scale * self.beta_s * kl_s
            # Reconstruction term only if decoder was instantiated (lambda_recon > 0).
            if self.lambda_recon > 0 and hasattr(self, 'decoder'):
                z_c_mean = causal_repr.mean(1)
                z_s_mean = spurious_repr.mean(1)
                x_hat  = self.decoder(torch.cat([z_c_mean, z_s_mean], dim=-1))
                x_orig = torch.cat([video_feats.mean(1), audio_feats.mean(1)], dim=-1)
                loss_vae = loss_vae + self.lambda_recon * F.mse_loss(x_hat, x_orig)

        # ── MINE or orthogonality independence loss ─────────────────────────────
        if self.use_mine:
            # ── MINE alternating optimisation ──────────────────────────────────
            # Guard: skip batches whose base losses are non-finite (e.g. a clip
            # whose AV-HuBERT features have a zero-norm frame → NaN after L2 norm).
            if not (torch.isfinite(loss_cls) and torch.isfinite(loss_adv)):
                return

            # Pool Z_c / Z_s over time → [B, D] for MINE input.
            opt_main, opt_mine, _opt_spu = self.optimizers()

            z_c_flat = causal_repr.mean(1)    # [B, D], gradients flow to projections
            z_s_flat = spurious_repr.mean(1)  # [B, D]

            # --- Step 1: Update MINE network (maximise MI lower bound) ----------
            # Run mine_inner_steps updates so the MI estimate is well-converged
            # before the encoder step (WGAN-style inner loop).
            for _inner in range(self.mine_inner_steps):
                z_s_shuffled = z_s_flat[torch.randperm(z_s_flat.size(0))].detach()
                T_joint    = self.mine_net(z_c_flat.detach(), z_s_flat.detach())
                T_marginal = self.mine_net(z_c_flat.detach(), z_s_shuffled)
                mi_for_mine = self._mine_lower_bound(T_joint, T_marginal)
                opt_mine.zero_grad()
                self.manual_backward(-mi_for_mine)  # maximise = minimise negative
                opt_mine.step()

            # --- Step 2: Update encoder (minimise MI lower bound) ---------------
            # Freeze MINE net so gradients flow only to projection layers.
            for p in self.mine_net.parameters():
                p.requires_grad_(False)

            z_s_shuffled_enc = z_s_flat[torch.randperm(z_s_flat.size(0))]
            T_joint_enc    = self.mine_net(z_c_flat, z_s_flat)
            T_marginal_enc = self.mine_net(z_c_flat, z_s_shuffled_enc)
            mi_for_encoder = self._mine_lower_bound(T_joint_enc, T_marginal_enc)

            loss = (
                loss_cls
                + self.lambda_adv          * loss_adv
                + self.lambda_domain       * loss_domain
                + self.lambda_manip_domain * loss_manip_domain
                + self.lambda_mine         * mi_for_encoder
                + loss_vae
            )

            opt_main.zero_grad()
            self.manual_backward(loss)
            # Clip gradients before encoder step.
            # lambda_adv=10 * grl_alpha=5 gives an effective ~50x gradient scale
            # on projection layers via GRL.  max_norm must be large enough to
            # let the GRL signal through (raised from 1.0 → 50.0 so adversarial
            # gradients are not clipped away, which was suppressing Z_s cleanup).
            torch.nn.utils.clip_grad_norm_(
                [p for pg in opt_main.param_groups for p in pg['params']
                 if p.grad is not None],
                max_norm=50.0,
            )
            opt_main.step()

            # Restore MINE net gradients for next MINE update.
            for p in self.mine_net.parameters():
                p.requires_grad_(True)

            loss_orth = mi_for_mine.detach()   # log the MI estimate as "orth" metric

            # Spurious probe independent update (detached from encoder).
            opt_spu = self.optimizers()[2]
            opt_spu.zero_grad()
            self.manual_backward(loss_spurious_probe)
            opt_spu.step()

        else:
            # ── Standard orthogonality loss ────────────────────────────────────
            loss_orth = self._orth_loss(causal_repr, spurious_repr)

            loss = (
                loss_cls
                + self.lambda_adv          * loss_adv
                + self.lambda_domain       * loss_domain
                + self.lambda_manip_domain * loss_manip_domain
                + self.lambda_orth         * loss_orth
                + loss_vae
            )

        # With manual optimisation Lightning needs on_step=True to accumulate epoch mean.
        log_kwargs = dict(on_step=True, on_epoch=True, prog_bar=False) if self.use_mine \
                     else dict(on_step=False, on_epoch=True)
        self.log("train_loss",              loss,                **log_kwargs)
        self.log("train_loss_cls",          loss_cls,            **log_kwargs)
        self.log("train_loss_adv",          loss_adv,            **log_kwargs)
        self.log("train_loss_domain",       loss_domain,         **log_kwargs)
        self.log("train_loss_manip_domain", loss_manip_domain,   **log_kwargs)
        self.log("train_loss_spu_probe",    loss_spurious_probe, **log_kwargs)
        self.log("train_loss_orth",         loss_orth,           **log_kwargs)
        self.log("train_loss_vae",          loss_vae,            **log_kwargs)
        if not self.use_mine:
            return loss

    # ── Validation ─────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        # Validation is always AV1M only (no FAVC domain samples needed)
        if self.use_domain or self.use_manip_domain:
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
        if self.use_mine:
            # Three separate optimisers:
            #   opt_main [0]: encoder + all heads except mine_net and spurious_head
            #   opt_mine [1]: mine_net only (updated to maximise MI)
            #   opt_spu  [2]: spurious_head only (independent probe, detached encoder)
            mine_params = list(self.mine_net.parameters())
            spu_params  = list(self.spurious_head.parameters())
            mine_param_ids = {id(p) for p in mine_params}
            spu_param_ids  = {id(p) for p in spu_params}
            main_params = [p for p in self.parameters()
                           if id(p) not in mine_param_ids and id(p) not in spu_param_ids]
            opt_main = torch.optim.Adam(main_params, lr=self.lr)
            opt_mine = torch.optim.Adam(mine_params,  lr=self.mine_lr)
            opt_spu  = torch.optim.Adam(spu_params,   lr=self.lr)
            return [opt_main, opt_mine, opt_spu]
        return torch.optim.Adam(self.parameters(), lr=self.lr)
