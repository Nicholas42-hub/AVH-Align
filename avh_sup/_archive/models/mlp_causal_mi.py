"""
AVH-Align Causal + CLUB Model
==============================
Replaces the GRL+CE adversarial loss (and the failed MINE DV bound) with a
CLUB (Contrastive Log-ratio Upper Bound) based I(Z_s; Y) minimisation.

Why CLUB instead of MINE DV bound
----------------------------------
MINE DV bound collapsed to near-zero every epoch (mine_spu ~ 1e-9) because
the critic got stuck at a trivial solution — no gradient to the encoder.

CLUB (Cheng et al., NeurIPS 2020) is an *upper* bound:
    I_CLUB(Z_s; Y) = E_{p(z,y)} [log q_φ(y|z)] - E_{p(z)} E_{p(y)} [log q_φ(y|z)]
    where q_φ is a variational binary predictor.

Properties:
  • Always non-negative, finite-variance gradients even when MI ≈ 0
  • Designed for minimisation (upper bound → minimising reduces MI)
  • Stable: only needs a binary CE-trained predictor

Training algorithm (alternating, via manual_optimization)
----------------------------------------------------------
  Step A — Update CLUB predictor (tighten the bound):
      Minimise CE(q_φ(y | z_s.detach()), y)  →  q_φ learns accurate posterior

  Step B — Update main model (minimise MI upper bound via GRL):
      CLUB_ub = E[log q_φ(y|z_s_grl)] − E[E_{y'}[log q_φ(y'|z_s_grl)]]
      Encoder receives REVERSED gradient via GRL  →  Z_s becomes uninformative

Total loss (Step B):
    L = CE(full) + λ_c * CE(causal) + λ_club * CLUB_ub + λ_orth * orth_loss

Evaluation
----------
  v_full     → should be highest AUC
  v_causal   → hypothesis: ≈ v_full
  v_spurious → hypothesis: ≈ 0.5 on AV1M  AND  FakeAVCeleb
                (baseline gave 0.26 on FAVC → target: ~0.5)
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


# ── Gradient Reversal Layer ───────────────────────────────────────────────────
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


# ── Shared helpers ────────────────────────────────────────────────────────────
def _make_mlp(input_dim: int = 1024) -> nn.Sequential:
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


def _make_club_predictor(z_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    """
    Variational posterior q_φ(y|z_s) → binary logit.
    Input: z_s [B, z_dim] → scalar logit [B, 1]
    """
    return nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
    )


# ── CLUB upper bound ──────────────────────────────────────────────────────────
def club_upper_bound(
    predictor: nn.Module,
    z_s: torch.Tensor,      # [B, D] — GRL-transformed for encoder gradient
    y_float: torch.Tensor,  # [B, 1] — float labels {0.0, 1.0}
) -> torch.Tensor:
    """
    CLUB upper bound of I(Z_s; Y).

    CLUB = E_{p(z,y)} [log q(y|z)] - E_{p(z)} E_{p(y)} [log q(y|z)]

    For binary Y:
      log q(y=1|z) = log σ(logit)
      log q(y=0|z) = log(1 - σ(logit))

    encoder sees reversed gradient via GRL → minimises CLUB_ub → reduces I(Z_s;Y)
    """
    logits = predictor(z_s)  # [B, 1]

    # Joint: log q(y_i | z_s_i)
    log_q_joint = -F.binary_cross_entropy_with_logits(
        logits, y_float, reduction='none'
    )  # [B, 1]

    # Marginal: E_{y'~p(Y)} [log q(y'|z_s_i)]
    p_y1 = y_float.mean()
    log_q_y1 = F.logsigmoid(logits)
    log_q_y0 = F.logsigmoid(-logits)
    log_q_marginal = p_y1 * log_q_y1 + (1.0 - p_y1) * log_q_y0  # [B, 1]

    return (log_q_joint - log_q_marginal).mean()


# ── Main model ────────────────────────────────────────────────────────────────
class AVH_Causal_MI(L.LightningModule):
    """
    Causal-disentangled deepfake detector with CLUB-based I(Z_s; Y) minimisation.

    Config keys (under model_hparams):
        feat_dim           : int   (default 1024)
        proj_dim           : int   (default 512)
        club_hidden_dim    : int   (default 256)
        lambda_causal      : float (default 1.0)
        lambda_club        : float (default 1.0)
        lambda_orth        : float (default 0.5)
        grl_alpha          : float (default 5.0)
        lr                 : float (default 1e-3)
        lr_club_multiplier : float (default 2.0)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # manual alternating updates

        hp = config.get("model_hparams", {})
        feat_dim           = hp.get("feat_dim",            1024)
        proj_dim           = hp.get("proj_dim",             512)
        club_hidden_dim    = hp.get("club_hidden_dim",      256)
        self.lambda_causal = hp.get("lambda_causal",        1.0)
        self.lambda_club   = hp.get("lambda_club",          1.0)
        self.lambda_orth   = hp.get("lambda_orth",          0.5)
        grl_alpha          = hp.get("grl_alpha",            5.0)
        self.lr            = hp.get("lr",                  1e-3)
        self.lr_club_mult  = hp.get("lr_club_multiplier",   2.0)

        fused_dim = proj_dim * 2  # 1024

        # ── Projections ───────────────────────────────────────────────────────
        self.visual_proj_causal   = nn.Linear(feat_dim, proj_dim)
        self.visual_proj_spurious = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_causal    = nn.Linear(feat_dim, proj_dim)
        self.audio_proj_spurious  = nn.Linear(feat_dim, proj_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.full_head     = _make_mlp(fused_dim)
        self.causal_head   = _make_mlp(fused_dim)
        self.spurious_head = _make_mlp(fused_dim)   # probe only

        # ── CLUB predictor: q_φ(y|z_s) → binary logit ────────────────────────
        self.club_predictor = _make_club_predictor(fused_dim, club_hidden_dim)
        self.grl            = GradientReversal(alpha=grl_alpha)

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Representation helpers ────────────────────────────────────────────────
    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        v_c = self.visual_proj_causal(video_feats)
        v_s = self.visual_proj_spurious(video_feats)
        a_c = self.audio_proj_causal(audio_feats)
        a_s = self.audio_proj_spurious(audio_feats)
        causal_repr   = torch.cat([v_c, a_c], dim=-1)
        spurious_repr = torch.cat([v_s, a_s], dim=-1)
        return causal_repr, spurious_repr

    @staticmethod
    def _orth_loss(causal_repr, spurious_repr):
        B, T, D = causal_repr.shape
        c = F.normalize(causal_repr.reshape(-1, D), dim=-1)
        s = F.normalize(spurious_repr.reshape(-1, D), dim=-1)
        return ((c * s).sum(dim=-1) ** 2).mean()

    @staticmethod
    def _cls_score(head, repr_):
        return torch.logsumexp(head(repr_)[..., 0], dim=-1)

    @staticmethod
    def _ce_loss(score, labels):
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    # ── Forward (inference) ───────────────────────────────────────────────────
    def forward(self, video_feats, audio_feats, mode="full"):
        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)
        if mode == "full":
            return self._cls_score(self.full_head, causal_repr + spurious_repr)
        elif mode == "causal":
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, spurious_repr)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats, audio_feats, mode="full"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training (manual alternating optimisation) ────────────────────────────
    def training_step(self, batch, batch_idx):
        opt_main, opt_club = self.optimizers()

        video_feats, audio_feats, labels, _ = batch
        labels  = labels.long()
        y_float = labels.float().unsqueeze(-1)  # [B, 1]

        causal_repr, spurious_repr = self._encode(video_feats, audio_feats)
        z_s = spurious_repr.mean(dim=1)  # [B, fused_dim]

        # ── Step A: Update CLUB predictor ──────────────────────────────────────
        # Minimise CE(q_φ(y|z_s.detach()), y) → q_φ learns accurate posterior
        # z_s detached: encoder receives NO gradient from this step
        opt_club.zero_grad()
        loss_club_fit = F.binary_cross_entropy_with_logits(
            self.club_predictor(z_s.detach()), y_float
        )
        self.manual_backward(loss_club_fit)
        opt_club.step()

        # ── Step B: Update main model ──────────────────────────────────────────
        opt_main.zero_grad()

        score_full     = self._cls_score(self.full_head,     causal_repr + spurious_repr.detach())
        score_causal   = self._cls_score(self.causal_head,   causal_repr)
        score_spurious = self._cls_score(self.spurious_head, spurious_repr.detach())  # probe

        loss_cls      = self._ce_loss(score_full,     labels)
        loss_causal   = self._ce_loss(score_causal,   labels)
        loss_spurious = self._ce_loss(score_spurious, labels)  # logged only
        loss_orth     = self._orth_loss(causal_repr, spurious_repr)

        # CLUB upper bound via GRL
        # GRL reverses gradient → encoder minimises CLUB_ub → reduce I(Z_s;Y)
        z_s_grl   = self.grl(z_s)
        loss_club = club_upper_bound(self.club_predictor, z_s_grl, y_float)

        loss = (
            loss_cls
            + self.lambda_causal * loss_causal
            + self.lambda_club   * loss_club
            + self.lambda_orth   * loss_orth
        )
        self.manual_backward(loss)
        opt_main.step()

        self.log("train_loss",          loss,          on_step=False, on_epoch=True)
        self.log("train_loss_cls",      loss_cls,      on_step=False, on_epoch=True)
        self.log("train_loss_causal",   loss_causal,   on_step=False, on_epoch=True)
        self.log("train_loss_club",     loss_club,     on_step=False, on_epoch=True)
        self.log("train_loss_club_fit", loss_club_fit, on_step=False, on_epoch=True)
        self.log("train_loss_orth",     loss_orth,     on_step=False, on_epoch=True)
        self.log("train_loss_spurious", loss_spurious, on_step=False, on_epoch=True)

    # ── Validation ────────────────────────────────────────────────────────────
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

        self.log("val_auc_causal", results["causal"], prog_bar=True)
        print(
            f"\n[Epoch {self.current_epoch}] "
            f"AUC  full={results['full']:.4f}  "
            f"causal={results['causal']:.4f}  "
            f"spurious={results['spurious']:.4f}",
            flush=True,
        )

    # ── Optimisers (two: main + CLUB predictor) ───────────────────────────────
    def configure_optimizers(self):
        club_params = list(self.club_predictor.parameters())
        main_params = [p for n, p in self.named_parameters()
                       if not n.startswith("club_predictor")]
        opt_main = torch.optim.Adam(main_params, lr=self.lr)
        opt_club = torch.optim.Adam(club_params, lr=self.lr * self.lr_club_mult)
        return opt_main, opt_club
