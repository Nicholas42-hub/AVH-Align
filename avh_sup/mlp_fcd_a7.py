"""
AVH-Align A7 — FCD + Domain Adversarial + Label Adversarial on Z_s
=====================================================================

Extends A6 with one targeted fix for the Z_s > Z_c label-predictability
inversion: an explicit label adversarial loss on the spurious branch.

Problem diagnosed in A6:
  - In-domain probe: Z_s AUC (0.832) > Z_c AUC (0.770)
  - The spurious branch retains MORE fake/real signal than the causal branch.
  - Root cause: A6 only applies domain adversarial pressure on Z_s.
    Nothing directly prevents Z_s from encoding fake/real labels.

A7 fix — L_ladv (label adversarial on Z_s):
  GRL(Z_s_pool) → label_head_s → BCE(fake/real)  [AV1M clips only]
  This explicitly EXPELS fake/real label information from Z_s.

Combined with stronger domain adversarial (lambda_dadv 1.0 → 3.0), the
push-pull structure becomes:

  Z_c branch:
    ✓ L_task    — must predict fake/real (via causal_head)
    ✓ L_MI      — unique features must encode label
    ✗ L_dadv    — must NOT encode domain (GRL on Z_c → domain_head_c)

  Z_s branch:
    ✓ L_ddis    — must encode domain (discriminative on Z_s → domain_head_s)
    ✗ L_ladv    — must NOT encode fake/real label (GRL on Z_s → label_head_s)

Losses:
  L_task      — CE on causal_head([s_v, u_v, s_a, u_a])   (AV1M only)
  L_MI        — CE proxy MI(unique; label)                  (AV1M only)
  L_SDA       — Sinkhorn synergistic alignment              (AV1M only)
  L_Dis       — modality discrimination on redundant        (AV1M only)
  L_orth      — u ⊥ r orthogonality penalty                (AV1M only)
  L_dadv      — CE(domain_head_c(GRL(Z_c_pool)), domain)   (all clips)
  L_ddis      — CE(domain_head_s(Z_s_pool),      domain)   (all clips)
  L_ladv      — CE(label_head_s(GRL(Z_s_pool)),  label)    (AV1M only) [NEW]

Overall:
  L = L_task + λ_mi·L_MI + λ_sda·L_SDA + λ_dis·L_Dis + λ_orth·L_orth
    + λ_dadv·L_dadv + λ_ddis·L_ddis + λ_ladv·L_ladv
"""

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Re-use identical building blocks from A5/A6
from mlp_fcd import (
    _make_mlp,
    CCD,
    URD,
    SDA,
    _sinkhorn_divergence,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Gradient Reversal Layer  (identical to A6)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Small adversarial head  (domain or label, shared structure)
# ─────────────────────────────────────────────────────────────────────────────

def _make_adv_head(input_dim: int) -> nn.Sequential:
    """Lightweight 3-layer MLP → scalar logit."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  A7 Lightning module: AVH_FCD_A7
# ─────────────────────────────────────────────────────────────────────────────

class AVH_FCD_A7(L.LightningModule):
    """
    A7 — FCD + domain adversarial (A6) + label adversarial on Z_s (new).

    Key additions over A6:
      label_head_s + GRL : adversarial on fake/real labels for Z_s branch
                           forces Z_s to NOT predict fake/real labels
      lambda_dadv        : increased 1.0 → 3.0 for stronger domain disentanglement

    Batch format:
      (video_feats, audio_feats, cls_labels, domain_labels, paths)
      cls_labels   = -1   for FAVC clips (task loss and L_ladv skipped)
      domain_labels ∈ {0.0, 1.0}  for all clips
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})

        feat_dim = int(hp.get("feat_dim",  1024))
        syn_dim  = int(hp.get("syn_dim",    256))
        spec_dim = int(hp.get("spec_dim",   256))
        sda_dim  = int(hp.get("sda_dim",    128))
        mi_dim   = int(hp.get("mi_dim",     128))

        causal_dim   = (syn_dim + spec_dim) * 2   # 1024
        spurious_dim = spec_dim * 2               # 512

        # ── A5 components (identical) ─────────────────────────────────────────
        self.ccd_v = CCD(feat_dim, syn_dim, spec_dim)
        self.ccd_a = CCD(feat_dim, syn_dim, spec_dim)
        self.urd_v = URD(spec_dim)
        self.urd_a = URD(spec_dim)
        self.sda   = SDA(syn_dim, sda_dim)

        self.unique_head_v = nn.Sequential(
            nn.Linear(spec_dim, mi_dim), nn.LayerNorm(mi_dim), nn.ReLU(),
            nn.Linear(mi_dim, 1),
        )
        self.unique_head_a = nn.Sequential(
            nn.Linear(spec_dim, mi_dim), nn.LayerNorm(mi_dim), nn.ReLU(),
            nn.Linear(mi_dim, 1),
        )
        self.modality_head = nn.Sequential(
            nn.Linear(spec_dim, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.causal_head    = _make_mlp(causal_dim)
        self.redundant_head = _make_mlp(spurious_dim)

        # ── A6 domain heads (identical) ───────────────────────────────────────
        self.domain_head_c = _make_adv_head(causal_dim)    # adv on Z_c
        self.domain_head_s = _make_adv_head(spurious_dim)  # dis on Z_s

        # ── A7 new: label adversarial head on Z_s ────────────────────────────
        # GRL(Z_s_pool) → label_head_s → BCE(fake/real)
        # Forces Z_s to NOT encode fake/real label information
        self.label_head_s  = _make_adv_head(spurious_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_mi   = float(hp.get("lambda_mi",   1.0))
        self.lambda_sda  = float(hp.get("lambda_sda",  0.5))
        self.lambda_dis  = float(hp.get("lambda_dis",  1.0))
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.lambda_dadv = float(hp.get("lambda_dadv", 3.0))  # increased: 1.0 → 3.0
        self.lambda_ddis = float(hp.get("lambda_ddis", 1.0))
        self.lambda_ladv = float(hp.get("lambda_ladv", 2.0))  # NEW: label adv on Z_s
        self.grl_alpha   = float(hp.get("grl_alpha",   1.0))
        self.sda_eps     = float(hp.get("sda_eps",     0.05))
        self.sda_n_iter  = int  (hp.get("sda_n_iter",  5))
        self.lr          = float(hp.get("lr",          1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Internal helpers (identical to A6) ───────────────────────────────────

    def _encode(self, video_feats, audio_feats):
        s_v, h_v = self.ccd_v(video_feats)
        s_a, h_a = self.ccd_a(audio_feats)
        u_v, r_v = self.urd_v(h_v)
        u_a, r_a = self.urd_a(h_a)
        causal_repr   = torch.cat([s_v, u_v, s_a, u_a], dim=-1)
        spurious_repr = torch.cat([r_v, r_a],            dim=-1)
        return causal_repr, spurious_repr, {
            "s_v": s_v, "u_v": u_v, "r_v": r_v,
            "s_a": s_a, "u_a": u_a, "r_a": r_a,
        }

    @staticmethod
    def _cls_score(head, repr_):
        per_frame = head(repr_)[..., 0]
        return torch.logsumexp(per_frame, dim=-1)

    @staticmethod
    def _ce_loss(score, labels):
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _orth_loss(u, r):
        u_n = F.normalize(u.reshape(-1, u.shape[-1]), dim=-1)
        r_n = F.normalize(r.reshape(-1, r.shape[-1]), dim=-1)
        return ((u_n * r_n).sum(dim=-1) ** 2).mean()

    # ── Forward (inference — identical interface to A5/A6) ───────────────────

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        causal_repr, spurious_repr, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.redundant_head, spurious_repr.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()    # [B], -1 for FAVC
        domain_labels = domain_labels.float() # [B], 0/1

        causal_repr, spurious_repr, comp = self._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]

        # Pool over time for adversarial heads
        Z_c_pool = causal_repr.mean(1)    # [B, causal_dim]
        Z_s_pool = spurious_repr.mean(1)  # [B, spurious_dim]

        # ── Domain losses (all clips — both AV1M and FAVC) ───────────────────
        # Adversarial on Z_c: GRL prevents Z_c from encoding domain
        Z_c_rev   = grad_reverse(Z_c_pool, self.grl_alpha)
        d_score_c = self.domain_head_c(Z_c_rev).squeeze(-1)
        loss_dadv = F.binary_cross_entropy_with_logits(d_score_c, domain_labels)

        # Discriminative on Z_s: concentrate domain signal in spurious branch
        d_score_s = self.domain_head_s(Z_s_pool).squeeze(-1)
        loss_ddis = F.binary_cross_entropy_with_logits(d_score_s, domain_labels)

        # ── Task + label-adversarial losses (AV1M clips only) ────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            c_repr_av1m = causal_repr[av1m_mask]
            s_repr_av1m = spurious_repr[av1m_mask]
            lbl_av1m    = cls_labels[av1m_mask]
            s_v_a, u_v_a, r_v_a = s_v[av1m_mask], u_v[av1m_mask], r_v[av1m_mask]
            s_a_a, u_a_a, r_a_a = s_a[av1m_mask], u_a[av1m_mask], r_a[av1m_mask]
            Z_s_pool_av1m = Z_s_pool[av1m_mask]

            # Task classification loss (causal branch)
            score_task = self._cls_score(self.causal_head, c_repr_av1m)
            loss_task  = self._ce_loss(score_task, lbl_av1m)

            # MI proxy: unique features must predict label
            score_uv = self._cls_score(self.unique_head_v, u_v_a)
            score_ua = self._cls_score(self.unique_head_a, u_a_a)
            loss_mi  = self._ce_loss(score_uv, lbl_av1m) + self._ce_loss(score_ua, lbl_av1m)

            # Sinkhorn divergence: align synergistic projections
            s_v_proj = self.sda(s_v_a.mean(1))
            s_a_proj = self.sda(s_a_a.mean(1))
            loss_sda = _sinkhorn_divergence(s_v_proj, s_a_proj,
                                            eps=self.sda_eps, n_iter=self.sda_n_iter)

            # Modality discrimination on redundant features
            r_v_pool_av = r_v_a.mean(1)
            r_a_pool_av = r_a_a.mean(1)
            r_stack     = torch.cat([r_v_pool_av, r_a_pool_av], dim=0)
            mod_score   = self.modality_head(r_stack).squeeze(-1)
            mod_labels  = torch.cat([
                torch.zeros(r_v_pool_av.shape[0], device=r_v_pool_av.device),
                torch.ones( r_a_pool_av.shape[0], device=r_a_pool_av.device),
            ])
            loss_dis = F.binary_cross_entropy_with_logits(mod_score, mod_labels)

            # Orthogonality: u ⊥ r
            loss_orth = 0.5 * (self._orth_loss(u_v_a, r_v_a) + self._orth_loss(u_a_a, r_a_a))

            # Spurious probe (monitoring only — detached, not backpropagated)
            score_spu = self._cls_score(self.redundant_head, s_repr_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)

            # ── [A7 NEW] Label adversarial on Z_s ────────────────────────────
            # GRL: gradient is reversed so encoder is pushed to EXPEL label info from Z_s
            Z_s_rev   = grad_reverse(Z_s_pool_av1m, self.grl_alpha)
            l_score_s = self.label_head_s(Z_s_rev).squeeze(-1)
            # Convert integer labels {0,1} to float
            lbl_float = lbl_av1m.float()
            loss_ladv = F.binary_cross_entropy_with_logits(l_score_s, lbl_float)

        else:
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth = \
                loss_spu_probe = loss_ladv = \
                torch.tensor(0.0, device=video_feats.device, requires_grad=True)

        # SVD regularisation every 50 steps
        if self.global_step % 50 == 0:
            self.urd_v.apply_svd_regularization()
            self.urd_a.apply_svd_regularization()

        loss = (
            loss_task
            + self.lambda_mi   * loss_mi
            + self.lambda_sda  * loss_sda
            + self.lambda_dis  * loss_dis
            + self.lambda_orth * loss_orth
            + self.lambda_dadv * loss_dadv
            + self.lambda_ddis * loss_ddis
            + self.lambda_ladv * loss_ladv
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_mi",        loss_mi,        **log_kw)
        self.log("train_loss_sda",       loss_sda,       **log_kw)
        self.log("train_loss_dis",       loss_dis,       **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_dadv",      loss_dadv,      **log_kw)
        self.log("train_loss_ddis",      loss_ddis,      **log_kw)
        self.log("train_loss_ladv",      loss_ladv,      **log_kw)
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
        return loss

    # ── Validation (AV1M val — identical to A6) ───────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            video_feats, audio_feats, labels, _, _ = batch
        else:
            video_feats, audio_feats, labels, _ = batch
        with torch.no_grad():
            s_causal   = self.forward(video_feats, audio_feats, mode="causal")
            s_spurious = self.forward(video_feats, audio_feats, mode="spurious")
        self._val_scores["causal"].append(s_causal.cpu().numpy())
        self._val_scores["spurious"].append(s_spurious.cpu().numpy())
        self._val_scores["full"].append(s_causal.cpu().numpy())
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
            f"\n[Epoch {self.current_epoch}]"
            f"  AUC  full={results['full']:.4f}"
            f"  causal={results['causal']:.4f}"
            f"  spurious={results['spurious']:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
