"""
AVH-Align A6 — FCD + Explicit Domain Adversarial Disentanglement
=================================================================

Extends A5 (FCD three-way decomposition) with an explicit push-pull on
domain information:

  Push: GRL(domain_classifier(Z_c)) → domain info EXPELLED from causal branch
  Pull: domain_classifier(Z_s)       → domain info CONCENTRATED in spurious branch

Training data: AV1M train/val  (domain=0)  +  FAVC val-real clips (domain=1)
Eval  data:    FAVC test split  (untouched — never seen during training)

Batch format (5-tuple):
  (video_feats [B,T,1024], audio_feats [B,T,1024],
   cls_labels  [B]   (int, -1 for FAVC clips = skip task loss),
   domain_labels [B] (float, 0=AV1M, 1=FAVC),
   paths)

Losses:
  L_task      — CE on causal_head([s_v, u_v, u_Δ, s_a, u_a])  (AV1M only)
  L_MI        — CE proxy MI(unique; label) for u_v, u_a, u_Δ    (AV1M only)
  L_SDA       — Sinkhorn synergistic alignment                   (AV1M only)
  L_Dis       — modality discrimination on redundant             (AV1M only)
  L_orth      — u ⊥ r orthogonality penalty (incl. u_Δ)         (AV1M only)
  L_dadv      — CE(domain_head_c(GRL(Z_c_pool)), domain)        (all clips)
  L_ddis      — CE(domain_head_s(Z_s_pool),      domain)        (all clips)

Four-way causal factorisation:
  s_v, s_a  — cross-modal shared (redundant) information
  u_v, u_a  — modality-unique manipulation evidence
  u_Δ       — cross-modal inconsistency: CrossAttn(z_v,z_a) − CrossAttn(z_a,z_v)

Overall:
  L = L_task + λ_mi·L_MI + λ_sda·L_SDA + λ_dis·L_Dis + λ_orth·L_orth
    + λ_dadv·L_dadv + λ_ddis·L_ddis
"""

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Re-use the identical building blocks from A5
from mlp_fcd import (
    _make_mlp,
    CCD,
    URD,
    SDA,
    _sinkhorn_divergence,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Gradient Reversal Layer
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
#  Small domain head  (shared for both causal-adv and spurious-dis variants)
# ─────────────────────────────────────────────────────────────────────────────

def _make_domain_head(input_dim: int) -> nn.Sequential:
    """Lightweight 3-layer MLP → scalar domain logit."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-modal inconsistency factor
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalInconsistency(nn.Module):
    """
    Computes u_Δ = CrossAttn(z_v, z_a) − CrossAttn(z_a, z_v) and projects
    the asymmetric difference to incon_dim dimensions.

    For genuine clips the two attention maps are symmetric and u_Δ ≈ 0.
    For FV-RA fakes the visual stream attends differently to audio than
    audio attends to video, producing a non-zero directional residual that
    directly encodes audio-visual temporal misalignment.
    """

    def __init__(self, feat_dim: int, incon_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_v2a = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.cross_a2v = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, incon_dim),
            nn.LayerNorm(incon_dim),
            nn.ReLU(),
        )

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        # z_v, z_a: [B, T, feat_dim]
        # CrossAttn(z_v, z_a): query=z_v, key/value=z_a
        c_v2a, _ = self.cross_v2a(z_v, z_a, z_a)  # [B, T, feat_dim]
        # CrossAttn(z_a, z_v): query=z_a, key/value=z_v
        c_a2v, _ = self.cross_a2v(z_a, z_v, z_v)  # [B, T, feat_dim]
        u_delta = c_v2a - c_a2v                    # [B, T, feat_dim]
        return self.proj(u_delta)                  # [B, T, incon_dim]


# ─────────────────────────────────────────────────────────────────────────────
#  A6 Lightning module: AVH_FCD_A6
# ─────────────────────────────────────────────────────────────────────────────

class AVH_FCD_A6(L.LightningModule):
    """
    A6 — FCD + cross-modal inconsistency factor + domain adversarial disentanglement.

    Key additions over A5:
      cross_incon    + incon_head : fourth causal factor u_Δ encoding AV asymmetry
      domain_head_c  + GRL        : adversarial, forces Z_c to NOT predict domain
      domain_head_s               : discriminative, pulls domain signal into Z_s

    Batch format:
      (video_feats, audio_feats, cls_labels, domain_labels, paths)
      cls_labels = -1  for FAVC clips (task loss skipped)
      domain_labels ∈ {0.0, 1.0}  for all clips
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})

        feat_dim  = int(hp.get("feat_dim",   1024))
        syn_dim   = int(hp.get("syn_dim",     256))
        spec_dim  = int(hp.get("spec_dim",    256))
        sda_dim   = int(hp.get("sda_dim",     128))
        mi_dim    = int(hp.get("mi_dim",      128))
        incon_dim = int(hp.get("incon_dim",   256))
        self._incon_dim = incon_dim

        causal_dim   = (syn_dim + spec_dim) * 2 + incon_dim
        spurious_dim = spec_dim * 2
        self.expose_residual_to_task_head = bool(
            hp.get("expose_residual_to_task_head", False)
        )
        task_head_dim = (
            causal_dim + spurious_dim
            if self.expose_residual_to_task_head
            else causal_dim
        )

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
        self.causal_head    = _make_mlp(task_head_dim)
        self.redundant_head = _make_mlp(spurious_dim)

        # ── A6 additions: cross-modal inconsistency factor ────────────────────
        if incon_dim > 0:
            self.cross_incon = CrossModalInconsistency(feat_dim, incon_dim)
            self.incon_head  = nn.Sequential(
                nn.Linear(incon_dim, mi_dim), nn.LayerNorm(mi_dim), nn.ReLU(),
                nn.Linear(mi_dim, 1),
            )

        # ── A6 additions: domain heads ────────────────────────────────────────
        # domain_head_c operates on GRL(Z_c_pool) → adversarial on causal branch
        # domain_head_s operates on Z_s_pool       → discriminative on spurious branch
        self.domain_head_c = _make_domain_head(causal_dim)
        self.domain_head_s = _make_domain_head(spurious_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_mi   = float(hp.get("lambda_mi",   1.0))
        self.lambda_sda  = float(hp.get("lambda_sda",  0.5))
        self.lambda_dis  = float(hp.get("lambda_dis",  1.0))
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.lambda_dadv         = float(hp.get("lambda_dadv",         1.0))  # domain-adv on Z_c
        self.lambda_ddis         = float(hp.get("lambda_ddis",         1.0))  # domain-dis on Z_s
        self.grl_alpha           = float(hp.get("grl_alpha",           1.0))  # GRL reversal strength
        self.lambda_incon_sparse = float(hp.get("lambda_incon_sparse", 0.0))  # ||u_Δ||² on real clips
        self.sda_eps     = float(hp.get("sda_eps",     0.05))
        self.sda_n_iter  = int  (hp.get("sda_n_iter",  5))
        self.lr          = float(hp.get("lr",          1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Internal helpers (identical to A5) ───────────────────────────────────

    def _encode(self, video_feats, audio_feats):
        s_v, h_v = self.ccd_v(video_feats)
        s_a, h_a = self.ccd_a(audio_feats)
        u_v, r_v = self.urd_v(h_v)
        u_a, r_a = self.urd_a(h_a)
        # Fourth factor: cross-modal inconsistency u_Δ (only if incon_dim > 0)
        if self._incon_dim > 0:
            u_delta = self.cross_incon(video_feats, audio_feats)  # [B, T, incon_dim]
            causal_repr = torch.cat([s_v, u_v, u_delta, s_a, u_a], dim=-1)
        else:
            u_delta = None
            causal_repr = torch.cat([s_v, u_v, s_a, u_a], dim=-1)
        spurious_repr = torch.cat([r_v, r_a],                     dim=-1)
        return causal_repr, spurious_repr, {
            "s_v": s_v, "u_v": u_v, "r_v": r_v,
            "s_a": s_a, "u_a": u_a, "r_a": r_a,
            "u_delta": u_delta,  # None when incon_dim=0
        }

    def _task_repr(self, causal_repr, spurious_repr):
        if self.expose_residual_to_task_head:
            return torch.cat([causal_repr, spurious_repr], dim=-1)
        return causal_repr

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

    # ── Forward (inference — identical interface to A5) ───────────────────────

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        causal_repr, spurious_repr, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            task_repr = self._task_repr(causal_repr, spurious_repr)
            return self._cls_score(self.causal_head, task_repr)
        elif mode == "spurious":
            return self._cls_score(self.redundant_head, spurious_repr.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()                          # [B], -1 for FAVC
        domain_labels = domain_labels.float()                      # [B], 0/1

        causal_repr, spurious_repr, comp = self._encode(video_feats, audio_feats)
        task_repr = self._task_repr(causal_repr, spurious_repr)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]
        u_delta = comp["u_delta"]

        # Pool over time for domain heads
        Z_c_pool = causal_repr.mean(1)    # [B, causal_dim]
        Z_s_pool = spurious_repr.mean(1)  # [B, spurious_dim]

        # ── Domain losses (only clips with valid domain_label >= 0) ──────────
        # domain_label == -1 signals "no domain supervision" (e.g. fake FAVC clips
        # when using real-only domain labeling). Backward-compatible: existing
        # training runs set domain_label=0/1 for all clips, so mask = all-True.
        Z_c_rev   = grad_reverse(Z_c_pool, self.grl_alpha)
        domain_mask = (domain_labels >= 0)
        if domain_mask.any():
            d_score_c = self.domain_head_c(Z_c_rev[domain_mask]).squeeze(-1)
            loss_dadv = F.binary_cross_entropy_with_logits(
                d_score_c, domain_labels[domain_mask])
            d_score_s = self.domain_head_s(Z_s_pool[domain_mask]).squeeze(-1)
            loss_ddis = F.binary_cross_entropy_with_logits(
                d_score_s, domain_labels[domain_mask])
        else:
            loss_dadv = torch.tensor(0.0, device=video_feats.device, requires_grad=True)
            loss_ddis = torch.tensor(0.0, device=video_feats.device, requires_grad=True)

        # ── Task losses (AV1M clips only, cls_label >= 0) ────────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            c_repr_av1m = task_repr[av1m_mask]
            s_repr_av1m = spurious_repr[av1m_mask]
            lbl_av1m    = cls_labels[av1m_mask]
            s_v_a, u_v_a, r_v_a = s_v[av1m_mask], u_v[av1m_mask], r_v[av1m_mask]
            s_a_a, u_a_a, r_a_a = s_a[av1m_mask], u_a[av1m_mask], r_a[av1m_mask]
            u_delta_a = u_delta[av1m_mask] if u_delta is not None else None

            score_task = self._cls_score(self.causal_head, c_repr_av1m)
            loss_task  = self._ce_loss(score_task, lbl_av1m)

            score_uv = self._cls_score(self.unique_head_v, u_v_a)
            score_ua = self._cls_score(self.unique_head_a, u_a_a)
            if u_delta_a is not None:
                score_udelta = self._cls_score(self.incon_head, u_delta_a)
                loss_mi = (
                    self._ce_loss(score_uv,    lbl_av1m)
                    + self._ce_loss(score_ua,    lbl_av1m)
                    + self._ce_loss(score_udelta, lbl_av1m)
                )
            else:
                loss_mi = (
                    self._ce_loss(score_uv, lbl_av1m)
                    + self._ce_loss(score_ua, lbl_av1m)
                )

            s_v_proj = self.sda(s_v_a.mean(1))
            s_a_proj = self.sda(s_a_a.mean(1))
            loss_sda = _sinkhorn_divergence(s_v_proj, s_a_proj,
                                            eps=self.sda_eps, n_iter=self.sda_n_iter)

            r_v_pool_av = r_v_a.mean(1)
            r_a_pool_av = r_a_a.mean(1)
            r_stack     = torch.cat([r_v_pool_av, r_a_pool_av], dim=0)
            mod_score   = self.modality_head(r_stack).squeeze(-1)
            mod_labels  = torch.cat([
                torch.zeros(r_v_pool_av.shape[0], device=r_v_pool_av.device),
                torch.ones( r_a_pool_av.shape[0], device=r_a_pool_av.device),
            ])
            loss_dis = F.binary_cross_entropy_with_logits(mod_score, mod_labels)

            loss_orth_base = (1 / 2) * (
                self._orth_loss(u_v_a, r_v_a)
                + self._orth_loss(u_a_a, r_a_a)
            )
            if u_delta_a is not None:
                loss_orth = (1 / 3) * (
                    self._orth_loss(u_v_a,    r_v_a)
                    + self._orth_loss(u_a_a,    r_a_a)
                    + 0.5 * (self._orth_loss(u_delta_a, r_v_a) + self._orth_loss(u_delta_a, r_a_a))
                )
            else:
                loss_orth = loss_orth_base

            # Real-clip sparsity: penalise ||u_Δ||² on genuine AV1M clips so
            # that u_Δ is only active when audio-visual streams actually mismatch.
            real_mask_av1m = (lbl_av1m == 0)
            if u_delta_a is not None and real_mask_av1m.any() and self.lambda_incon_sparse > 0.0:
                loss_incon_sparse = (u_delta_a[real_mask_av1m] ** 2).mean()
            else:
                loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)

            score_spu = self._cls_score(self.redundant_head, s_repr_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)
        else:
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth = loss_spu_probe = \
                torch.tensor(0.0, device=video_feats.device, requires_grad=True)
            loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)

        # SVD regularisation every 50 steps
        if self.global_step % 50 == 0:
            self.urd_v.apply_svd_regularization()
            self.urd_a.apply_svd_regularization()

        loss = (
            loss_task
            + self.lambda_mi           * loss_mi
            + self.lambda_sda          * loss_sda
            + self.lambda_dis          * loss_dis
            + self.lambda_orth         * loss_orth
            + self.lambda_dadv         * loss_dadv
            + self.lambda_ddis         * loss_ddis
            + self.lambda_incon_sparse * loss_incon_sparse
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_mi",        loss_mi,        **log_kw)
        self.log("train_loss_sda",       loss_sda,       **log_kw)
        self.log("train_loss_dis",       loss_dis,       **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_dadv",         loss_dadv,         **log_kw)
        self.log("train_loss_ddis",         loss_ddis,         **log_kw)
        self.log("train_loss_incon_sparse", loss_incon_sparse, **log_kw)
        self.log("train_loss_spu_probe",    loss_spu_probe,    **log_kw)
        return loss

    # ── Validation (AV1M val — same as A5) ───────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        # Val batches are AV1M only (4-tuple, no domain_label)
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
