"""
AVH-Align DAOF — Dual-Axis Orthogonal Factorization
=====================================================

Motivation
----------
All prior methods (DANN/GRL, FCD, TriRoute/A6) decompose representations along
a single axis: either task-relevance OR domain-stability. This is insufficient
because a subspace can be simultaneously task-relevant AND domain-unstable
(the "causal shortcut" Z_cs quadrant). Standard GRL cannot remove domain signal
from Z_cs without also destroying task signal, so the domain probe stays ~0.97.

DAOF decomposes along two independent axes, yielding four quadrants:

  ┌─────────────────────┬──────────────────────────┐
  │                     │ Domain-stable  │ Domain-unstable │
  ├─────────────────────┼────────────────┼─────────────────┤
  │ Task-relevant       │  Z_ct  ✅     │  Z_cs  ⚠️       │
  │ Task-irrelevant     │  Z_nt  🗑️    │  Z_res 📌        │
  └─────────────────────┴────────────────┴─────────────────┘

- Z_ct (Causal Transferable): only this feeds the task head. Adversarially
  trained NOT to predict domain (GRL), conditionally on label Y.
- Z_cs (Causal Shortcut): task-relevant but domain-unstable signal. Expelled
  from the task head and trained to predict domain GIVEN label Y (conditional
  domain head). This is the key missing quadrant in all prior work.
- Z_res (Domain Signature): absorbs unconditional domain signal (codec, identity,
  recording conditions). Same role as A6's spurious branch.
- Z_nt: implicit residual, not explicitly modeled.

Implementation
--------------
Starting from A6's 4-factor decomposition (s_v, u_v, u_Δ, s_a, u_a):
  Z_task_raw = cat(s_v, u_v, u_Δ, s_a, u_a)  [causal_dim=1280]
  Z_res      = cat(r_v, r_a)                  [res_dim=512]

Second-level axis splitter:
  Z_ct = AxisSplitter_ct(Z_task_raw)  [ct_dim=512]
  Z_cs = AxisSplitter_cs(Z_task_raw)  [cs_dim=512]

Losses:
  L_task         — CE(task_head(Z_ct), Y)                           (labeled only)
  L_ct_dadv      — CE(domain_head_ct(GRL(Z_ct)), D)                 (all clips)
  L_cs_cond_dis  — CE(cond_domain_head([Z_cs, Y_emb]), D)           (all clips)
  L_res_dis      — CE(domain_head_res(Z_res), D)                    (all clips)
  L_ct_cs_orth   — ||Z_ct · Z_cs||_F²                               (all clips)
  L_mi           — CE proxy MI(u_v, u_a, u_Δ; Y)                   (labeled only)
  L_sda          — Sinkhorn(s_v, s_a)                               (labeled only)
  L_orth_ur      — u ⊥ r per modality                               (labeled only)

Batch format (5-tuple):
  (video_feats [B,T,1024], audio_feats [B,T,1024],
   cls_labels  [B]   (-1 = FAVC, skip task loss),
   domain_labels [B] (0.0=AV1M, 1.0=FAVC),
   paths)
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mlp_fcd import _make_mlp, CCD, URD, SDA, _sinkhorn_divergence
from mlp_fcd_a6 import (
    _GRLFunction,
    grad_reverse,
    _make_domain_head,
    CrossModalInconsistency,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Axis Splitter: maps Z_task_raw → [Z_ct, Z_cs]
# ─────────────────────────────────────────────────────────────────────────────

class AxisSplitter(nn.Module):
    """
    Projects a shared representation into two orthogonal subspaces:
      ct_branch: Causal Transferable  (domain-stable task signal)
      cs_branch: Causal Shortcut      (domain-unstable task signal)

    Uses separate projection heads rather than a split of a joint projection,
    so each branch can independently specialise to its objective.
    """

    def __init__(self, input_dim: int, ct_dim: int, cs_dim: int):
        super().__init__()
        self.ct_proj = nn.Sequential(
            nn.Linear(input_dim, ct_dim),
            nn.LayerNorm(ct_dim),
            nn.ReLU(),
            nn.Linear(ct_dim, ct_dim),
            nn.LayerNorm(ct_dim),
        )
        self.cs_proj = nn.Sequential(
            nn.Linear(input_dim, cs_dim),
            nn.LayerNorm(cs_dim),
            nn.ReLU(),
            nn.Linear(cs_dim, cs_dim),
            nn.LayerNorm(cs_dim),
        )

    def forward(self, z: torch.Tensor):
        return self.ct_proj(z), self.cs_proj(z)


# ─────────────────────────────────────────────────────────────────────────────
#  Conditional Domain Head: predicts D given (Z_cs, Y)
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalDomainHead(nn.Module):
    """
    Predicts domain D conditioned on label Y.
    This is the key mechanism for identifying Z_cs:
      - If a representation can predict D *even given* Y, it is Z_cs.
      - Standard GRL ignores Y and can't distinguish Z_ct from Z_cs.

    Y is embedded as a learned 32-dim vector.
    For FAVC real clips (cls_label=-1), Y=0 (real) is used.
    """

    def __init__(self, cs_dim: int, label_emb_dim: int = 32):
        super().__init__()
        self.y_embed = nn.Embedding(2, label_emb_dim)  # 0=real, 1=fake
        in_dim = cs_dim + label_emb_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_cs: torch.Tensor, cls_labels: torch.Tensor) -> torch.Tensor:
        """
        z_cs:       [B, cs_dim]
        cls_labels: [B] int, -1 treated as 0 (real)
        Returns:    [B] domain logits
        """
        # Treat unknown label (FAVC real, cls=-1) as real (0)
        y = cls_labels.clone()
        y[y < 0] = 0
        y_emb = self.y_embed(y)          # [B, label_emb_dim]
        inp = torch.cat([z_cs, y_emb], dim=-1)
        return self.head(inp).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  DAOF Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class AVH_FCD_DAOF(L.LightningModule):
    """
    DAOF — Dual-Axis Orthogonal Factorization for cross-domain AV deepfake detection.

    Architecture:
      Layer 1 (modality factorization, same as A6):
        CCD_v/a + URD_v/a + CrossModalInconsistency
        → s_v, u_v, r_v, s_a, u_a, r_a, u_delta

      Layer 2 (axis factorization, new):
        AxisSplitter(Z_task_raw) → [Z_ct, Z_cs]
        Z_res = cat(r_v, r_a)  (unchanged)

      Prediction:
        task_head(Z_ct) only  ← no Z_cs, no Z_res
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
        ct_dim    = int(hp.get("ct_dim",      512))
        cs_dim    = int(hp.get("cs_dim",      512))

        # causal_dim = (syn_dim + spec_dim)*2 + incon_dim = 1280
        causal_dim = (syn_dim + spec_dim) * 2 + incon_dim
        res_dim    = spec_dim * 2  # 512

        # ── Layer 1: modality factorization (identical to A6) ─────────────────
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
        self.cross_incon = CrossModalInconsistency(feat_dim, incon_dim)
        self.incon_head  = nn.Sequential(
            nn.Linear(incon_dim, mi_dim), nn.LayerNorm(mi_dim), nn.ReLU(),
            nn.Linear(mi_dim, 1),
        )

        # ── Layer 2: axis factorization (new) ─────────────────────────────────
        self.axis_splitter  = AxisSplitter(causal_dim, ct_dim, cs_dim)

        # Task head reads only Z_ct
        self.task_head      = _make_mlp(ct_dim)

        # Spurious probe (detached, for monitoring only)
        self.res_probe_head = _make_mlp(res_dim)

        # ── Domain heads ──────────────────────────────────────────────────────
        # Z_ct: adversarial (GRL), unconditional
        self.domain_head_ct  = _make_domain_head(ct_dim)

        # Z_cs: discriminative, CONDITIONAL on label Y (key DAOF mechanism)
        self.cond_domain_head_cs = ConditionalDomainHead(cs_dim,
                                                          label_emb_dim=int(hp.get("label_emb_dim", 32)))

        # Z_res: discriminative, unconditional (same role as A6's domain_head_s)
        self.domain_head_res = _make_domain_head(res_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_mi            = float(hp.get("lambda_mi",            1.0))
        self.lambda_sda           = float(hp.get("lambda_sda",           0.5))
        self.lambda_dis           = float(hp.get("lambda_dis",           1.0))
        self.lambda_orth_ur       = float(hp.get("lambda_orth_ur",       0.1))
        self.lambda_ct_dadv       = float(hp.get("lambda_ct_dadv",       1.0))
        self.lambda_cs_cond_dis   = float(hp.get("lambda_cs_cond_dis",   1.0))
        self.lambda_res_dis       = float(hp.get("lambda_res_dis",       1.0))
        self.lambda_ct_cs_orth    = float(hp.get("lambda_ct_cs_orth",    0.5))
        self.grl_alpha            = float(hp.get("grl_alpha",            1.0))
        self.lambda_incon_sparse  = float(hp.get("lambda_incon_sparse",  0.0))
        self.sda_eps              = float(hp.get("sda_eps",              0.05))
        self.sda_n_iter           = int  (hp.get("sda_n_iter",           5))
        self.lr                   = float(hp.get("lr",                   1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"task": [], "res": []}
        self._val_labels = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, video_feats, audio_feats):
        """Returns Z_ct, Z_cs, Z_res and individual components."""
        s_v, h_v = self.ccd_v(video_feats)
        s_a, h_a = self.ccd_a(audio_feats)
        u_v, r_v = self.urd_v(h_v)
        u_a, r_a = self.urd_a(h_a)
        u_delta  = self.cross_incon(video_feats, audio_feats)

        Z_task_raw = torch.cat([s_v, u_v, u_delta, s_a, u_a], dim=-1)  # [B,T,1280]
        Z_res      = torch.cat([r_v, r_a],                    dim=-1)  # [B,T,512]

        Z_ct, Z_cs = self.axis_splitter(Z_task_raw)  # each [B,T,ct_dim/cs_dim]

        return Z_ct, Z_cs, Z_res, {
            "s_v": s_v, "u_v": u_v, "r_v": r_v,
            "s_a": s_a, "u_a": u_a, "r_a": r_a,
            "u_delta": u_delta,
        }

    @staticmethod
    def _cls_score(head, z):
        """Pool over time via log-sum-exp after scoring each frame."""
        per_frame = head(z)[..., 0]
        return torch.logsumexp(per_frame, dim=-1)

    @staticmethod
    def _ce_loss(score, labels):
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _orth_loss(a, b):
        """Cross-covariance orthogonality penalty: ||A^T B||_F² / (N * da * db).
        Works when a and b have different feature dimensions."""
        a_flat = a.reshape(-1, a.shape[-1])          # [N, da]
        b_flat = b.reshape(-1, b.shape[-1])          # [N, db]
        a_c = a_flat - a_flat.mean(0, keepdim=True)  # center
        b_c = b_flat - b_flat.mean(0, keepdim=True)
        N = a_flat.shape[0]
        cross_cov = (a_c.T @ b_c) / N               # [da, db]
        return (cross_cov ** 2).mean()

    # ── Forward (inference) ───────────────────────────────────────────────────

    def forward(self, video_feats, audio_feats, mode: str = "task"):
        Z_ct, Z_cs, Z_res, _ = self._encode(video_feats, audio_feats)
        if mode in ("task", "causal", "full"):
            return self._cls_score(self.task_head, Z_ct)
        elif mode in ("res", "spurious"):
            return self._cls_score(self.res_probe_head, Z_res.detach())
        elif mode == "cs":
            # Z_cs probe (domain shortcut signal, should be near chance)
            return self._cls_score(self.task_head, Z_cs)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "task"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()    # [B], -1 for FAVC
        domain_labels = domain_labels.float() # [B], 0/1

        Z_ct, Z_cs, Z_res, comp = self._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]
        u_delta = comp["u_delta"]

        # Pool over time
        Z_ct_pool  = Z_ct.mean(1)   # [B, ct_dim]
        Z_cs_pool  = Z_cs.mean(1)   # [B, cs_dim]
        Z_res_pool = Z_res.mean(1)  # [B, res_dim]

        # ── Domain losses (all clips) ─────────────────────────────────────────

        # 1. Z_ct: unconditional adversarial — make it hard to predict domain
        Z_ct_rev = grad_reverse(Z_ct_pool, self.grl_alpha)
        loss_ct_dadv = F.binary_cross_entropy_with_logits(
            self.domain_head_ct(Z_ct_rev).squeeze(-1), domain_labels
        )

        # 2. Z_cs: conditional discriminative — Z_cs should predict domain GIVEN label
        #    This is the key: identifies the causal-shortcut quadrant
        loss_cs_cond_dis = F.binary_cross_entropy_with_logits(
            self.cond_domain_head_cs(Z_cs_pool, cls_labels), domain_labels
        )

        # 3. Z_res: unconditional discriminative — absorb raw domain signal
        loss_res_dis = F.binary_cross_entropy_with_logits(
            self.domain_head_res(Z_res_pool).squeeze(-1), domain_labels
        )

        # 4. Z_ct ⊥ Z_cs: keep the two quadrants orthogonal
        loss_ct_cs_orth = self._orth_loss(Z_ct, Z_cs)

        # ── Task losses (labeled AV1M clips only) ─────────────────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            Z_ct_av  = Z_ct[av1m_mask]
            Z_res_av = Z_res[av1m_mask]
            lbl_av   = cls_labels[av1m_mask]
            s_v_a, u_v_a, r_v_a = s_v[av1m_mask], u_v[av1m_mask], r_v[av1m_mask]
            s_a_a, u_a_a, r_a_a = s_a[av1m_mask], u_a[av1m_mask], r_a[av1m_mask]
            u_delta_a = u_delta[av1m_mask]

            # Primary task loss: task head reads only Z_ct
            score_task = self._cls_score(self.task_head, Z_ct_av)
            loss_task  = self._ce_loss(score_task, lbl_av)

            # MI proxy on modality-specific unique factors
            score_uv     = self._cls_score(self.unique_head_v,  u_v_a)
            score_ua     = self._cls_score(self.unique_head_a,  u_a_a)
            score_udelta = self._cls_score(self.incon_head,     u_delta_a)
            loss_mi = (
                self._ce_loss(score_uv,     lbl_av)
                + self._ce_loss(score_ua,     lbl_av)
                + self._ce_loss(score_udelta, lbl_av)
            )

            # Sinkhorn alignment of shared factors across modalities
            s_v_proj = self.sda(s_v_a.mean(1))
            s_a_proj = self.sda(s_a_a.mean(1))
            loss_sda = _sinkhorn_divergence(s_v_proj, s_a_proj,
                                            eps=self.sda_eps, n_iter=self.sda_n_iter)

            # Modality discrimination on residual (keeps r_v ≠ r_a)
            r_v_pool = r_v_a.mean(1)
            r_a_pool = r_a_a.mean(1)
            r_stack  = torch.cat([r_v_pool, r_a_pool], dim=0)
            mod_labels = torch.cat([
                torch.zeros(r_v_pool.shape[0], device=video_feats.device),
                torch.ones( r_a_pool.shape[0], device=video_feats.device),
            ])
            loss_dis = F.binary_cross_entropy_with_logits(
                self.modality_head(r_stack).squeeze(-1), mod_labels
            )

            # Orthogonality: u ⊥ r per modality
            loss_orth_ur = (1 / 3) * (
                self._orth_loss(u_v_a,    r_v_a)
                + self._orth_loss(u_a_a,    r_a_a)
                + 0.5 * (self._orth_loss(u_delta_a, r_v_a)
                         + self._orth_loss(u_delta_a, r_a_a))
            )

            # Real-clip sparsity on u_delta
            real_mask = (lbl_av == 0)
            if real_mask.any() and self.lambda_incon_sparse > 0.0:
                loss_incon_sparse = (u_delta_a[real_mask] ** 2).mean()
            else:
                loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)

            # Spurious probe (detached, for monitoring)
            score_res_probe = self._cls_score(
                self.res_probe_head, Z_res_av.detach()
            )
            loss_res_probe = self._ce_loss(score_res_probe, lbl_av)
        else:
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth_ur = \
                loss_res_probe = torch.tensor(0.0, device=video_feats.device,
                                              requires_grad=True)
            loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)

        # SVD regularisation on URD modules
        if self.global_step % 50 == 0:
            self.urd_v.apply_svd_regularization()
            self.urd_a.apply_svd_regularization()

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (
            loss_task
            + self.lambda_mi           * loss_mi
            + self.lambda_sda          * loss_sda
            + self.lambda_dis          * loss_dis
            + self.lambda_orth_ur      * loss_orth_ur
            + self.lambda_ct_dadv      * loss_ct_dadv
            + self.lambda_cs_cond_dis  * loss_cs_cond_dis
            + self.lambda_res_dis      * loss_res_dis
            + self.lambda_ct_cs_orth   * loss_ct_cs_orth
            + self.lambda_incon_sparse * loss_incon_sparse
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",              loss,              **log_kw)
        self.log("train_loss_task",         loss_task,         **log_kw)
        self.log("train_loss_mi",           loss_mi,           **log_kw)
        self.log("train_loss_sda",          loss_sda,          **log_kw)
        self.log("train_loss_dis",          loss_dis,          **log_kw)
        self.log("train_loss_orth_ur",      loss_orth_ur,      **log_kw)
        self.log("train_loss_ct_dadv",      loss_ct_dadv,      **log_kw)
        self.log("train_loss_cs_cond_dis",  loss_cs_cond_dis,  **log_kw)
        self.log("train_loss_res_dis",      loss_res_dis,      **log_kw)
        self.log("train_loss_ct_cs_orth",   loss_ct_cs_orth,   **log_kw)
        self.log("train_loss_incon_sparse", loss_incon_sparse, **log_kw)
        self.log("train_loss_res_probe",    loss_res_probe,    **log_kw)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"task": [], "res": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:
            video_feats, audio_feats, labels, _, _ = batch
        else:
            video_feats, audio_feats, labels, _ = batch
        with torch.no_grad():
            s_task = self.forward(video_feats, audio_feats, mode="task")
            s_res  = self.forward(video_feats, audio_feats, mode="res")
        self._val_scores["task"].append(s_task.cpu().numpy())
        self._val_scores["res"].append(s_res.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        labels = np.concatenate(self._val_labels)
        results = {}
        for mode in ("task", "res"):
            scores = np.concatenate(self._val_scores[mode])
            try:
                auc = roc_auc_score(y_true=labels, y_score=scores)
            except ValueError:
                auc = float("nan")
            results[mode] = auc
            # Use "val_auc_causal" key for task head so existing callbacks work
            log_key = "val_auc_causal" if mode == "task" else f"val_auc_{mode}"
            self.log(log_key, auc, prog_bar=(mode == "task"))

        # Also read training domain losses from trainer metrics for monitoring
        cb = self.trainer.callback_metrics if self.trainer is not None else {}
        loss_res_dis = cb.get("train_loss_res_dis", float("nan"))
        loss_ct_dadv = cb.get("train_loss_ct_dadv", float("nan"))
        if hasattr(loss_res_dis, "item"):
            loss_res_dis = loss_res_dis.item()
        if hasattr(loss_ct_dadv, "item"):
            loss_ct_dadv = loss_ct_dadv.item()
        print(
            f"\n[DAOF Epoch {self.current_epoch}]"
            f"  task_AUC={results['task']:.4f}"
            f"  res_AUC={results['res']:.4f}"
            f"  loss_res_dis={loss_res_dis:.4f}"
            f"  loss_ct_dadv={loss_ct_dadv:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
