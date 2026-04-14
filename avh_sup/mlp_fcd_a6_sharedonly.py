"""
AVH-Align A6-sharedonly — FCD + Domain Adversarial on Shared Subspace Only
===========================================================================

Key modification over A6:
  - domain_head_c operates on GRL(Z_shared_pool) where
    Z_shared_pool = cat([s_v, s_a]).mean(T)  [dim = syn_dim*2 = 512]
    instead of GRL(Z_c_pool) = GRL(cat([s_v, u_v, s_a, u_a]).mean(T))
  - This PROTECTS u_v and u_a from domain adversarial gradient pressure
  - u_v and u_a remain task-focused (only supervised by L_MI + L_task)
  - domain_head_s still operates on Z_s_pool = cat([r_v, r_a]).mean(T)

Rationale: mechanistic probe results suggest A6's gain comes from u_v and u_a.
Applying domain pressure to the full Z_c = [s_v, u_v, s_a, u_a] risks
interfering with exactly the channels that carry the useful signal.
The more principled design: push domain only out of shared (s_v, s_a),
pull domain into redundant (r_v, r_a), and keep unique channels clean.
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mlp_fcd import _make_mlp, CCD, URD, SDA, _sinkhorn_divergence
from mlp_fcd_a6 import _make_domain_head, grad_reverse


class AVH_FCD_A6_SharedOnly(L.LightningModule):
    """
    A6-sharedonly: domain adversarial pressure only on synergistic (shared) subspace.
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

        causal_dim   = (syn_dim + spec_dim) * 2   # 1024  (Z_c = [s_v, u_v, s_a, u_a])
        spurious_dim = spec_dim * 2               # 512   (Z_s = [r_v, r_a])
        shared_dim   = syn_dim * 2                # 512   (Z_shared = [s_v, s_a]) ← domain target

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

        # ── A6-sharedonly domain heads ────────────────────────────────────────
        # domain_head_c: adversarial on shared (s_v, s_a) only — dim=512
        # domain_head_s: discriminative on spurious (r_v, r_a)  — dim=512
        self.domain_head_c = _make_domain_head(shared_dim)   # 512, not 1024
        self.domain_head_s = _make_domain_head(spurious_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_mi   = float(hp.get("lambda_mi",   1.0))
        self.lambda_sda  = float(hp.get("lambda_sda",  0.5))
        self.lambda_dis  = float(hp.get("lambda_dis",  1.0))
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.lambda_dadv = float(hp.get("lambda_dadv", 1.0))
        self.lambda_ddis = float(hp.get("lambda_ddis", 1.0))
        self.grl_alpha   = float(hp.get("grl_alpha",   1.0))
        self.sda_eps     = float(hp.get("sda_eps",     0.05))
        self.sda_n_iter  = int  (hp.get("sda_n_iter",  5))
        self.lr          = float(hp.get("lr",          1e-3))

        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def _encode(self, video_feats, audio_feats):
        s_v, h_v = self.ccd_v(video_feats)
        s_a, h_a = self.ccd_a(audio_feats)
        u_v, r_v = self.urd_v(h_v)
        u_a, r_a = self.urd_a(h_a)
        causal_repr   = torch.cat([s_v, u_v, s_a, u_a], dim=-1)
        spurious_repr = torch.cat([r_v, r_a],            dim=-1)
        shared_repr   = torch.cat([s_v, s_a],            dim=-1)   # new
        return causal_repr, spurious_repr, shared_repr, {
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

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        causal_repr, spurious_repr, _, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            return self._cls_score(self.redundant_head, spurious_repr.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        causal_repr, spurious_repr, shared_repr, comp = self._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]

        # ── Domain losses — on shared (s_v,s_a) and spurious (r_v,r_a) only ──
        # Adversarial: push domain OUT of synergistic channels only
        Z_shared_pool = shared_repr.mean(1)         # [B, syn_dim*2=512]
        Z_s_pool      = spurious_repr.mean(1)       # [B, spec_dim*2=512]

        Z_shared_rev = grad_reverse(Z_shared_pool, self.grl_alpha)
        d_score_c    = self.domain_head_c(Z_shared_rev).squeeze(-1)
        loss_dadv    = F.binary_cross_entropy_with_logits(d_score_c, domain_labels)

        # Discriminative: pull domain INTO spurious channels
        d_score_s = self.domain_head_s(Z_s_pool).squeeze(-1)
        loss_ddis = F.binary_cross_entropy_with_logits(d_score_s, domain_labels)

        # ── Task losses (AV1M clips only) ─────────────────────────────────────
        av1m_mask = (cls_labels >= 0)
        if av1m_mask.any():
            c_repr_av1m = causal_repr[av1m_mask]
            s_repr_av1m = spurious_repr[av1m_mask]
            lbl_av1m    = cls_labels[av1m_mask]
            s_v_a, u_v_a, r_v_a = s_v[av1m_mask], u_v[av1m_mask], r_v[av1m_mask]
            s_a_a, u_a_a, r_a_a = s_a[av1m_mask], u_a[av1m_mask], r_a[av1m_mask]

            score_task = self._cls_score(self.causal_head, c_repr_av1m)
            loss_task  = self._ce_loss(score_task, lbl_av1m)

            score_uv = self._cls_score(self.unique_head_v, u_v_a)
            score_ua = self._cls_score(self.unique_head_a, u_a_a)
            loss_mi  = self._ce_loss(score_uv, lbl_av1m) + self._ce_loss(score_ua, lbl_av1m)

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
            loss_orth = 0.5 * (self._orth_loss(u_v_a, r_v_a) + self._orth_loss(u_a_a, r_a_a))

            score_spu = self._cls_score(self.redundant_head, s_repr_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)
        else:
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth = loss_spu_probe = \
                torch.tensor(0.0, device=video_feats.device, requires_grad=True)

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
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
        return loss

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
