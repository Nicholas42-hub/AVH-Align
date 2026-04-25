"""
AVH-Align A7 — A6 + GRL task-adversarial head on Z_s
=====================================================

Extends A6 (FCD + domain push-pull) with an additional adversarial loss that
prevents fake/real (task) information from leaking into the spurious branch:

  New loss:
    L_tadv_s = BCE(task_adv_head_s(GRL(Z_s_pool)), fake_label)  (AV1M only)

This mirrors the domain adversarial loss on Z_c, but for the task label on Z_s:
  - domain_head_c(GRL(Z_c)) → domain info EXPELLED from causal branch
  - task_adv_head_s(GRL(Z_s)) → task (fake/real) info EXPELLED from spurious branch

Expected effect: spurious AUC → 0.5  (currently ~0.13–0.18 in A5/A6 = anti-predictive)

All other components are identical to A6 (mlp_fcd_a6.py).
New hyperparameter: lambda_tadv_s (default 0.5)
"""

import torch
import torch.nn.functional as F

from mlp_fcd_a6 import AVH_FCD_A6, _make_domain_head, grad_reverse


class AVH_FCD_A7(AVH_FCD_A6):
    """A6 + GRL task-adversarial head on spurious branch."""

    def __init__(self, config: dict):
        super().__init__(config)

        hp = config.get("model_hparams", {})
        spec_dim     = int(hp.get("spec_dim", 256))
        spurious_dim = spec_dim * 2

        # GRL head: expels fake/real signal from Z_s
        self.task_adv_head_s = _make_domain_head(spurious_dim)
        self.lambda_tadv_s   = float(hp.get("lambda_tadv_s", 0.5))

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        causal_repr, spurious_repr, comp = self._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]
        u_delta = comp["u_delta"]

        Z_c_pool = causal_repr.mean(1)
        Z_s_pool = spurious_repr.mean(1)

        # ── Domain losses ────────────────────────────────────────────────────
        Z_c_rev     = grad_reverse(Z_c_pool, self.grl_alpha)
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

        # ── Task losses + new task-adversarial loss on Z_s ───────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            c_repr_av1m = causal_repr[av1m_mask]
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
                    self._ce_loss(score_uv,     lbl_av1m)
                    + self._ce_loss(score_ua,   lbl_av1m)
                    + self._ce_loss(score_udelta, lbl_av1m)
                )
            else:
                loss_mi = (
                    self._ce_loss(score_uv, lbl_av1m)
                    + self._ce_loss(score_ua, lbl_av1m)
                )

            s_v_proj = self.sda(s_v_a.mean(1))
            s_a_proj = self.sda(s_a_a.mean(1))
            from mlp_fcd import _sinkhorn_divergence
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

            if u_delta_a is not None:
                loss_orth = (1 / 3) * (
                    self._orth_loss(u_v_a,    r_v_a)
                    + self._orth_loss(u_a_a,  r_a_a)
                    + 0.5 * (self._orth_loss(u_delta_a, r_v_a) + self._orth_loss(u_delta_a, r_a_a))
                )
            else:
                loss_orth = (1 / 2) * (
                    self._orth_loss(u_v_a, r_v_a)
                    + self._orth_loss(u_a_a, r_a_a)
                )

            real_mask_av1m = (lbl_av1m == 0)
            if u_delta_a is not None and real_mask_av1m.any() and self.lambda_incon_sparse > 0.0:
                loss_incon_sparse = (u_delta_a[real_mask_av1m] ** 2).mean()
            else:
                loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)

            score_spu = self._cls_score(self.redundant_head, s_repr_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)

            # ── NEW: GRL task-adversarial loss on Z_s ────────────────────────
            # Reverse gradient so encoder is pushed to remove fake/real info from Z_s
            Z_s_rev_av1m   = grad_reverse(Z_s_pool[av1m_mask], self.grl_alpha)
            task_adv_score = self.task_adv_head_s(Z_s_rev_av1m).squeeze(-1)
            loss_tadv_s    = F.binary_cross_entropy_with_logits(
                task_adv_score, lbl_av1m.float()
            )
        else:
            loss_task = loss_mi = loss_sda = loss_dis = loss_orth = loss_spu_probe = \
                torch.tensor(0.0, device=video_feats.device, requires_grad=True)
            loss_incon_sparse = torch.tensor(0.0, device=video_feats.device)
            loss_tadv_s       = torch.tensor(0.0, device=video_feats.device, requires_grad=True)

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
            + self.lambda_tadv_s       * loss_tadv_s
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",              loss,              **log_kw)
        self.log("train_loss_task",         loss_task,         **log_kw)
        self.log("train_loss_mi",           loss_mi,           **log_kw)
        self.log("train_loss_sda",          loss_sda,          **log_kw)
        self.log("train_loss_dis",          loss_dis,          **log_kw)
        self.log("train_loss_orth",         loss_orth,         **log_kw)
        self.log("train_loss_dadv",         loss_dadv,         **log_kw)
        self.log("train_loss_ddis",         loss_ddis,         **log_kw)
        self.log("train_loss_incon_sparse", loss_incon_sparse, **log_kw)
        self.log("train_loss_spu_probe",    loss_spu_probe,    **log_kw)
        self.log("train_loss_tadv_s",       loss_tadv_s,       **log_kw)
        return loss
