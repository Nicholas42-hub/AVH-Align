"""
AVH-Align A8 — SAD: Synchrony-Appearance Decomposition
=======================================================

Diagnosis from A5–A7:
  The FCD decomposition's information-theoretic axes are INVERTED for this task.
  FCD assigns: redundant(V,A) → Z_s (spurious)
               synergistic(V,A) + unique(V,A) → Z_c (causal)

  But for AV deepfake detection:
    redundant(V,A) = AV cross-modal coherence  → the actual manipulation signal!
    unique(V), unique(A)                       → speaker identity, visual/audio style

  No amount of adversarial regularisation can fix this misalignment — the
  architecture's inductive bias is structurally wrong for the task.

A8 replaces FCD with a task-aware Synchrony-Appearance Decomposition (SAD):

  CAUSAL  =  Z_sync: cross-modal interaction features
               interaction(V, A) = cat(V, A, V⊙A, V−A)  [B, T, 4·feat_dim]
               Z_sync = proj_sync(interaction)           [B, T, sync_dim]

               • V⊙A (element-wise product): captures AV coherence  — large when
                 both modalities encode the same content (real); disrupted when
                 sources mismatch (fake).
               • V−A (difference): captures AV discrepancy  — large when content
                 differs across modalities.
               • Together these are the natural algebraic fingerprint of AV sync.

  SPURIOUS = Z_app: within-modal appearance features
               Z_app_v = proj_app(V)                    [B, T, app_dim]
               Z_app_a = proj_app(A)                    [B, T, app_dim]
               Z_app   = cat(Z_app_v, Z_app_a)          [B, T, 2·app_dim]

               • Each encoder sees ONLY one modality  →  structurally unable to
                 reason about cross-modal coherence.
               • Captures speaker identity, recording conditions, visual style  —
                 the domain-correlated shortcut features.

Push-pull design:
  Z_sync:    ✓ L_task  — must detect fake/real  (causal_head)
             ✗ L_dadv  — must NOT encode domain  (GRL → domain_head_c)

  Z_app:     ✓ L_ddis  — must encode domain      (domain_head_s)
             ✗ L_ladv  — must NOT predict label   (GRL → label_head_s)

  Both:      ○ L_orth  — Z_sync ⊥ Z_app per frame  (prevent information leakage)

Losses:
  L_task  — CE on causal_head(Z_sync)                  [AV1M only]
  L_orth  — mean cos²(Z_sync_frame, Z_app_frame)       [AV1M only, over T frames]
  L_dadv  — BCE(domain_head_c(GRL(Z_sync_pool)), dom)  [all clips]
  L_ddis  — BCE(domain_head_s(Z_app_pool),       dom)  [all clips]
  L_ladv  — BCE(label_head_s(GRL(Z_app_pool)),   lbl)  [AV1M only]

Overall:
  L = L_task + λ_orth·L_orth + λ_dadv·L_dadv + λ_ddis·L_ddis + λ_ladv·L_ladv

Why this beats FCD+adversarial:
  • Cross-modal arithmetic (V⊙A, V−A) directly encodes the manipulation cue
    as a structural prior — manipulation *breaks* V⊙A and *amplifies* V−A.
  • Within-modal projectors structurally cannot encode AV sync, so the
    architecture itself enforces the desired information partition.
  • Adversarial losses reinforce rather than fight the inductive bias.
"""

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


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
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _make_adv_head(input_dim: int) -> nn.Sequential:
    """Lightweight 3-layer MLP → scalar adversarial logit."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def _make_cls_head(input_dim: int) -> nn.Sequential:
    """4-layer classification MLP  input_dim → 512 → 256 → 128 → 1."""
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


class SynchronyEncoder(nn.Module):
    """
    Cross-modal synchrony encoder.

    Input : V [B, T, feat_dim],  A [B, T, feat_dim]
    Output: Z_sync [B, T, sync_dim]

    The interaction vector cat(V, A, V⊙A, V−A) encodes:
      • V⊙A  — coherence (element-wise agreement)
      • V−A  — discrepancy (element-wise mismatch)
    Both are disrupted in deepfake content where V and A come from different sources.
    """

    def __init__(self, feat_dim: int, sync_dim: int):
        super().__init__()
        in_dim = 4 * feat_dim  # [V, A, V⊙A, V−A]
        self.proj = nn.Sequential(
            nn.Linear(in_dim, sync_dim * 2),
            nn.LayerNorm(sync_dim * 2),
            nn.GELU(),
            nn.Linear(sync_dim * 2, sync_dim),
            nn.LayerNorm(sync_dim),
            nn.GELU(),
        )

    def forward(self, V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """V, A: [B, T, feat_dim] → [B, T, sync_dim]"""
        interaction = torch.cat([V, A, V * A, V - A], dim=-1)
        return self.proj(interaction)


class AppearanceEncoder(nn.Module):
    """
    Within-modal appearance encoder (shared weights for V and A).

    Input : X [B, T, feat_dim]
    Output: Z_app [B, T, app_dim]

    Processes ONE modality at a time — structurally unable to encode
    cross-modal coherence, so it captures identity/style features.
    """

    def __init__(self, feat_dim: int, app_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, app_dim),
            nn.LayerNorm(app_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, feat_dim] → [B, T, app_dim]"""
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
#  A8 Lightning module: AVH_SAD_A8
# ─────────────────────────────────────────────────────────────────────────────

class AVH_SAD_A8(L.LightningModule):
    """
    A8 — Synchrony-Appearance Decomposition (SAD).

    Z_sync (causal):   cross-modal interaction features — detects AV mismatch.
    Z_app  (spurious): per-modality appearance features — captures identity/domain.

    Batch format (training):
      (video_feats, audio_feats, cls_labels, domain_labels, paths)
      cls_labels   = -1  for FAVC clips (task loss and L_ladv skipped)
      domain_labels ∈ {0.0, 1.0}  for all clips

    Batch format (validation):
      (video_feats, audio_feats, cls_labels, paths)  [4-tuple, no domain label]
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp       = config.get("model_hparams", {})
        feat_dim = int(hp.get("feat_dim",  1024))
        sync_dim = int(hp.get("sync_dim",   512))   # causal branch dim
        app_dim  = int(hp.get("app_dim",    256))   # appearance per modality

        causal_dim   = sync_dim          # Z_sync dim for heads
        spurious_dim = 2 * app_dim       # cat(Z_app_v, Z_app_a)

        # ── SAD encoders ──────────────────────────────────────────────────────
        self.sync_enc  = SynchronyEncoder(feat_dim, sync_dim)
        self.app_enc_v = AppearanceEncoder(feat_dim, app_dim)
        self.app_enc_a = AppearanceEncoder(feat_dim, app_dim)

        # ── Classification heads ──────────────────────────────────────────────
        self.causal_head   = _make_cls_head(causal_dim)    # primary inference
        self.spurious_head = _make_cls_head(spurious_dim)  # probe (monitoring)

        # ── Adversarial / discriminative domain heads ─────────────────────────
        self.domain_head_c = _make_adv_head(causal_dim)    # GRL → adversarial
        self.domain_head_s = _make_adv_head(spurious_dim)  # direct → discriminative

        # ── Label adversarial head on Z_app ───────────────────────────────────
        # GRL on Z_app_pool → prevents appearance branch from encoding fake/real
        self.label_head_s  = _make_adv_head(spurious_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.lambda_dadv = float(hp.get("lambda_dadv", 1.0))
        self.lambda_ddis = float(hp.get("lambda_ddis", 1.0))
        self.lambda_ladv = float(hp.get("lambda_ladv", 1.0))
        self.grl_alpha   = float(hp.get("grl_alpha",   1.0))
        self.lr          = float(hp.get("lr",          1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Encoding ─────────────────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """
        Returns:
          causal_repr   [B, T, sync_dim]  — Z_sync cross-modal features
          spurious_repr [B, T, 2·app_dim] — Z_app appearance features
          components    dict with keys: z_sync, z_app_v, z_app_a
        """
        z_sync  = self.sync_enc(video_feats, audio_feats)
        z_app_v = self.app_enc_v(video_feats)
        z_app_a = self.app_enc_a(audio_feats)
        z_app   = torch.cat([z_app_v, z_app_a], dim=-1)
        return z_sync, z_app, {"z_sync": z_sync, "z_app_v": z_app_v, "z_app_a": z_app_a}

    @staticmethod
    def _cls_score(head: nn.Module, repr_: torch.Tensor) -> torch.Tensor:
        """logsumexp pooling over time frames → scalar score per sample."""
        per_frame = head(repr_)[..., 0]   # [B, T]
        return torch.logsumexp(per_frame, dim=-1)  # [B]

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _orth_loss(sync_frames: torch.Tensor, app_frames: torch.Tensor) -> torch.Tensor:
        """
        Frame-level orthogonality penalty  (works with batch_size = 1).

        sync_frames, app_frames: [B, T, D]  (same D after concatenation)
        Returns: mean cos²(sync_frame_i, app_frame_i) over all B·T frames.
        """
        s = F.normalize(sync_frames.reshape(-1, sync_frames.shape[-1]), dim=-1)
        a = F.normalize(app_frames.reshape(-1, app_frames.shape[-1]),   dim=-1)
        return ((s * a).sum(dim=-1) ** 2).mean()

    # ── Forward (inference) ───────────────────────────────────────────────────

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor,
                mode: str = "causal") -> torch.Tensor:
        z_sync, z_app, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            return self._cls_score(self.causal_head, z_sync)
        elif mode == "spurious":
            return self._cls_score(self.spurious_head, z_app.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def predict_scores(self, video_feats: torch.Tensor, audio_feats: torch.Tensor,
                       mode: str = "causal") -> torch.Tensor:
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, domain_labels, _ = batch
        cls_labels    = cls_labels.long()     # [B], −1 for FAVC
        domain_labels = domain_labels.float() # [B], 0/1

        z_sync, z_app, comp = self._encode(video_feats, audio_feats)

        # Pool over time for adversarial heads
        z_sync_pool = z_sync.mean(1)   # [B, sync_dim]
        z_app_pool  = z_app.mean(1)    # [B, spurious_dim]

        # ── Domain losses (all clips — AV1M + FAVC) ──────────────────────────
        # Adversarial: Z_sync must NOT encode domain
        z_sync_rev   = grad_reverse(z_sync_pool, self.grl_alpha)
        d_score_sync = self.domain_head_c(z_sync_rev).squeeze(-1)
        loss_dadv    = F.binary_cross_entropy_with_logits(d_score_sync, domain_labels)

        # Discriminative: Z_app MUST encode domain (concentrate domain signal here)
        d_score_app = self.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Task + label-adversarial losses (AV1M clips only) ────────────────
        av1m_mask = (cls_labels >= 0)

        if av1m_mask.any():
            z_sync_av1m  = z_sync[av1m_mask]   # [N, T, sync_dim]
            z_app_av1m   = z_app[av1m_mask]    # [N, T, spurious_dim]
            lbl_av1m     = cls_labels[av1m_mask]
            z_app_pool_av1m = z_app_pool[av1m_mask]

            # Task detection (causal branch)
            score_task = self._cls_score(self.causal_head, z_sync_av1m)
            loss_task  = self._ce_loss(score_task, lbl_av1m)

            # Spurious monitoring probe (detached — not backpropagated)
            score_spu = self._cls_score(self.spurious_head, z_app_av1m.detach())
            loss_spu_probe = self._ce_loss(score_spu, lbl_av1m)

            # Orthogonality: Z_sync ⊥ Z_app per frame
            loss_orth = self._orth_loss(z_sync_av1m, z_app_av1m)

            # Label adversarial on Z_app: appearance branch must NOT predict label
            z_app_rev   = grad_reverse(z_app_pool_av1m, self.grl_alpha)
            l_score_app = self.label_head_s(z_app_rev).squeeze(-1)
            loss_ladv   = F.binary_cross_entropy_with_logits(
                l_score_app, lbl_av1m.float()
            )
        else:
            # Batch contains only FAVC clips — skip task losses
            loss_task      = torch.tensor(0.0, device=video_feats.device)
            loss_orth      = torch.tensor(0.0, device=video_feats.device)
            loss_ladv      = torch.tensor(0.0, device=video_feats.device)
            loss_spu_probe = torch.tensor(0.0, device=video_feats.device)

        loss = (
              loss_task
            + self.lambda_orth * loss_orth
            + self.lambda_dadv * loss_dadv
            + self.lambda_ddis * loss_ddis
            + self.lambda_ladv * loss_ladv
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_dadv",      loss_dadv,      **log_kw)
        self.log("train_loss_ddis",      loss_ddis,      **log_kw)
        self.log("train_loss_ladv",      loss_ladv,      **log_kw)
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

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
