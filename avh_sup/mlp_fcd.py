"""
AVH-Align A5 — FCD-Inspired Feature Causality Decomposition
=============================================================

Each modality's AV-HuBERT feature (1024-dim per frame) is decomposed into
three components via CCD + URD modules (following NeurIPS 2025 FCD paper):

  synergistic  s^m   — cross-modal shared, task-relevant signal
                        aligned across modalities via Sinkhorn divergence (L_SDA)
                        ≈ shared AV consistency cue, temporal synchrony
  unique       u^m   — modality-specific, task-relevant cues
                        constrained via classification proxy (L_MI)
                        ≈ visual rendering artefacts / audio synthesis patterns
  redundant    r^m   — modality-specific noise / spurious shortcuts
                        r = h - u  (residual after extracting unique from h)
                        pushed to retain modality identity but NOT label signal (L_Dis)
                        ≈ compression noise, background, identity bias

Architecture:
  z^v, z^a [B, T, 1024]
      │
      ├─ CCD_v : F_s_v → s_v [B,T,syn_dim]
      │          F_h_v → h_v [B,T,spec_dim]
      ├─ CCD_a : F_s_a → s_a [B,T,syn_dim]
      │          F_h_a → h_a [B,T,spec_dim]
      │
      ├─ URD_v : h_v → u_v [B,T,spec_dim],  r_v = h_v - u_v
      ├─ URD_a : h_a → u_a [B,T,spec_dim],  r_a = h_a - u_a
      │
      ├─ causal_head    ( [s_v, u_v, s_a, u_a] → 1 )   ← primary inference
      └─ redundant_head ( [r_v, r_a]           → 1 )   ← probe (detached)

Losses:
  L_task  — CrossEntropy on causal_head([s_v, u_v, s_a, u_a])
  L_MI    — CE on unique_head_v(mean u_v) + CE on unique_head_a(mean u_a)
             (proxy for maximising MI(unique; label), works with batch_size=1)
  L_SDA   — Debiased Sinkhorn divergence between proj(mean s_v) and proj(mean s_a)
             (reduces to L2 for batch_size=1, works correctly)
  L_Dis   — CE(modality_head([r_v_pooled, r_a_pooled]) → {0,1})
             (keeps redundant space modality-discriminative; works with batch_size=1)
  L_orth  — ||cos(u^m, r^m)||²  per modality  (u ⊥ r geometric constraint)

Overall: L = L_task + λ_mi·L_MI + λ_sda·L_SDA + λ_dis·L_Dis + λ_orth·L_orth

Eval modes (compatible with existing eval scripts):
  mode="causal"    → causal_head([s_v, u_v, s_a, u_a])   (primary)
  mode="spurious"  → redundant_head([r_v, r_a])           (probe, target ≈ 0.5 AUC)
  mode="full"      → same as causal (for A5, full = causal)

Reference:
  "Plug-and-play Feature Causality Decomposition for Multimodal Representation
   Learning", NeurIPS 2025.
"""

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
#  Shared utility: standard classification MLP
# ─────────────────────────────────────────────────────────────────────────────

def _make_mlp(input_dim: int) -> nn.Sequential:
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


# ─────────────────────────────────────────────────────────────────────────────
#  CCD: Causality Components Decomposition  (one instance per modality)
# ─────────────────────────────────────────────────────────────────────────────

class CCD(nn.Module):
    """
    Split unimodal feature z into:
      s ∈ R^{syn_dim}   — modality-invariant / synergistic
      h ∈ R^{spec_dim}  — modality-specific   (passed to URD)
    """

    def __init__(self, feat_dim: int, syn_dim: int, spec_dim: int):
        super().__init__()
        self.F_s = nn.Sequential(
            nn.Linear(feat_dim, syn_dim),
            nn.LayerNorm(syn_dim),
            nn.ReLU(),
        )
        self.F_h = nn.Sequential(
            nn.Linear(feat_dim, spec_dim),
            nn.LayerNorm(spec_dim),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor):
        """z: [B, T, feat_dim]  →  (s [B,T,syn_dim], h [B,T,spec_dim])"""
        return self.F_s(z), self.F_h(z)


# ─────────────────────────────────────────────────────────────────────────────
#  URD: Unique-Redundant Decompose  (one instance per modality)
# ─────────────────────────────────────────────────────────────────────────────

class URD(nn.Module):
    """
    From modality-specific h ∈ R^{spec_dim} extract:
      u ∈ R^{spec_dim}  — unique / task-relevant
      r = h − u          — redundant / task-agnostic

    Architecture: two-layer MLP for u, residual for r.
    SVD regularisation (clamp singular values of the first linear layer to ≥ 1e-5)
    can be applied periodically via apply_svd_regularization(), following the
    "measure-preserving bijective" requirement from the FCD paper.
    """

    def __init__(self, spec_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(spec_dim, spec_dim, bias=True)
        self.norm = nn.LayerNorm(spec_dim)
        self.fc2  = nn.Linear(spec_dim, spec_dim, bias=True)

    def forward(self, h: torch.Tensor):
        """h: [B, T, spec_dim]  →  (u [B,T,spec_dim], r [B,T,spec_dim])"""
        u = self.fc2(F.relu(self.norm(self.fc1(h))))
        r = h - u
        return u, r

    @torch.no_grad()
    def apply_svd_regularization(self, min_sv: float = 1e-5):
        """
        Clamp near-zero singular values of fc1 weight matrix to min_sv.
        Prevents information collapse: if a singular value → 0, the corresponding
        direction of h becomes invisible to u, landing entirely in r.
        Applied periodically (every N steps) to keep overhead small.
        """
        W = self.fc1.weight.data.float()                                   # [D, D] — cast to fp32 (SVD not supported in fp16)
        U, S, Vt = torch.linalg.svd(W, full_matrices=True)
        S_clamped = S.clamp(min=min_sv)
        # U is [D, D] (full), Vt is [D, D]; S has length min(D, D) = D
        self.fc1.weight.data = ((U * S_clamped.unsqueeze(0)) @ Vt).to(self.fc1.weight.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  SDA projection  (shared MLP, one instance for both modalities)
# ─────────────────────────────────────────────────────────────────────────────

class SDA(nn.Module):
    """
    Shared linear projection used in Synergistic Distribution Alignment.
    Parameter sharing forces both modalities' synergistic embeddings into
    a common space before measuring distribution divergence.
    """

    def __init__(self, syn_dim: int, sda_dim: int):
        super().__init__()
        self.proj = nn.Linear(syn_dim, sda_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: [B, syn_dim]  →  L2-normalised [B, sda_dim]."""
        return F.normalize(self.proj(s), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  Debiased Sinkhorn divergence  (pure PyTorch, no external OT library)
# ─────────────────────────────────────────────────────────────────────────────

def _sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """
    Debiased Sinkhorn divergence  S_ε(x, y) ≥ 0:
        S_ε(x, y) = W_ε(x,y) − ½·W_ε(x,x) − ½·W_ε(y,y)
    where W_ε is entropy-regularised OT cost with uniform marginals.

    x, y: [N, D] — empirical point clouds.  Cost = squared L2 / D.

    For N=1 (batch_size=1), degenerates to  S_ε = ‖x − y‖²/D,
    which is a valid L2 alignment loss.
    """

    def _weps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        N, M   = a.shape[0], b.shape[0]
        D      = a.shape[1]
        C      = torch.cdist(a, b).pow(2) / D               # [N, M], cost matrix
        lp     = torch.full((N,), -math.log(N), device=a.device)  # log-uniform
        lq     = torch.full((M,), -math.log(M), device=b.device)
        f      = torch.zeros(N, device=a.device)
        g      = torch.zeros(M, device=b.device)
        for _ in range(n_iter):
            # Standard log-domain Sinkhorn: full replacement (not += )
            f = eps * (lp - torch.logsumexp((-C   + g.unsqueeze(0)) / eps, dim=1))
            g = eps * (lq - torch.logsumexp((-C.T + f.unsqueeze(0)) / eps, dim=1))
        P  = torch.exp((-C + f.unsqueeze(1) + g.unsqueeze(0)) / eps)
        return (P * C).sum()

    return _weps(x, y) - 0.5 * _weps(x, x) - 0.5 * _weps(y, y)


# ─────────────────────────────────────────────────────────────────────────────
#  A5 Lightning module: AVH_FCD
# ─────────────────────────────────────────────────────────────────────────────

class AVH_FCD(L.LightningModule):
    """
    A5 — FCD-inspired three-way decomposition for AV deepfake detection.

    Batch format (same as A1/A2):
        (video_feats [B,T,1024], audio_feats [B,T,1024], cls_labels [B], paths)

    Training runs with batch_size=1 (variable-length clips), so all losses are
    designed to work correctly for B=1:
      L_task  — CE on causal_head([s_v, u_v, s_a, u_a])           → works at B=1
      L_MI    — CE(unique_head_v(pool u_v)) + CE(unique_head_a())  → works at B=1
      L_SDA   — Sinkhorn(s_v_proj, s_a_proj)  (= L2 at B=1)       → works at B=1
      L_Dis   — CE(modality_head([r_v,r_a]) → {0:video,1:audio})   → 2 samples/step
      L_orth  — cosine-alignment penalty  u ⊥ r                    → works at B=1
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()

        hp = config.get("model_hparams", {})

        feat_dim = int(hp.get("feat_dim",  1024))
        syn_dim  = int(hp.get("syn_dim",    256))   # synergistic dim per modality
        spec_dim = int(hp.get("spec_dim",   256))   # modality-specific dim per modality
        sda_dim  = int(hp.get("sda_dim",    128))   # SDA projection dim
        mi_dim   = int(hp.get("mi_dim",     128))   # unique classification head hidden

        # Derived: causal_repr = [s_v, u_v, s_a, u_a], spurious_repr = [r_v, r_a]
        causal_dim   = (syn_dim + spec_dim) * 2     # (256+256)*2 = 1024  ← same as A1-A4
        spurious_dim = spec_dim * 2                 # 256*2 = 512

        # ── CCD: two MLP branches per modality ───────────────────────────────
        self.ccd_v = CCD(feat_dim, syn_dim, spec_dim)
        self.ccd_a = CCD(feat_dim, syn_dim, spec_dim)

        # ── URD: unique extraction per modality ───────────────────────────────
        self.urd_v = URD(spec_dim)
        self.urd_a = URD(spec_dim)

        # ── SDA: shared projection for synergistic alignment ─────────────────
        self.sda = SDA(syn_dim, sda_dim)

        # ── L_MI proxy: small classification heads on unique features ─────────
        # Train unique_head_v and unique_head_a to predict fake/real from u.
        # Maximising MI(u; y) via a classification proxy converges stably at B=1.
        self.unique_head_v = nn.Sequential(
            nn.Linear(spec_dim, mi_dim),
            nn.LayerNorm(mi_dim),
            nn.ReLU(),
            nn.Linear(mi_dim, 1),
        )
        self.unique_head_a = nn.Sequential(
            nn.Linear(spec_dim, mi_dim),
            nn.LayerNorm(mi_dim),
            nn.ReLU(),
            nn.Linear(mi_dim, 1),
        )

        # ── L_Dis: modality discrimination head on redundant features ─────────
        # Predicts which modality (0=video, 1=audio) from a pooled r embedding.
        # Trained without GRL so the encoder KEEPS modality-specific information
        # in r.  Combined with L_MI (task signal flows only to u), this forces
        # r to encode modality fingerprint but NOT label signal.
        self.modality_head = nn.Sequential(
            nn.Linear(spec_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # ── Primary classification heads ──────────────────────────────────────
        self.causal_head    = _make_mlp(causal_dim)
        self.redundant_head = _make_mlp(spurious_dim)   # probe — encoder always detached

        # ── Loss weights & hyperparameters ─────────────────────────────────────
        self.lambda_mi   = float(hp.get("lambda_mi",   1.0))
        self.lambda_sda  = float(hp.get("lambda_sda",  0.5))
        self.lambda_dis  = float(hp.get("lambda_dis",  1.0))
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.sda_eps     = float(hp.get("sda_eps",     0.05))
        self.sda_n_iter  = int  (hp.get("sda_n_iter",  50))
        self.lr          = float(hp.get("lr",          1e-3))

        # ── Validation accumulators ───────────────────────────────────────────
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, video_feats: torch.Tensor, audio_feats: torch.Tensor):
        """
        Decompose AV features into three components per modality.

        Returns:
          causal_repr   [B, T, (syn_dim+spec_dim)*2]  = cat(s_v, u_v, s_a, u_a)
          spurious_repr [B, T, spec_dim*2]             = cat(r_v, r_a)
          comp (dict)   per-modality components, each [B, T, dim]
        """
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
    def _cls_score(head: nn.Sequential, repr_: torch.Tensor) -> torch.Tensor:
        """head(repr_) → LogSumExp pool over time → clip-level score [B]."""
        per_frame = head(repr_)[..., 0]              # [B, T]
        return torch.logsumexp(per_frame, dim=-1)    # [B]

    @staticmethod
    def _ce_loss(score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([-score, score], dim=1)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _orth_loss(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Cosine-alignment penalty  ||cos(u, r)||²  averaged over all (B×T) tokens.
        Encourages unique and redundant subspaces to be orthogonal.
        """
        u_n = F.normalize(u.reshape(-1, u.shape[-1]), dim=-1)
        r_n = F.normalize(r.reshape(-1, r.shape[-1]), dim=-1)
        return ((u_n * r_n).sum(dim=-1) ** 2).mean()

    # ── Forward (inference) ───────────────────────────────────────────────────

    def forward(self, video_feats, audio_feats, mode: str = "causal"):
        """
        Inference entry-point.

        mode="causal"   / "full" → causal_head([s_v, u_v, s_a, u_a])
        mode="spurious"          → redundant_head([r_v, r_a])  (probe, detach)
        """
        causal_repr, spurious_repr, _ = self._encode(video_feats, audio_feats)
        if mode in ("causal", "full"):
            return self._cls_score(self.causal_head, causal_repr)
        elif mode == "spurious":
            # Probe only — never updates encoder
            return self._cls_score(self.redundant_head, spurious_repr.detach())
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'causal', 'spurious', or 'full'.")

    def predict_scores(self, video_feats, audio_feats, mode: str = "causal"):
        return self.forward(video_feats, audio_feats, mode=mode)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, cls_labels, _ = batch
        cls_labels = cls_labels.long()

        causal_repr, spurious_repr, comp = self._encode(video_feats, audio_feats)
        s_v, u_v, r_v = comp["s_v"], comp["u_v"], comp["r_v"]
        s_a, u_a, r_a = comp["s_a"], comp["u_a"], comp["r_a"]

        # ── L_task: CE on task-related representation  [s_v, u_v, s_a, u_a] ──
        score_task = self._cls_score(self.causal_head, causal_repr)
        loss_task  = self._ce_loss(score_task, cls_labels)

        # ── L_MI proxy: classify fake/real from unique features separately ────
        # Pool unique embeddings over time → [B, spec_dim], then classify.
        # This is a practical lower-bound proxy for maximising MI(u; y) at B=1.
        score_uv = self._cls_score(self.unique_head_v, u_v)
        score_ua = self._cls_score(self.unique_head_a, u_a)
        loss_mi  = self._ce_loss(score_uv, cls_labels) + self._ce_loss(score_ua, cls_labels)

        # ── L_SDA: Sinkhorn divergence between synergistic distributions ──────
        # Mean-pool synergistic features over time, apply shared projection.
        s_v_proj = self.sda(s_v.mean(1))    # [B, sda_dim], L2-normalised
        s_a_proj = self.sda(s_a.mean(1))    # [B, sda_dim], L2-normalised
        loss_sda = _sinkhorn_divergence(
            s_v_proj, s_a_proj,
            eps=self.sda_eps,
            n_iter=self.sda_n_iter,
        )

        # ── L_Dis: modality discrimination on redundant space ────────────────
        # For each training step (B=1): stack [r_v_pooled, r_a_pooled] → 2 samples.
        # modality_head learns to tell video from audio using r → encoder preserves
        # modality-specific identity in r without encoding label signal.
        r_v_pool = r_v.mean(1)   # [B, spec_dim]
        r_a_pool = r_a.mean(1)   # [B, spec_dim]
        r_stack  = torch.cat([r_v_pool, r_a_pool], dim=0)   # [2B, spec_dim]
        mod_score = self.modality_head(r_stack).squeeze(-1)  # [2B]
        mod_labels = torch.cat([
            torch.zeros(r_v_pool.shape[0], dtype=torch.float32, device=r_v_pool.device),
            torch.ones( r_a_pool.shape[0], dtype=torch.float32, device=r_a_pool.device),
        ])
        loss_dis = F.binary_cross_entropy_with_logits(mod_score, mod_labels)

        # ── L_orth: orthogonality  u ⊥ r  per modality ───────────────────────
        loss_orth = 0.5 * (self._orth_loss(u_v, r_v) + self._orth_loss(u_a, r_a))

        # ── Redundant probe update (diagnostic only, encoder always detached) ─
        score_spu      = self._cls_score(self.redundant_head, spurious_repr.detach())
        loss_spu_probe = self._ce_loss(score_spu, cls_labels)

        # ── Apply SVD regularisation to URD weight matrices ───────────────────
        # Every 50 steps; clamps near-zero singular values to ≥ 1e-5.
        if self.global_step % 50 == 0:
            self.urd_v.apply_svd_regularization()
            self.urd_a.apply_svd_regularization()

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (
            loss_task
            + self.lambda_mi   * loss_mi
            + self.lambda_sda  * loss_sda
            + self.lambda_dis  * loss_dis
            + self.lambda_orth * loss_orth
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",           loss,           **log_kw)
        self.log("train_loss_task",      loss_task,      **log_kw)
        self.log("train_loss_mi",        loss_mi,        **log_kw)
        self.log("train_loss_sda",       loss_sda,       **log_kw)
        self.log("train_loss_dis",       loss_dis,       **log_kw)
        self.log("train_loss_orth",      loss_orth,      **log_kw)
        self.log("train_loss_spu_probe", loss_spu_probe, **log_kw)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_scores = {"full": [], "causal": [], "spurious": []}
        self._val_labels = []

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        with torch.no_grad():
            s_causal   = self.forward(video_feats, audio_feats, mode="causal")
            s_spurious = self.forward(video_feats, audio_feats, mode="spurious")
        self._val_scores["causal"].append(s_causal.cpu().numpy())
        self._val_scores["spurious"].append(s_spurious.cpu().numpy())
        self._val_scores["full"].append(s_causal.cpu().numpy())   # full = causal for A5
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

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
