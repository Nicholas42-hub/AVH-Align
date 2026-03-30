"""
E2E Training — A36: Conditional GRL + Subspace Split + Class-Conditional Alignment
====================================================================================

**Root cause recap (from A26-A35 analysis)**

  All prior alignment methods (mean, SWD, variance) attacked the z_sync_pool
  distribution without structural protection of label-relevant signal.  When λ
  is large enough to actually move the distribution, the task gradient competes
  and either the task collapses (λ too large) or the probe doesn't move (λ too
  small).  A plain GRL/DANN that sees the ENTIRE z_c can always find the fastest
  path to "fool" the discriminator — which often means erasing label info rather
  than domain info if they're entangled in the same directions.

**A36 strategy: structured separation with conditional adversarial**

  Two orthogonal insights drive A36:

  1. SUBSPACE SPLIT — give domain info a "legal slot":
       z_sync [B,T,512]
         ├─ proj_inv → z_c_inv [B,T,384]  ← task head + conditional GRL target
         └─ proj_aux → z_c_aux [B,T,128]  ← explicit domain attractor (positive BCE)
     The model can satisfy the GRL by routing domain information to z_c_aux
     instead of having to erase it entirely.  This avoids the "destroy label info
     to please the discriminator" shortcut.
     Initialization: proj_inv = first 384 dims of z_sync (warm start),
                     proj_aux = last  128 dims of z_sync (complementary).

  2. CONDITIONAL GRL — attack domain info that is NOT explained by the label:
       domain head input = concat(GRL(z_c_inv_pool), stopgrad(y_hat_prob))
     Plain GRL pushes z_c towards domain-indistinguishable, but if real/fake
     distributions differ across domains (likely), plain GRL fights label info.
     Conditional GRL asks: "given the predicted class, can you still tell domains
     apart?" — this is the targeted signal.  The model only needs to remove the
     domain residual *within each predicted class*.

  3. CLASS-CONDITIONAL CENTER ALIGNMENT — safe alternative to unconditional MMD:
       For each class c ∈ {0=real, 1=fake}:
         L += || mean(z_c_inv[domain=0, cls=c]) - mean(z_c_inv[domain=1, cls=c]) ||²_per_dim
     For FAVC clips (no ground truth label), model-predicted labels are used.
     This is a class-conditional version of the simple mean alignment, orthogonal
     to the conditional GRL.

  4. CURRICULUM λ — prevent early collapse:
       ep < warmup_start    : λ_grl = 0.0  (task head + proj_inv warm up first)
       warmup_start ≤ ep < (warmup_start + ramp_epochs): linear ramp
       ep ≥ (warmup_start + ramp_epochs): λ_grl = λ_max

**Architecture changes from A35:**
  - proj_inv: Linear(512→384, bias=False) — learnable subspace projection
  - proj_aux: Linear(512→128, bias=False) — complementary subspace
  - causal_head_inv: _make_cls_head(384)  — new task head (replaces original 512d)
  - cond_domain_head: MLP(385 → 128 → 64 → 1) — conditional GRL head
  - aux_domain_head:  _make_adv_head(128) — z_c_aux domain attractor

**Eval script compatibility (⚠ important)**
  eval_e2e_auc.py calls model.head.forward(v, a, mode="causal").
  eval_e2e_domain.py calls model.head._encode(v, a).
  Both are monkey-patched in __init__ to use the A36 subspace projection.
  The original methods are preserved as head._orig_encode / head._orig_forward.

**Full loss:**
  L = L_task
    + λ_grl(ep)       * L_cond_grl     [conditional GRL on z_c_inv_pool]
    + λ_dom_aux       * L_dom_aux       [positive domain BCE on z_c_aux_pool]
    + λ_cond_align    * L_cond_align    [class-conditional center MSE on z_c_inv_pool]
    + λ_ddis          * L_ddis          [domain BCE on z_app_pool — routing unchanged]
    + λ_orth_split    * L_orth_split    [cross-cov(z_c_inv_pool, z_c_aux_pool)]
    + λ_orth_app      * L_orth_app      [cross-cov(z_c_inv_pool, z_app_pool)]

**Expected dynamics:**
  ep 0-2: task AUC ≈ same as A9_e2e start (~1.0); proj_inv + causal_head_inv
          warm up; no domain adversarial pressure yet.
  ep 3-8: λ_grl ramp begins; cond_domain_head learns to distinguish domains
          within each class; GRL pushes z_c_inv to remove that residual signal.
  ep 8+:  z_c_inv domain AUC should fall; z_c_aux domain AUC stays ≥0.85 (attractor).

**Success criteria:**
  z_c_inv domain AUC < 0.65 (true invariance)
  z_c_aux domain AUC > 0.80 (domain attractor working)
  val_auc_causal    > 0.95  (task not killed)
  favc cross-domain AUC > 0.70 (OOD improvement)

  python train_e2e_a36.py --config_path configs/A36_e2e.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Sampler, Subset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_sad_a9 import AVH_SAD_A9
from mlp_sad_a8 import grad_reverse, _make_cls_head, _make_adv_head


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Utility losses ────────────────────────────────────────────────────────────

def cross_cov_orth_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    VICReg-style cross-covariance orthogonality (works for d1 ≠ d2).

    Penalises all entries of the normalised cross-covariance matrix
      C[i,j] = cov(z1[:,i], z2[:,j]) / sqrt(var(z1[:,i]) * var(z2[:,j]))
    so that z1 and z2 are linearly decorrelated across every pair of dimensions.

    Args:
        z1: [B, d1] — e.g. z_c_inv_pool
        z2: [B, d2] — e.g. z_c_aux_pool or z_app_pool
    Returns:
        scalar: mean squared cross-covariance (smaller = more orthogonal)
    """
    B = z1.shape[0]
    if B < 2:
        return torch.zeros(1, device=z1.device).squeeze()
    z1c = z1 - z1.mean(0)          # [B, d1]
    z2c = z2 - z2.mean(0)          # [B, d2]
    cross_cov = (z1c.T @ z2c) / (B - 1)   # [d1, d2]
    # Normalise by per-dim stds to get correlation matrix (scale-invariant)
    std1 = z1c.std(0).clamp(min=1e-8)      # [d1]
    std2 = z2c.std(0).clamp(min=1e-8)      # [d2]
    cross_corr = cross_cov / (std1.unsqueeze(1) * std2.unsqueeze(0))  # [d1, d2]
    return cross_corr.pow(2).mean()


def class_conditional_center_loss(
    z_inv_pool: torch.Tensor,           # [B, inv_dim]
    cls_labels: torch.Tensor,           # [B] — int64, -1 for unlabeled FAVC
    domain_labels_float: torch.Tensor,  # [B] — float, 0.=AV1M, 1.=FAVC
    y_hat_prob: torch.Tensor,           # [B] — float in [0,1], stop-gradded
) -> torch.Tensor:
    """
    Class-conditional center alignment.

    For each class c ∈ {0=real, 1=fake}:
      loss += || mean(z_c_inv[domain=0, cls=c]) - mean(z_c_inv[domain=1, cls=c]) ||²_per_dim

    For clips with known labels (cls_labels >= 0): use true label.
    For FAVC clips (cls_labels == -1): use binary threshold on y_hat_prob.

    Only terms where BOTH domains have ≥1 sample of class c contribute.

    This is a class-safe alternative to unconditional mean alignment:
    it only aligns distributions within the same predicted class, so
    it never forces real AV1M features ≈ fake FAVC features.
    """
    device = z_inv_pool.device
    domain_int = (domain_labels_float >= 0.5).long()   # [B]

    # Effective labels: true for AV1M, model-predicted for FAVC
    eff_labels = cls_labels.clone()
    unknown_mask = (cls_labels < 0)
    if unknown_mask.any():
        pred_labels = (y_hat_prob.detach() > 0.5).long()
        eff_labels[unknown_mask] = pred_labels[unknown_mask]

    centers: dict = {}
    for c in [0, 1]:
        for d in [0, 1]:
            mask = (eff_labels == c) & (domain_int == d)
            if mask.sum() >= 1:
                centers[(c, d)] = z_inv_pool[mask].mean(0)   # [inv_dim], has grad

    loss = torch.zeros(1, device=device).squeeze()
    n_terms = 0
    for c in [0, 1]:
        if (c, 0) in centers and (c, 1) in centers:
            diff = centers[(c, 0)] - centers[(c, 1)]
            loss = loss + diff.pow(2).mean()
            n_terms += 1

    if n_terms > 0:
        return loss / n_terms
    return loss


# ── Dataset utilities ─────────────────────────────────────────────────────────

class BalancedDomainSampler(Sampler):
    """Yields interleaved batches of domain-0 and domain-1 samples."""
    def __init__(self, domain_labels, batch_size: int, drop_last: bool = True):
        self.domain_labels = np.array(domain_labels, dtype=np.int32)
        self.batch_size    = batch_size
        self.drop_last     = drop_last
        self.idx0 = np.where(self.domain_labels == 0)[0]
        self.idx1 = np.where(self.domain_labels == 1)[0]
        self._n_per_domain = batch_size // 2

    def __iter__(self):
        rng  = np.random.default_rng()
        idx0 = rng.permutation(self.idx0)
        idx1 = rng.permutation(self.idx1)
        n    = max(len(idx0), len(idx1))
        idx0 = np.resize(idx0, n)
        idx1 = np.resize(idx1, n)
        n_batches = n // self._n_per_domain
        for i in range(n_batches):
            s = i * self._n_per_domain
            e = s + self._n_per_domain
            batch = np.concatenate([idx0[s:e], idx1[s:e]])
            rng.shuffle(batch)
            yield from batch.tolist()

    def __len__(self):
        n = max(len(self.idx0), len(self.idx1))
        return (n // self._n_per_domain) * self.batch_size


class DomainLabeledDataset(Dataset):
    def __init__(self, dataset, domain_label: int, cls_label_override=None):
        self.dataset            = dataset
        self.domain_label       = domain_label
        self.cls_label_override = cls_label_override

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, audio, cls_label, path = self.dataset[idx]
        if self.cls_label_override is not None:
            cls_label = self.cls_label_override
        return video, audio, cls_label, self.domain_label, path


class AugmentedDataset(Dataset):
    def __init__(self, dataset,
                 aug_video_noise: float = 0.3,
                 aug_brightness:  float = 0.5,
                 aug_flip_p:      float = 0.5,
                 aug_audio_noise: float = 0.05,
                 p: float = 0.9):
        self.dataset         = dataset
        self.aug_video_noise = aug_video_noise
        self.aug_brightness  = aug_brightness
        self.aug_flip_p      = aug_flip_p
        self.aug_audio_noise = aug_audio_noise
        self.p               = p

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        video, audio = item[0], item[1]
        rest = item[2:]
        if random.random() > self.p:
            return item
        sigma_v = random.uniform(0.0, self.aug_video_noise)
        if sigma_v > 0:
            video = video + torch.randn_like(video) * sigma_v
        if self.aug_brightness > 0:
            delta = random.uniform(-self.aug_brightness, self.aug_brightness)
            video = video + delta
        if random.random() < self.aug_flip_p:
            video = torch.flip(video, dims=[-1])
        sigma_a = random.uniform(0.0, self.aug_audio_noise)
        if sigma_a > 0:
            audio = audio + torch.randn_like(audio) * sigma_a
        return (video, audio) + rest


# ── E2E Lightning Module ───────────────────────────────────────────────────────

class E2EModelA36(L.LightningModule):
    """
    A36: A2_e2e (AVH_SAD_A9) fine-tuned from A9_e2e with:
      - Subspace split: z_sync → z_c_inv (384d) + z_c_aux (128d)
      - Conditional GRL on z_c_inv: domain head sees concat(GRL(z_c_inv), y_hat)
      - z_c_aux domain attractor: positive domain BCE (routing, no GRL)
      - Class-conditional center alignment on z_c_inv
      - Curriculum λ_grl: 0 → λ_max over warmup+ramp epochs
      - Cross-covariance orthogonality: z_c_inv ⊥ z_c_aux, z_c_inv ⊥ z_app
      - z_app domain disc (ddis) preserved from A35

    Eval script compatibility: monkey-patches head.forward and head._encode so
    eval_e2e_auc.py / eval_e2e_domain.py see the A36 subspace projection without
    any changes to those scripts.  See _patch_head_for_eval() for details.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        self.encoder = AVHubertWrapper(
            ckpt_path         = hp["avhubert_ckpt"],
            avhubert_root     = hp.get("avhubert_root", ""),
            n_unfreeze_layers = int(hp.get("n_unfreeze_layers", -1)),
        )
        self.head = AVH_SAD_A9(config=config)

        # ── Subspace projection layers ────────────────────────────────────────
        sync_dim = int(hp.get("sync_dim",  512))
        app_dim  = int(hp.get("app_dim",   256))
        self.inv_dim = int(hp.get("inv_dim", 384))
        self.aux_dim = int(hp.get("aux_dim", 128))
        spurious_dim = 2 * app_dim   # same as AVH_SAD_A9

        self.proj_inv = nn.Linear(sync_dim, self.inv_dim, bias=False)
        self.proj_aux = nn.Linear(sync_dim, self.aux_dim, bias=False)

        # Task head for z_c_inv (replaces head.causal_head which expected 512d)
        self.causal_head_inv = _make_cls_head(self.inv_dim)

        # Conditional GRL head: input = concat(z_c_inv_pool, y_hat_prob) = inv_dim+1
        self.cond_domain_head = nn.Sequential(
            nn.Linear(self.inv_dim + 1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # z_c_aux domain attractor (positive discrimination, no GRL)
        self.aux_domain_head = _make_adv_head(self.aux_dim)

        # z_c_aux label adversarial: GRL pushes z_c_aux AWAY from label info.
        # Directly addresses Risk 2: prevents z_c_aux from becoming a
        # task-relevant shortcut collector instead of a pure domain container.
        # Ideal outcome: z_c_aux label probe LOW, domain probe HIGH.
        self.label_head_aux = _make_adv_head(self.aux_dim)

        # ── Loss weights ──────────────────────────────────────────────────────
        self.lambda_cond_grl_max  = float(hp.get("lambda_cond_grl",      0.5))
        self.grl_warmup_start     = int(hp.get("grl_warmup_start",         3))
        self.grl_ramp_epochs      = int(hp.get("grl_ramp_epochs",          6))
        self.lambda_dom_aux       = float(hp.get("lambda_dom_aux",        1.0))
        self.lambda_label_adv_aux = float(hp.get("lambda_label_adv_aux",  0.3))
        self.lambda_cond_align    = float(hp.get("lambda_cond_align",     0.1))
        self.lambda_ddis          = float(hp.get("lambda_ddis",           1.0))
        self.lambda_orth_split    = float(hp.get("lambda_orth_split",     0.1))
        self.lambda_orth_app      = float(hp.get("lambda_orth_app",      0.05))
        self.grl_alpha            = float(hp.get("grl_alpha",             1.0))

        self._val_scores: list = []
        self._val_labels: list = []
        self.domain_probe_loader = None

        # ── Warm-start initialization for subspace projections ────────────────
        # proj_inv = "take first inv_dim dims of z_sync"
        # proj_aux = "take last aux_dim dims of z_sync"
        # These ensure task AUC is preserved at epoch 0.
        self._init_subspace_weights()

        # ── Patch head.forward / head._encode for eval script compatibility ────
        self._patch_head_for_eval()

    def _init_subspace_weights(self):
        """
        Orthonormal warm start: proj_inv = I[:inv_dim, :], proj_aux = I[sync_dim-aux_dim:, :].
        After loading A9_e2e checkpoint weights into sync_enc, the first
        self.inv_dim dims of z_sync will be copied into z_c_inv, preserving
        all task-discriminative information for the first epoch.
        """
        inv_dim  = self.inv_dim
        aux_dim  = self.aux_dim
        sync_dim = self.proj_inv.weight.shape[1]   # 512

        # proj_inv.weight shape: [inv_dim, sync_dim]
        nn.init.zeros_(self.proj_inv.weight)
        self.proj_inv.weight.data[:, :inv_dim] = torch.eye(inv_dim)

        # proj_aux.weight shape: [aux_dim, sync_dim]
        nn.init.zeros_(self.proj_aux.weight)
        self.proj_aux.weight.data[:, sync_dim - aux_dim:] = torch.eye(aux_dim)

    def _patch_head_for_eval(self):
        """
        Monkey-patch head.forward and head._encode so eval scripts that call
        model.head.forward(v, a, mode='causal') or model.head._encode(v, a)
        transparently use the A36 subspace projection.

        head._orig_encode  ← original AVH_SAD_A9._encode (raw z_sync, z_app)
        head._orig_forward ← original AVH_SAD_A9.forward
        head.forward       ← A36 version (uses proj_inv + causal_head_inv)
        head._encode       ← A36 version (returns z_c_inv instead of z_sync)

        These closures capture self by reference so all forward passes after
        checkpoint load use the correct (updated) projection weights.
        """
        _m = self   # capture by reference

        def _a36_forward(video_feats, audio_feats, mode="causal"):
            z_sync, z_app, extras = _m.head._orig_encode(video_feats, audio_feats)
            if mode == "causal":
                z_c_inv = _m.proj_inv(z_sync)                    # [B, T, inv_dim]
                per_frame = _m.causal_head_inv(z_c_inv)[..., 0]  # [B, T]
                return torch.logsumexp(per_frame, dim=-1)         # [B]
            elif mode == "spurious":
                return AVH_SAD_A9._cls_score(_m.head.spurious_head, z_app.detach())
            else:
                raise ValueError(f"Unknown mode: {mode!r}")

        def _a36_encode(video_feats, audio_feats):
            # Returns z_c_inv (NOT raw z_sync) so domain probe probes the right repr.
            z_sync, z_app, extras = _m.head._orig_encode(video_feats, audio_feats)
            z_c_inv = _m.proj_inv(z_sync)   # [B, T, inv_dim]
            return z_c_inv, z_app, extras

        # Store originals as instance attributes on head (bound methods)
        self.head._orig_encode  = self.head._encode
        self.head._orig_forward = self.head.forward
        # Install A36 versions
        self.head.forward = _a36_forward
        self.head._encode = _a36_encode

    # ── Score helper ──────────────────────────────────────────────────────────

    def _cls_score_inv(self, z_c_inv: torch.Tensor) -> torch.Tensor:
        """logsumexp-pooled causal score using causal_head_inv on z_c_inv [B,T,d]."""
        per_frame = self.causal_head_inv(z_c_inv)[..., 0]   # [B, T]
        return torch.logsumexp(per_frame, dim=-1)            # [B]

    # ── Curriculum λ for conditional GRL ─────────────────────────────────────

    def _grl_lambda(self) -> float:
        """
        Linearly ramps λ_grl from 0.0 to lambda_cond_grl_max.
          ep < warmup_start                  → 0.0
          warmup_start ≤ ep < warmup+ramp    → linear ramp
          ep ≥ warmup_start + ramp_epochs    → lambda_cond_grl_max
        """
        ep = self.current_epoch
        if ep < self.grl_warmup_start:
            return 0.0
        progress = min(1.0, (ep - self.grl_warmup_start) / max(1, self.grl_ramp_epochs))
        return self.lambda_cond_grl_max * progress

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp           = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder", 5e-6))
        lr_head      = float(hp.get("lr",         1e-4))
        weight_decay = float(hp.get("weight_decay", 1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]

        # Head params: base A9 modules + A36-specific modules
        # (Original causal_head in self.head won't receive gradients in A36,
        # but including it is harmless since its gradient will be zero.)
        head_params = (
            list(self.head.parameters())
            + list(self.proj_inv.parameters())
            + list(self.proj_aux.parameters())
            + list(self.causal_head_inv.parameters())
            + list(self.cond_domain_head.parameters())
            + list(self.aux_domain_head.parameters())
            + list(self.label_head_aux.parameters())
        )

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params,    "lr": lr_head},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val_auc_causal",
                "interval":  "epoch",
            },
        }

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        video_raw, audio_raw, cls_labels, domain_labels, _ = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)

        cls_labels    = cls_labels.long()
        domain_labels = domain_labels.float()

        # ── Encode: raw z_sync via original (unpatched) _encode ──────────────
        z_sync, z_app, _ = self.head._orig_encode(video_feats, audio_feats)

        # ── Subspace projections ──────────────────────────────────────────────
        z_c_inv = self.proj_inv(z_sync)    # [B, T, inv_dim]
        z_c_aux = self.proj_aux(z_sync)    # [B, T, aux_dim]

        z_c_inv_pool = z_c_inv.mean(1)     # [B, inv_dim]
        z_c_aux_pool = z_c_aux.mean(1)     # [B, aux_dim]
        z_app_pool   = z_app.mean(1)       # [B, spurious_dim=512]

        n0 = (domain_labels < 0.5).sum().item()
        n1 = (domain_labels >= 0.5).sum().item()

        # ── Task score for ALL clips (needed for y_hat conditioning) ──────────
        score_all  = self._cls_score_inv(z_c_inv)         # [B]
        y_hat_prob = torch.sigmoid(score_all).detach()    # [B], stop-grad

        # ── Task loss (AV1M clips only) ───────────────────────────────────────
        av1m_mask = (cls_labels >= 0)
        if av1m_mask.any():
            loss_task = AVH_SAD_A9._ce_loss(score_all[av1m_mask], cls_labels[av1m_mask])
        else:
            loss_task = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Conditional GRL on z_c_inv (curriculum-scheduled) ────────────────
        # domain head sees: concat(GRL(z_c_inv_pool), stopgrad(y_hat_prob))
        # Question: "given the predicted class, can you still predict domain?"
        # GRL gradient pushes sync_enc to produce domain-indistinguishable z_c_inv
        # WITHIN each predicted class.
        lambda_grl = self._grl_lambda()
        if lambda_grl > 0:
            z_c_inv_pool_grl = grad_reverse(z_c_inv_pool, self.grl_alpha)
            cond_input  = torch.cat(
                [z_c_inv_pool_grl, y_hat_prob.unsqueeze(-1)], dim=-1
            )  # [B, inv_dim+1]
            cond_logit  = self.cond_domain_head(cond_input).squeeze(-1)   # [B]
            loss_cond_grl = F.binary_cross_entropy_with_logits(
                cond_logit, domain_labels
            )
        else:
            loss_cond_grl = torch.zeros(1, device=video_feats.device).squeeze()

        # ── z_c_aux domain attractor (no GRL — positive discrimination) ───────
        # Encouraging z_c_aux to carry domain info gives the model a "safe exit":
        # it can route domain → z_c_aux rather than having to erase it entirely.
        #
        # CRITICAL — STOP-GRAD on z_sync for this loss:
        # Without stop-grad, dom_aux BCE propagates back through proj_aux → z_sync
        # → sync_enc, pushing the shared backbone toward domain-aware.
        # This directly counteracts the conditional GRL on z_c_inv (which pushes
        # the backbone toward domain-unaware). Stop-grad confines this loss to
        # training proj_aux + aux_domain_head only.
        z_c_aux_pool_sg = self.proj_aux(z_sync.detach().mean(1))   # [B, aux_dim]
        aux_logit    = self.aux_domain_head(z_c_aux_pool_sg).squeeze(-1)
        loss_dom_aux = F.binary_cross_entropy_with_logits(aux_logit, domain_labels)

        # ── z_c_aux label adversarial GRL ─────────────────────────────────────
        # Risk 2 mitigation: push z_c_aux to NOT predict labels.
        # Makes z_c_aux a clean domain-nuisance container rather than a
        # task-relevant shortcut absorber. Uses real z_c_aux_pool (WITH grad
        # through proj_aux and sync_enc) so sync_enc learns to not encode
        # label info in the z_c_aux subspace directions.
        if av1m_mask.any() and self.lambda_label_adv_aux > 0:
            z_c_aux_pool_rev   = grad_reverse(z_c_aux_pool[av1m_mask], self.grl_alpha)
            aux_label_logit    = self.label_head_aux(z_c_aux_pool_rev).squeeze(-1)
            loss_label_adv_aux = F.binary_cross_entropy_with_logits(
                aux_label_logit, cls_labels[av1m_mask].float()
            )
        else:
            loss_label_adv_aux = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Class-conditional center alignment ────────────────────────────────
        loss_cond_align = class_conditional_center_loss(
            z_c_inv_pool, cls_labels, domain_labels, y_hat_prob
        )

        # ── z_app domain discriminative loss (routing domain → spurious stream)
        d_score_app = self.head.domain_head_s(z_app_pool).squeeze(-1)
        loss_ddis   = F.binary_cross_entropy_with_logits(d_score_app, domain_labels)

        # ── Cross-covariance orthogonality (VICReg-style) ─────────────────────
        # z_c_inv ⊥ z_c_aux: encourages the two subspaces to be uncorrelated,
        # so domain info can be "absorbed" by z_c_aux without leaking back.
        loss_orth_split = cross_cov_orth_loss(z_c_inv_pool, z_c_aux_pool)
        # z_c_inv ⊥ z_app: weaker constraint, prevents domain bleed from z_app.
        loss_orth_app   = cross_cov_orth_loss(z_c_inv_pool, z_app_pool)

        # ── Spurious probe (monitoring only, no gradient to encoder) ──────────
        if av1m_mask.any():
            score_spu = AVH_SAD_A9._cls_score(
                self.head.spurious_head, z_app[av1m_mask].detach()
            )
            loss_spu_probe = AVH_SAD_A9._ce_loss(score_spu, cls_labels[av1m_mask])
        else:
            loss_spu_probe = torch.zeros(1, device=video_feats.device).squeeze()

        loss = (
              loss_task
            + lambda_grl                    * loss_cond_grl
            + self.lambda_dom_aux           * loss_dom_aux
            + self.lambda_label_adv_aux     * loss_label_adv_aux
            + self.lambda_cond_align        * loss_cond_align
            + self.lambda_ddis              * loss_ddis
            + self.lambda_orth_split        * loss_orth_split
            + self.lambda_orth_app          * loss_orth_app
        )

        log_kw = dict(on_step=False, on_epoch=True)
        self.log("train_loss",                  loss,                **log_kw)
        self.log("train_loss_task",             loss_task,           **log_kw)
        self.log("train_loss_cond_grl",         loss_cond_grl,       **log_kw)
        self.log("train_loss_dom_aux",          loss_dom_aux,        **log_kw)
        self.log("train_loss_label_adv_aux",    loss_label_adv_aux,  **log_kw)
        self.log("train_loss_cond_align",       loss_cond_align,     **log_kw)
        self.log("train_loss_ddis",         loss_ddis,         **log_kw)
        self.log("train_loss_orth_split",   loss_orth_split,   **log_kw)
        self.log("train_loss_orth_app",     loss_orth_app,     **log_kw)
        self.log("train_loss_spu_probe",    loss_spu_probe,    **log_kw)
        self.log("train_grl_lambda",        float(lambda_grl), **log_kw)
        self.log("train_n_domain0",         float(n0),         **log_kw)
        self.log("train_n_domain1",         float(n1),         **log_kw)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _domain, _paths = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        # head.forward is monkey-patched → uses causal_head_inv(proj_inv(z_sync))
        scores = self.head.forward(video_feats, audio_feats, mode="causal")
        self._val_scores.append(scores.detach().cpu())
        self._val_labels.append(cls_labels.cpu())

    # ── Domain probe ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_domain_probe(self):
        """
        Returns (auc_zinv_raw, auc_zinv_centered, auc_zaux_raw, auc_zsync_raw).

        Reports domain AUC (AV1M=0 vs FAVC=1) for:
          z_c_inv (PRIMARY target — should drop toward 0.5)
          z_c_aux (should stay high — domain attractor)
          z_sync  (reference — raw z_sync without subspace split)

        RAW    = uncentered logistic probe (tests true distributional similarity)
        CENTERED = after per-domain mean subtraction (tests covariance residual)
        """
        self.eval()
        zinv_list, zaux_list, zsync_list, dom_list, cls_list = [], [], [], [], []
        device = next(self.parameters()).device

        for batch in self.domain_probe_loader:
            if batch is None:
                continue
            video_raw, audio_raw, cls_labels_b, domain_labels, _ = batch
            video_raw = video_raw.to(device)
            audio_raw = audio_raw.to(device)

            video_feats, audio_feats = self.encoder(video_raw, audio_raw)

            # Use _orig_encode to get raw z_sync (before A36 subspace projection)
            z_sync, _, _ = self.head._orig_encode(video_feats, audio_feats)
            z_c_inv = self.proj_inv(z_sync)    # [B, T, inv_dim]
            z_c_aux = self.proj_aux(z_sync)    # [B, T, aux_dim]

            zinv_list.append(z_c_inv.mean(1).cpu().numpy())
            zaux_list.append(z_c_aux.mean(1).cpu().numpy())
            zsync_list.append(z_sync.mean(1).cpu().numpy())
            dom_list.extend(domain_labels.tolist())
            cls_list.extend(cls_labels_b.tolist())

        zinv_all  = np.concatenate(zinv_list)
        zaux_all  = np.concatenate(zaux_list)
        zsync_all = np.concatenate(zsync_list)
        dom_all   = np.array(dom_list)

        def _probe(X, y, n_folds=5):
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, C=1.0, solver="lbfgs"),
            )
            skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
            probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
            try:
                return roc_auc_score(y, probs)
            except Exception:
                return 0.5

        auc_zinv_raw  = _probe(zinv_all, dom_all)
        auc_zaux_raw  = _probe(zaux_all, dom_all)
        auc_zsync_raw = _probe(zsync_all, dom_all)

        # Centered z_c_inv: removes mean gap, probes covariance residual
        zinv_centered = zinv_all.copy()
        for d in [0, 1]:
            idx_d = (dom_all == d)
            if idx_d.sum() > 1:
                zinv_centered[idx_d] -= zinv_centered[idx_d].mean(0)
        auc_zinv_centered = _probe(zinv_centered, dom_all)

        # ── Label probes (AV1M real=0 / fake=1; FAVC clips have cls=-1) ───────
        # Risk 1: does z_c_inv still carry label info? (want HIGH)
        # Risk 2: does z_c_aux carry label info?       (want LOW)
        cls_all      = np.array(cls_list)
        av1m_np_mask = (cls_all >= 0)
        auc_zinv_label = 0.5
        auc_zaux_label = 0.5
        if av1m_np_mask.sum() >= 10 and len(np.unique(cls_all[av1m_np_mask])) > 1:
            auc_zinv_label = _probe(zinv_all[av1m_np_mask], cls_all[av1m_np_mask])
            auc_zaux_label = _probe(zaux_all[av1m_np_mask], cls_all[av1m_np_mask])

        self.train()
        return auc_zinv_raw, auc_zinv_centered, auc_zaux_raw, auc_zsync_raw, auc_zinv_label, auc_zaux_label

    # ── Validation epoch end ──────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        scores_local = torch.cat(self._val_scores)   # [N_local]  — keep as tensor
        labels_local = torch.cat(self._val_labels)   # [N_local]
        self._val_scores.clear()
        self._val_labels.clear()

        # Gather across all GPUs (no-op in single-GPU mode)
        if self.trainer.world_size > 1:
            scores_all = self.all_gather(scores_local)  # [world_size, N_local]
            labels_all = self.all_gather(labels_local)
            scores = scores_all.reshape(-1).cpu().numpy()
            labels = labels_all.reshape(-1).cpu().numpy()
        else:
            scores = scores_local.cpu().numpy()
            labels = labels_local.cpu().numpy()

        mask = labels >= 0
        if mask.sum() < 2:
            return
        try:
            auc = roc_auc_score(labels[mask], scores[mask])
        except Exception:
            auc = 0.5
        self.log("val_auc_causal", auc, prog_bar=True, sync_dist=False)

        # Domain probe: run only on rank-0 to avoid duplicate prints/inference
        if self.domain_probe_loader is not None and self.trainer.is_global_zero:
            zinv_raw, zinv_centered, zaux_raw, zsync_raw, zinv_label, zaux_label = self._run_domain_probe()
            ep = self.current_epoch
            lam = self._grl_lambda()
            print(
                f"\n[Epoch {ep:02d}|λ_grl={lam:.3f}] 6-metric probe report:"
                f"\n  z_c_inv domain RAW      = {zinv_raw:.4f}  ← PRIMARY target (want <0.65)"
                f"\n  z_c_inv domain CENTERED = {zinv_centered:.4f}  (covariance residual)"
                f"\n  z_c_inv label           = {zinv_label:.4f}  ← task signal (want HIGH)"
                f"\n  z_c_aux domain RAW      = {zaux_raw:.4f}  ← attractor check (want >0.80)"
                f"\n  z_c_aux label           = {zaux_label:.4f}  ← risk check (want LOW)"
                f"\n  z_sync  domain RAW      = {zsync_raw:.4f}  (reference)"
                f"\n  task AUC                = {auc:.4f}",
                flush=True,
            )
            self.log("domain_auc_zinv_raw",      zinv_raw,      on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("domain_auc_zinv_centered", zinv_centered, on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("label_auc_zinv",           zinv_label,    on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("domain_auc_zaux_raw",      zaux_raw,      on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("label_auc_zaux",           zaux_label,    on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("domain_auc_zsync_raw",     zsync_raw,     on_epoch=True, prog_bar=False, sync_dist=False)
            self.log("train_grl_lambda_ep",      lam,           on_epoch=True, prog_bar=False, sync_dist=False)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(config: dict):
    dp          = config["data_paths"]
    train_csv   = dp["e2e_train_csv"]
    val_csv     = dp["e2e_val_csv"]
    favc_root   = dp.get("favc_raw_root", "")
    hp          = config.get("model_hparams", {})
    max_frames  = int(hp.get("max_frames", 150))
    batch_size  = int(config.get("batch_size", 8))
    num_workers = int(config.get("num_workers", 6))

    aug_cfg         = config.get("augmentation", {})
    do_augment      = bool(aug_cfg.get("enabled", True))
    aug_video_noise = float(aug_cfg.get("video_noise",  0.3))
    aug_brightness  = float(aug_cfg.get("brightness",   0.5))
    aug_flip_p      = float(aug_cfg.get("flip_p",       0.5))
    aug_audio_noise = float(aug_cfg.get("audio_noise",  0.05))
    aug_p           = float(aug_cfg.get("p",            0.9))

    from datasets_e2e import (
        AV1M_E2E_FullPathDataset,
        FakeAVCeleb_E2E_Dataset,
        e2e_collate_fn,
    )

    # AV1M train (domain=0)
    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    if do_augment:
        av1m_train = AugmentedDataset(
            av1m_train,
            aug_video_noise=aug_video_noise,
            aug_brightness=aug_brightness,
            aug_flip_p=aug_flip_p,
            aug_audio_noise=aug_audio_noise,
            p=aug_p,
        )
    print(f"[load_data] AV1M train: {len(av1m_base)} clips", flush=True)

    # FAVC ALL clips (domain=1, real+fake; cls_label_override=-1 → no task loss)
    favc_oversample = int(config.get("favc_oversample", 3))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
        favc_dom = DomainLabeledDataset(favc_ds, domain_label=1, cls_label_override=-1)
        print(f"[load_data] FAVC all clips (real+fake): {len(favc_ds)}", flush=True)
        if do_augment:
            favc_dom = AugmentedDataset(
                favc_dom,
                aug_video_noise=aug_video_noise,
                aug_brightness=aug_brightness,
                aug_flip_p=aug_flip_p,
                aug_audio_noise=aug_audio_noise,
                p=aug_p,
            )
        if favc_oversample > 1:
            favc_dom = ConcatDataset([favc_dom] * favc_oversample)
            print(f"[load_data] FAVC oversampled {favc_oversample}× → {len(favc_dom)} clips", flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — training without FAVC.", flush=True)
        train_ds = av1m_train

    # AV1M val (domain=0, for validation AUC)
    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val: {len(av1m_val_base)} clips", flush=True)

    def _collect_domain_labels(ds):
        if isinstance(ds, ConcatDataset):
            labels = []
            for sub in ds.datasets:
                labels.extend(_collect_domain_labels(sub))
            return labels
        elif isinstance(ds, AugmentedDataset):
            return _collect_domain_labels(ds.dataset)
        elif isinstance(ds, DomainLabeledDataset):
            return [ds.domain_label] * len(ds)
        else:
            return [0] * len(ds)

    domain_labels = _collect_domain_labels(train_ds)
    n0 = sum(1 for l in domain_labels if l == 0)
    n1 = sum(1 for l in domain_labels if l == 1)
    print(f"[load_data] BalancedDomainSampler: {n0} domain-0 (AV1M), {n1} domain-1 (FAVC)", flush=True)

    sampler = BalancedDomainSampler(domain_labels, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, drop_last=False, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    # Domain probe loader (200 AV1M val + 200 FAVC, no augmentation)
    n_probe    = 200
    rng_probe  = np.random.default_rng(0)
    av1m_probe_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    av1m_probe = DomainLabeledDataset(
        Subset(av1m_probe_base, list(range(min(n_probe, len(av1m_probe_base))))),
        domain_label=0,
    )
    domain_probe_loader = None
    if favc_root and os.path.isdir(favc_root):
        favc_probe_base = FakeAVCeleb_E2E_Dataset(favc_root, max_frames=max_frames, real_only=False)
        favc_probe_idx  = rng_probe.choice(
            len(favc_probe_base), size=min(n_probe, len(favc_probe_base)), replace=False
        ).tolist()
        favc_probe = DomainLabeledDataset(
            Subset(favc_probe_base, favc_probe_idx),
            domain_label=1, cls_label_override=-1,
        )
        probe_ds = ConcatDataset([av1m_probe, favc_probe])
        domain_probe_loader = DataLoader(
            probe_ds, batch_size=8, shuffle=False,
            num_workers=2, collate_fn=e2e_collate_fn,
        )
        print(
            f"[load_data] Domain probe loader: {len(av1m_probe)} AV1M + "
            f"{len(favc_probe)} FAVC (no augmentation)", flush=True
        )

    return train_loader, val_loader, domain_probe_loader


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 43))
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader, domain_probe_loader = load_data(config)
    model = E2EModelA36(config=config)
    model.domain_probe_loader = domain_probe_loader

    # ── Load from A9_e2e checkpoint ───────────────────────────────────────────
    # Keys in A9_e2e checkpoint: "encoder.*", "head.*"
    # These map cleanly to E2EModelA36's self.encoder and self.head (AVH_SAD_A9).
    # New A36 modules (proj_inv, proj_aux, causal_head_inv, etc.) are missing
    # from the checkpoint → strict=False allows this, they keep _init_subspace_weights.
    hp = config.get("model_hparams", {})
    finetune_ckpt = hp.get("finetune_ckpt", "")
    if finetune_ckpt and os.path.isfile(finetune_ckpt):
        print(f"[A36] Loading weights from {finetune_ckpt} (strict=False)", flush=True)
        ckpt_data  = torch.load(finetune_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        n_miss = len(missing)
        n_unex = len(unexpected)
        print(f"[A36] Missing keys (new A36 modules, expected): {n_miss}", flush=True)
        if missing and n_miss <= 20:
            for k in missing:
                print(f"       missing: {k}", flush=True)
        print(f"[A36] Unexpected keys (removed modules, expected): {n_unex}", flush=True)
    else:
        print("[A36] WARNING: no finetune_ckpt found — starting from scratch.", flush=True)

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger",    {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A36_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A36_e2e/ckpts"),
        filename   = "model-{epoch:02d}-{val_auc_causal:.4f}",
        monitor    = ckpt_cfg.get("metric", "val_auc_causal"),
        mode       = ckpt_cfg.get("mode",   "max"),
        save_top_k = 1,
        save_last  = True,
    )
    es_cfg = config.get("early_stopping", {})
    es_cb  = EarlyStopping(
        monitor  = es_cfg.get("metric",   "val_auc_causal"),
        mode     = es_cfg.get("mode",     "max"),
        patience = int(es_cfg.get("patience", 10)),
        verbose  = True,
    )

    accum = int(config.get("accumulate_grad_batches", 4))
    clip  = float(config.get("gradient_clip_val", 1.0))
    prec  = config.get("precision", "16-mixed")

    num_gpus  = int(config.get("num_gpus",  1))
    num_nodes = int(config.get("num_nodes", 1))
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if (num_gpus > 1 or num_nodes > 1)
        else "auto"
    )
    trainer = L.Trainer(
        max_epochs              = int(config.get("epochs", 30)),
        accelerator             = "gpu",
        devices                 = num_gpus,
        num_nodes               = num_nodes,
        strategy                = strategy,
        precision               = prec,
        accumulate_grad_batches = accum,
        gradient_clip_val       = clip,
        callbacks               = [ckpt_cb, es_cb],
        logger                  = logger,
        log_every_n_steps       = 1,     # log after every step (SLURM — no progress bar)
        enable_progress_bar     = False,  # suppress tqdm junk in SLURM logs
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Training done: {__import__('datetime').datetime.now().strftime('%a %d %b %Y %H:%M:%S')}", flush=True)
    print(f"  Best ckpt in:  {ckpt_cfg.get('ckpt_dir', 'outputs_A36_e2e/ckpts')}/", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
