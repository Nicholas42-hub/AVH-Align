"""
E2E Training — A41: A40 + Z_s Label Adversarial (use_adv=True)
================================================================

**Motivation: Z_s still carries full task info in A39/A40**

  A39/A40 label probe results:
    Z_s label probe (in-domain)  = 0.999
    Z_s label probe (OOD FAVC)   = 0.990

  This means Z_s and Z_c are REDUNDANT — both independently encode
  fake/real signal. True causal/spurious separation requires Z_s to be
  task-irrelevant. The original A1/A2 design had this:

    A1/A2: adv_head(GRL(Z_s)) → expels label from Z_s

  A38/A39/A40 dropped use_adv=False, so Z_s was free to learn task signal.

**A41: all four losses together**

  L = L_task                          (Z_c classification, AV1M only)
    + lambda_adv    * L_adv_zs        (GRL: force Z_s task-irrelevant)
    + lambda_ladv_ds * L_ladv_ds      (label-aware domain pull into Z_s)
    + lambda_mean_zc * L_mean_zc      (Z_c mean alignment, domain-invariant)
    + lambda_orth    * L_orth         (optional: Z_c ⊥ Z_s on AV1M only)

  L_adv_zs: CE(adv_head(GRL(Z_s)), cls_labels)  — AV1M only (cls_label >= 0)
    GRL reverses gradient from adv_head → encoder pushes Z_s AWAY from
    label-discriminative directions. (lambda=10.0, grl_alpha=5.0, same as A2)

  L_ladv_ds and L_mean_zc are unchanged from A40.

**Tension between L_adv_zs and L_ladv_ds:**

  L_adv_zs says: Z_s should NOT predict cls (task-irrelevant)
  L_ladv_ds says: Z_s SHOULD predict domain (domain-aware)

  These are compatible if domain differences are NOT explained by cls:
    - AV1M has both real and fake clips (cls=0 and cls=1)
    - FAVC has both real and fake clips
    - Within-class domain differences (real-AV1M vs real-FAVC, fake-AV1M vs fake-FAVC)
      are NOT correlated with cls → Z_s can encode domain without encoding label
  The label-aware conditioning in L_ladv_ds (conditioning on y_hat_prob)
  explicitly targets within-class domain differences, making the two losses
  cooperative rather than conflicting.

**Expected outcome:**
  Z_c: domain-invariant (mean=0 difference) + task-relevant
  Z_s: domain-aware (domain pull) + task-irrelevant (adv)
  → True causal/spurious separation

**Success criteria:**
  domain_auc_zc_raw       < 0.65   (mean alignment working)
  domain_auc_zc_centered  < 0.10
  domain_auc_zs_raw       > 0.80   (domain pull working)
  val_auc_spurious        < 0.60   (Z_s not task-informative — NEW)
  val_auc_causal          > 0.95   (task not hurt)

  python train_e2e_a41.py --config_path configs/A41_e2e.yaml
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

sys.path.insert(0, os.path.dirname(__file__))
from avhubert_wrapper import AVHubertWrapper
from mlp_causal_ablation import AVH_Causal_Ablation
from mlp_sad_a8 import _make_adv_head
from datasets_e2e import (
    AV1M_E2E_FullPathDataset,
    FakeAVCeleb_E2E_Dataset,
    e2e_collate_fn,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Domain label wrapper ──────────────────────────────────────────────────────

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


class BalancedDomainSampler(Sampler):
    """50/50 domain-balanced sampler — guarantees both domains in every batch."""

    def __init__(self, domain_labels, batch_size: int, drop_last: bool = True):
        self.domain_labels = np.array(domain_labels, dtype=np.int32)
        self.batch_size    = batch_size
        self.drop_last     = drop_last
        self.idx0 = np.where(self.domain_labels == 0)[0]
        self.idx1 = np.where(self.domain_labels == 1)[0]
        self._n_per_domain = batch_size // 2

    def __iter__(self):
        rng = np.random.default_rng()
        idx0 = rng.permutation(self.idx0)
        idx1 = rng.permutation(self.idx1)
        n = max(len(idx0), len(idx1))
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
        n_batches = n // self._n_per_domain
        return n_batches * self.batch_size


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cond_domain_head(in_dim: int) -> nn.Sequential:
    """Label-aware domain head: input = concat(z_s_pool, y_hat_prob) [in_dim+1]."""
    return nn.Sequential(
        nn.Linear(in_dim + 1, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


# ── E2E Lightning Module ──────────────────────────────────────────────────────

class E2EModelA41(L.LightningModule):
    """
    A41: A40 + Z_s label adversarial (use_adv=True).

    Full loss:
      L_task        : CE on Z_c (AV1M only)
      L_adv_zs      : CE(adv_head(GRL(Z_s)), cls_labels)  — expels label from Z_s
      L_ladv_ds     : label-aware BCE on Z_s               — routes domain INTO Z_s
      L_mean_zc     : MSE between Z_c domain means         — closes Z_c mean gap
      L_orth        : optional orthogonality on (Z_c, Z_s) for AV1M clips only

    The adv head and GRL are the same as in the original A1/A2 design
    (lambda_adv=10.0, grl_alpha=5.0), accessed via self.head.adv_head and
    self.head.grl (instantiated when use_adv=True in AVH_Causal_Ablation).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        hp = config.get("model_hparams", {})

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = AVHubertWrapper(
            ckpt_path         = hp["avhubert_ckpt"],
            avhubert_root     = hp.get("avhubert_root", ""),
            n_unfreeze_layers = int(hp.get("n_unfreeze_layers", -1)),
        )

        # ── Causal/spurious head (use_adv=True, use_sg=True) ─────────────────
        # use_adv=True instantiates adv_head + grl inside AVH_Causal_Ablation.
        # use_sg=True restricts classification to Z_c only.
        abl = config.setdefault("ablation", {})
        abl["use_adv"]    = True
        abl["use_sg"]     = True
        abl["use_domain"] = False
        self.head = AVH_Causal_Ablation(config=config)

        # ── Dimensions ────────────────────────────────────────────────────────
        proj_dim     = int(hp.get("proj_dim", 512))
        proj_dim_spu = int(hp.get("proj_dim_spurious", proj_dim))
        spurious_dim = proj_dim_spu * 2      # concat(v_s, a_s)

        # ── Label-aware domain pull on Z_s (from A39) ────────────────────────
        self.cond_domain_head_s = _make_cond_domain_head(spurious_dim)

        # ── Hyperparams ───────────────────────────────────────────────────────
        self.lambda_adv      = float(hp.get("lambda_adv",      10.0))
        self.lambda_ladv_ds  = float(hp.get("lambda_ladv_ds",   1.0))
        self.lambda_mean_zc  = float(hp.get("lambda_mean_zc",   5.0))
        self.lambda_orth     = float(hp.get("lambda_orth",      0.0))

        self._val_scores: list = []
        self._val_labels: list = []
        self._val_spu_scores: list = []
        self.domain_probe_loader = None

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        hp = self.config.get("model_hparams", {})
        lr_encoder   = float(hp.get("lr_encoder",     5e-6))
        lr_head      = float(hp.get("lr",             1e-4))
        lr_domain    = float(hp.get("lr_domain_pred", lr_head))
        weight_decay = float(hp.get("weight_decay",   1e-4))

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params    = list(self.head.parameters())
        ladv_params    = list(self.cond_domain_head_s.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr_encoder},
            {"params": head_params,    "lr": lr_head},
            {"params": ladv_params,    "lr": lr_domain},
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
        domain_labels = domain_labels.long()

        # ── Encode → Z_c and Z_s ──────────────────────────────────────────────
        causal_repr, spurious_repr, _ = self.head._encode(video_feats, audio_feats)
        z_c_pool = causal_repr.mean(1)    # [B, causal_dim]
        z_s_pool = spurious_repr.mean(1)  # [B, spurious_dim]

        valid_mask = (cls_labels >= 0)    # AV1M clips only

        # ── 1. Task loss on Z_c (AV1M only) ──────────────────────────────────
        if valid_mask.any():
            logits_per_frame = self.head.causal_head(causal_repr[valid_mask])[..., 0]
            score_task       = torch.logsumexp(logits_per_frame, dim=-1)
            logits2          = torch.stack([-score_task, score_task], dim=1)
            loss_task        = F.cross_entropy(logits2, cls_labels[valid_mask])
        else:
            loss_task = torch.zeros(1, device=video_feats.device).squeeze()

        # ── 1b. Optional Z_c ⊥ Z_s orthogonality (AV1M only) ─────────────────
        if self.lambda_orth > 0 and valid_mask.any():
            loss_orth = self.head._orth_loss(
                causal_repr[valid_mask],
                spurious_repr[valid_mask],
            )
        else:
            loss_orth = torch.zeros(1, device=video_feats.device).squeeze()

        # ── 2. Z_s label adversarial (NEW vs A40) ─────────────────────────────
        # adv_head tries to predict cls from GRL(Z_s).
        # GRL reverses gradient → encoder pushes Z_s AWAY from label directions.
        # Only AV1M clips have real cls labels (cls_label >= 0).
        if valid_mask.any():
            z_s_grl      = self.head.grl(spurious_repr[valid_mask])  # GRL on full [B_v, T, D]
            adv_per_frame = self.head.adv_head(z_s_grl)[..., 0]      # [B_v, T]
            adv_score     = torch.logsumexp(adv_per_frame, dim=-1)    # [B_v]
            adv_logits2   = torch.stack([-adv_score, adv_score], dim=1)
            loss_adv_zs   = F.cross_entropy(adv_logits2, cls_labels[valid_mask])
        else:
            loss_adv_zs = torch.zeros(1, device=video_feats.device).squeeze()

        # ── 3. Label-aware domain pull on Z_s (from A39/A40) ─────────────────
        with torch.no_grad():
            logits_all = self.head.causal_head(causal_repr)[..., 0]  # [B, T]
            score_all  = torch.logsumexp(logits_all, dim=-1)          # [B]
            y_hat_prob = torch.sigmoid(score_all)                     # [B], stop-grad

        cond_input   = torch.cat([z_s_pool, y_hat_prob.unsqueeze(-1)], dim=-1)
        s_logits     = self.cond_domain_head_s(cond_input).squeeze(-1)
        loss_ladv_ds = F.binary_cross_entropy_with_logits(s_logits, domain_labels.float())

        # ── 4. Z_c mean alignment (from A40) ─────────────────────────────────
        mask_d0 = (domain_labels == 0)
        mask_d1 = (domain_labels == 1)
        if mask_d0.any() and mask_d1.any():
            mean_d0 = z_c_pool[mask_d0].mean(0)
            mean_d1 = z_c_pool[mask_d1].mean(0)
            loss_mean_zc = F.mse_loss(mean_d0, mean_d1)
        else:
            loss_mean_zc = torch.zeros(1, device=video_feats.device).squeeze()

        # ── Total loss ────────────────────────────────────────────────────────
        loss = (
            loss_task
            + self.lambda_adv     * loss_adv_zs
            + self.lambda_ladv_ds * loss_ladv_ds
            + self.lambda_mean_zc * loss_mean_zc
            + self.lambda_orth    * loss_orth
        )

        log_kw = dict(on_step=True, on_epoch=False)
        self.log("train_loss",          loss.item(),         **log_kw)
        self.log("train_loss_task",     loss_task.item(),    **log_kw)
        self.log("train_loss_orth",     loss_orth.item(),    **log_kw)
        self.log("train_loss_adv_zs",   loss_adv_zs.item(),  **log_kw)
        self.log("train_loss_ladv_ds",  loss_ladv_ds.item(), **log_kw)
        self.log("train_loss_mean_zc",  loss_mean_zc.item(), **log_kw)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 500 == 0:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_val = outputs["loss"].item()
            elif outputs is not None:
                loss_val = float(outputs)
            else:
                loss_val = float("nan")
            total = self.trainer.estimated_stepping_batches
            print(
                f"[Step {self.global_step:06d}/{int(total)}] "
                f"loss={loss_val:.4f}  epoch={self.current_epoch}",
                flush=True,
            )

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return
        video_raw, audio_raw, cls_labels, _, _ = batch
        video_feats, audio_feats = self.encoder(video_raw, audio_raw)
        causal_repr, spurious_repr, _ = self.head._encode(video_feats, audio_feats)
        scores_c = self.head._cls_score(self.head.causal_head, causal_repr)
        scores_s = self.head._cls_score(self.head.spurious_head, spurious_repr.detach())
        self._val_scores.append(scores_c.detach().cpu())
        self._val_spu_scores.append(scores_s.detach().cpu())
        self._val_labels.append(cls_labels.cpu())

    # ── Domain probe ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_domain_probe(self):
        """Logistic-regression domain probe on Z_c and Z_s."""
        self.eval()
        zc_list, zs_list, dom_list = [], [], []
        device = next(self.parameters()).device

        for batch in self.domain_probe_loader:
            if batch is None:
                continue
            video_raw, audio_raw, _, domain_labels, _ = batch
            video_raw = video_raw.to(device)
            audio_raw = audio_raw.to(device)
            video_feats, audio_feats = self.encoder(video_raw, audio_raw)
            causal_repr, spurious_repr, _ = self.head._encode(video_feats, audio_feats)
            zc_list.append(causal_repr.mean(1).cpu().numpy())
            zs_list.append(spurious_repr.mean(1).cpu().numpy())
            dom_list.extend(domain_labels.tolist())

        zc_all  = np.concatenate(zc_list)
        zs_all  = np.concatenate(zs_list)
        dom_all = np.array(dom_list)

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

        auc_zc_raw = _probe(zc_all, dom_all)
        auc_zs_raw = _probe(zs_all, dom_all)

        zc_centered = zc_all.copy()
        for d in [0, 1]:
            idx_d = (dom_all == d)
            if idx_d.sum() > 1:
                zc_centered[idx_d] -= zc_centered[idx_d].mean(0)
        auc_zc_centered = _probe(zc_centered, dom_all)

        self.train()
        return auc_zc_raw, auc_zc_centered, auc_zs_raw

    # ── Validation epoch end ──────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        if not self._val_scores:
            return
        scores   = torch.cat(self._val_scores).numpy()
        scores_s = torch.cat(self._val_spu_scores).numpy()
        labels   = torch.cat(self._val_labels).numpy()
        self._val_scores.clear()
        self._val_spu_scores.clear()
        self._val_labels.clear()
        mask = labels >= 0
        if mask.sum() < 2:
            return
        try:
            auc_c = roc_auc_score(labels[mask], scores[mask])
            auc_s = roc_auc_score(labels[mask], scores_s[mask])
        except Exception:
            auc_c, auc_s = 0.5, 0.5
        self.log("val_auc_causal",   auc_c, prog_bar=True)
        self.log("val_auc_spurious", auc_s, prog_bar=True)

        if self.domain_probe_loader is not None:
            auc_zc_raw, auc_zc_centered, auc_zs_raw = self._run_domain_probe()
            ep = self.current_epoch
            print(
                f"\n[Epoch {ep:02d}] val_auc_causal={auc_c:.4f}  val_auc_spurious={auc_s:.4f}"
                f"  |  domain probe:"
                f"  Z_c raw={auc_zc_raw:.4f} (want <0.65)"
                f"  Z_c centered={auc_zc_centered:.4f}"
                f"  Z_s raw={auc_zs_raw:.4f} (want >0.80)",
                flush=True,
            )
            self.log("domain_auc_zc_raw",      auc_zc_raw,      on_epoch=True, prog_bar=False)
            self.log("domain_auc_zc_centered", auc_zc_centered, on_epoch=True, prog_bar=False)
            self.log("domain_auc_zs_raw",      auc_zs_raw,      on_epoch=True, prog_bar=False)
        else:
            print(
                f"\n[Epoch {self.current_epoch:02d}] "
                f"val_auc_causal={auc_c:.4f}  val_auc_spurious={auc_s:.4f}",
                flush=True,
            )


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
    favc_real_only = bool(config.get("favc_real_only", True))
    use_balanced_sampler = bool(config.get("balanced_domain_sampler", False))

    av1m_base  = AV1M_E2E_FullPathDataset(train_csv, max_frames=max_frames)
    av1m_train = DomainLabeledDataset(av1m_base, domain_label=0)
    print(f"[load_data] AV1M train clips: {len(av1m_base)}", flush=True)

    favc_oversample = int(config.get("favc_oversample", 10))
    if favc_root and os.path.isdir(favc_root):
        favc_ds  = FakeAVCeleb_E2E_Dataset(
            favc_root, max_frames=max_frames, real_only=favc_real_only
        )
        favc_dom = DomainLabeledDataset(favc_ds, domain_label=1, cls_label_override=-1)
        favc_desc = "real-only" if favc_real_only else "all clips (real+fake)"
        print(f"[load_data] FAVC {favc_desc}: {len(favc_ds)} clips", flush=True)
        if favc_oversample > 1:
            favc_dom = ConcatDataset([favc_dom] * favc_oversample)
            print(f"[load_data] FAVC oversampled {favc_oversample}x → {len(favc_dom)}", flush=True)
        train_ds = ConcatDataset([av1m_train, favc_dom])
    else:
        print("[load_data] WARNING: favc_raw_root not found — no domain signal.", flush=True)
        train_ds = av1m_train

    av1m_val_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    val_ds        = DomainLabeledDataset(av1m_val_base, domain_label=0)
    print(f"[load_data] AV1M val clips: {len(av1m_val_base)}", flush=True)

    if use_balanced_sampler:
        def _collect_domain_labels(ds):
            if isinstance(ds, ConcatDataset):
                labels = []
                for sub in ds.datasets:
                    labels.extend(_collect_domain_labels(sub))
                return labels
            if isinstance(ds, DomainLabeledDataset):
                return [ds.domain_label] * len(ds)
            return [0] * len(ds)

        domain_labels = _collect_domain_labels(train_ds)
        n0 = sum(1 for l in domain_labels if l == 0)
        n1 = sum(1 for l in domain_labels if l == 1)
        print(f"[load_data] BalancedDomainSampler: {n0} AV1M, {n1} FAVC", flush=True)
        sampler = BalancedDomainSampler(domain_labels, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=e2e_collate_fn,
            pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=e2e_collate_fn,
            pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0),
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=e2e_collate_fn,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )

    # ── Domain probe loader ───────────────────────────────────────────────────
    n_probe   = 200
    rng_probe = np.random.default_rng(0)
    av1m_probe_base = AV1M_E2E_FullPathDataset(val_csv, max_frames=max_frames)
    av1m_probe = DomainLabeledDataset(
        Subset(av1m_probe_base, list(range(min(n_probe, len(av1m_probe_base))))),
        domain_label=0,
    )
    domain_probe_loader = None
    if favc_root and os.path.isdir(favc_root):
        favc_probe_base = FakeAVCeleb_E2E_Dataset(
            favc_root, max_frames=max_frames, real_only=favc_real_only
        )
        favc_probe_idx = rng_probe.choice(
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
            f"[load_data] Domain probe: {len(av1m_probe)} AV1M + {len(favc_probe)} FAVC",
            flush=True,
        )

    return train_loader, val_loader, domain_probe_loader


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 43))
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader, domain_probe_loader = load_data(config)
    model = E2EModelA41(config=config)
    model.domain_probe_loader = domain_probe_loader

    cb_cfg   = config.get("callbacks", {})
    ckpt_cfg = cb_cfg.get("ckpt_args", {})
    log_cfg  = cb_cfg.get("logger", {})

    logger = CSVLogger(
        save_dir = log_cfg.get("log_path", "outputs_A41_e2e/logs"),
        name     = "",
    )
    ckpt_cb = ModelCheckpoint(
        dirpath    = ckpt_cfg.get("ckpt_dir",  "outputs_A41_e2e/ckpts"),
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

    trainer = L.Trainer(
        max_epochs              = int(config.get("epochs", 30)),
        accelerator             = "gpu",
        devices                 = 1,
        precision               = config.get("precision", "16-mixed"),
        accumulate_grad_batches = int(config.get("accumulate_grad_batches", 4)),
        gradient_clip_val       = float(config.get("gradient_clip_val", 1.0)),
        callbacks               = [ckpt_cb, es_cb],
        logger                  = logger,
        log_every_n_steps       = 10,
        enable_progress_bar     = True,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best checkpoint: {ckpt_cb.best_model_path}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  Training done: {__import__('datetime').datetime.now().strftime('%a %d %b %Y %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
