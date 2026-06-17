"""
AVHubertWrapper
===============
Wraps the fairseq AV-HuBERT model so it can be used as a trainable
(or partially frozen) feature encoder inside a Lightning module.

Key behaviours
--------------
* Loads the fairseq AV-HuBERT model from a fairseq checkpoint file
  (self_large_vox_433h.pt or similar).
* By default freezes ALL layers and only unfreezes the top
  `n_unfreeze_layers` transformer encoder layers + layer-norm.
  Set n_unfreeze_layers=0 to reproduce the previous frozen-encoder
  baseline; set n_unfreeze_layers=-1 to unfreeze everything.
* forward(video, audio) → (video_feats, audio_feats) each [B, T, 1024]
  matching the shape of the pre-extracted .npz features used before.

Usage
-----
    wrapper = AVHubertWrapper(
        ckpt_path = "av_hubert/avhubert/self_large_vox_433h.pt",
        avhubert_root = "av_hubert/avhubert",
        n_unfreeze_layers = 4,   # unfreeze top 4 transformer layers
    )
    v_feats, a_feats = wrapper(video_tensor, audio_tensor)
    # v_feats, a_feats: [B, T, 1024]
"""

import os
import sys
import torch
import torch.nn as nn


def _add_avhubert_to_path(avhubert_root: str):
    # avhubert_root is the *avhubert/* package dir; we need its parent
    # on sys.path so that `import avhubert.hubert_pretraining` resolves
    # relative imports (from .hubert_dataset import ...) correctly.
    parent = os.path.dirname(avhubert_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Also keep the package dir itself for direct sub-module imports
    if avhubert_root not in sys.path:
        sys.path.insert(0, avhubert_root)


class AVHubertWrapper(nn.Module):
    """
    Thin wrapper around a fairseq AV-HuBERT model that exposes
    separate video and audio feature streams.

    Parameters
    ----------
    ckpt_path : str
        Path to the fairseq AV-HuBERT checkpoint
        (e.g. self_large_vox_433h.pt).
    avhubert_root : str
        Path to the avhubert Python package directory
        (the one containing hubert_pretraining.py).
    n_unfreeze_layers : int
        Number of top transformer encoder layers to unfreeze for
        fine-tuning.
          0  → fully frozen (identical to pre-extracted baseline)
         -1  → fully unfrozen
          N  → top N encoder layers + final layer-norm unfrozen
    """

    def __init__(self,
                 ckpt_path: str,
                 avhubert_root: str,
                 n_unfreeze_layers: int = 4):
        super().__init__()

        _add_avhubert_to_path(avhubert_root)
        # hubert_pretraining.py sets DBG=True only when len(sys.argv)==1 and
        # uses bare (non-relative) imports in that branch.  When DBG=False it
        # uses `from .hubert_dataset import …` which fails for top-level imports.
        # Temporarily truncate sys.argv so DBG=True during the imports.
        _saved_argv = sys.argv[:]
        sys.argv = sys.argv[:1]
        try:
            import hubert_pretraining  # noqa: F401 — registers tasks
            import hubert              # noqa: F401 — registers models
            import hubert_asr          # noqa: F401 — registers more models
        finally:
            sys.argv = _saved_argv
        from fairseq import checkpoint_utils

        models, _, task = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
        model = models[0]
        # For fine-tuned checkpoints the model is wrapped in an ASR encoder;
        # unwrap to get the core AV-HuBERT model.
        if hasattr(model, "decoder"):
            model = model.encoder.w2v_model

        self.model = model
        self.feat_dim = 1024  # AV-HuBERT large output dim

        # ── Store task config for transforms ──────────────────────────────────
        self.image_crop_size = task.cfg.image_crop_size   # 88
        self.image_mean      = task.cfg.image_mean        # 0.421
        self.image_std       = task.cfg.image_std         # 0.165

        # ── Freeze / unfreeze ─────────────────────────────────────────────────
        self._apply_freeze(n_unfreeze_layers)

    # ── Freeze logic ──────────────────────────────────────────────────────────

    def _apply_freeze(self, n_unfreeze_layers: int):
        # Start fully frozen
        for p in self.model.parameters():
            p.requires_grad_(False)

        if n_unfreeze_layers == 0:
            return  # fully frozen baseline

        if n_unfreeze_layers == -1:
            # Unfreeze everything
            for p in self.model.parameters():
                p.requires_grad_(True)
            return

        # Unfreeze top N transformer encoder layers.
        # AV-HuBERT stores encoder layers in model.encoder.layers
        # (a ModuleList of TransformerSentenceEncoderLayer).
        encoder_layers = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layers"):
            encoder_layers = self.model.encoder.layers
        elif hasattr(self.model, "layers"):
            encoder_layers = self.model.layers

        if encoder_layers is not None:
            total = len(encoder_layers)
            unfreeze_from = max(0, total - n_unfreeze_layers)
            for layer in encoder_layers[unfreeze_from:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

        # Also unfreeze final layer norm
        for name in ("layer_norm", "final_layer_norm", "encoder.layer_norm"):
            parts = name.split(".")
            m = self.model
            try:
                for part in parts:
                    m = getattr(m, part)
                for p in m.parameters():
                    p.requires_grad_(True)
            except AttributeError:
                pass

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.model.parameters())
        print(f"[AVHubertWrapper] unfrozen top {n_unfreeze_layers} encoder layers "
              f"— trainable params: {n_trainable:,} / {n_total:,}", flush=True)

        # Ensure the unfrozen sub-modules are in train mode so that
        # Dropout / BatchNorm (if any) behave correctly during training.
        # Frozen sub-modules stay in eval mode to save memory + be deterministic.
        self.model.eval()   # start all in eval
        if encoder_layers is not None:
            for layer in encoder_layers[unfreeze_from:]:
                layer.train()
        # Also set the final layer norm to train if unfrozen
        for name in ("layer_norm", "final_layer_norm", "encoder.layer_norm"):
            parts = name.split(".")
            m = self.model
            try:
                for part in parts:
                    m = getattr(m, part)
                if any(p.requires_grad for p in m.parameters()):
                    m.train()
            except AttributeError:
                pass

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                video: torch.Tensor,
                audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        video : FloatTensor [B, T, 1, H, W]
            Grayscale mouth-ROI frames, normalised.
        audio : FloatTensor [B, T, 104]
            Stacked log-filterbank features, layer-normed.

        Returns
        -------
        video_feats : FloatTensor [B, T, 1024]
        audio_feats : FloatTensor [B, T, 1024]
        """
        B, T = video.shape[:2]

        # AV-HuBERT expects:
        #   video: [B, 1, T, H, W]   (channel dim = 1 for grayscale)
        #   audio: [B, 104, T]       (freq dim first, then time)
        vid_in = video.permute(0, 2, 1, 3, 4)       # [B, 1, T, H, W]
        aud_in = audio.permute(0, 2, 1)              # [B, 104, T]

        # extract_finetune returns (features, padding_mask)
        # features: [B, T, D]
        v_feats, _ = self.model.extract_finetune(
            {"video": vid_in, "audio": None}, None, None
        )
        a_feats, _ = self.model.extract_finetune(
            {"video": None, "audio": aud_in}, None, None
        )
        return v_feats, a_feats   # both [B, T, 1024]
