"""
End-to-end datasets that load *raw* preprocessed video/audio files
instead of pre-extracted .npz features.

Each item returns the raw modality inputs required by AVHubertWrapper:
    video_frames : FloatTensor [T, H, W]   (grayscale, normalised)
    audio_feats  : FloatTensor [T_a, 104]  (log-filterbank, layer-normed)
    label        : int
    path         : str  (relative path for bookkeeping)

Compatible with the existing DomainLabeledDataset wrapper in
train_test_causal_ablation.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from python_speech_features import logfbank
from torch.utils.data import Dataset

# ── AV-HuBERT utilities (from av_hubert repo) ─────────────────────────────────
# Add av_hubert/avhubert/ to sys.path so utils.py (no relative imports) can be
# imported directly.  The parent dir is NOT added here to avoid triggering
# avhubert/__init__.py which causes duplicate fairseq model registrations.
_AVHUBERT_PKG  = os.path.join(os.path.dirname(__file__), "..", "av_hubert", "avhubert")
_AVHUBERT_ROOT = os.path.dirname(_AVHUBERT_PKG)   # av_hubert/
for _p in (_AVHUBERT_ROOT, _AVHUBERT_PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import utils as avhubert_utils  # load_video, Compose, Normalize, CenterCrop

FPS = 25
SAMPLE_RATE = 16000
STACK_ORDER_AUDIO = 4   # must match AV-HuBERT pretraining config
IMAGE_CROP_SIZE = 88    # standard for AV-HuBERT
IMAGE_MEAN = 0.421
IMAGE_STD  = 0.165


def build_video_transform():
    return avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)),
        avhubert_utils.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


def load_audio_feats(wav_path: str) -> torch.Tensor:
    """Load wav → stacked log-filterbank feats [T_a, 104], layer-normed."""
    wav, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    feats = logfbank(wav, samplerate=SAMPLE_RATE).astype(np.float32)
    # pad to multiple of stack_order
    rem = len(feats) % STACK_ORDER_AUDIO
    if rem != 0:
        feats = np.concatenate(
            [feats, np.zeros((STACK_ORDER_AUDIO - rem, feats.shape[1]), dtype=feats.dtype)]
        )
    feats = feats.reshape(-1, STACK_ORDER_AUDIO * feats.shape[1])  # [T_a, 104]
    feats = torch.from_numpy(feats)
    with torch.no_grad():
        feats = F.layer_norm(feats, feats.shape[1:])
    return feats  # [T_a, 104]


def load_video_frames(roi_path: str, transform) -> torch.Tensor:
    """Load mouth-ROI mp4 → FloatTensor [T, H, W]."""
    frames = avhubert_utils.load_video(roi_path)   # np [T, H, W] uint8
    frames = transform(frames)                     # np [T, H, W] float32
    return torch.FloatTensor(frames)               # [T, H, W]


def _align_av(video: torch.Tensor, audio: torch.Tensor):
    """Trim video and audio to the same temporal length."""
    T_v = video.shape[0]
    T_a = audio.shape[0]
    T   = min(T_v, T_a)
    return video[:T], audio[:T]


# ── AV1M end-to-end dataset ────────────────────────────────────────────────────

class AV1M_E2E_Dataset(Dataset):
    """
    Loads raw `*_roi.mp4` + `*.wav` files from avd1m_preprocessed.

    Directory layout expected:
        raw_root/
            {split}/
                {speaker_id}/{video_id}/{clip_id}/
                    real_roi.mp4  +  real.wav             (label=0)
                    fake_video_fake_audio_roi.mp4  +  ...  (label=1)
                    ...

    The CSV (`csv_path`) maps relative .mp4 paths (without _roi suffix) to
    integer labels, matching the existing avh_sup/csv_metadata format.
    The raw_root for .mp4 path resolution uses the PREPROCESSED directory
    (avd1m_preprocessed/{split}/...) rather than the feature directory.
    """

    def __init__(self, raw_root: str, csv_path: str, split: str = "train",
                 max_frames: int = 200, return_manip_type: bool = False):
        self.raw_root          = os.path.join(raw_root, split)
        self.max_frames        = max_frames
        self.transform         = build_video_transform()
        self.return_manip_type = return_manip_type

        df = pd.read_csv(csv_path)
        has_manip = "manip_type" in df.columns
        # Keep only entries whose preprocessed files exist
        self.items = []
        for _, row in df.iterrows():
            rel        = row["path"]          # e.g. id00012/.../real.mp4
            label      = int(row["label"])
            manip_type = int(row["manip_type"]) if has_manip else -1
            stem  = rel[:-4]             # strip .mp4
            roi   = os.path.join(self.raw_root, stem + "_roi.mp4")
            wav   = os.path.join(self.raw_root, stem + ".wav")
            if os.path.isfile(roi) and os.path.isfile(wav):
                self.items.append((roi, wav, label, manip_type, rel))

        print(f"[AV1M_E2E_Dataset] split={split}  found {len(self.items)} clips "
              f"(dropped {len(df) - len(self.items)} missing)", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        roi_path, wav_path, label, manip_type, rel = self.items[idx]
        try:
            video = load_video_frames(roi_path, self.transform)  # [T, H, W]
            audio = load_audio_feats(wav_path)                   # [T_a, 104]
            video, audio = _align_av(video, audio)
            # cap length to avoid OOM
            if self.max_frames > 0:
                video = video[:self.max_frames]
                audio = audio[:self.max_frames]
        except Exception as e:
            print(f"[AV1M_E2E_Dataset] WARNING: failed to load {roi_path}: {e}", flush=True)
            # return a zero-length dummy — collate_fn handles this via skip
            video = torch.zeros(1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
            audio = torch.zeros(1, STACK_ORDER_AUDIO * 26)
        if self.return_manip_type:
            return video, audio, label, manip_type, rel
        return video, audio, label, rel


# ── FakeAVCeleb end-to-end dataset ────────────────────────────────────────────

_FAVC_REAL = "RealVideo-RealAudio"
_FAVC_FAKE = {"FakeVideo-RealAudio", "FakeVideo-FakeAudio", "RealVideo-FakeAudio"}

class FakeAVCeleb_E2E_Dataset(Dataset):
    """
    Loads raw `*_roi.mp4` + `*.wav` files from favc_preprocessed.

    Directory layout:
        raw_root/
            {category}/  (e.g. RealVideo-RealAudio, FakeVideo-FakeAudio, ...)
                {ethnicity}/{gender}/{speaker_id}/
                    {clip_id}_roi.mp4
                    {clip_id}.wav
    """

    def __init__(self, raw_root: str, fake_category: str = "all",
                 max_frames: int = 200, real_only: bool = False):
        self.max_frames = max_frames
        self.transform  = build_video_transform()

        self.items = []
        for cat in os.listdir(raw_root):
            cat_dir = os.path.join(raw_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            if cat == _FAVC_REAL:
                label = 0
            elif cat in _FAVC_FAKE:
                if real_only:
                    continue  # skip fake categories when real_only=True
                label = 1
                if fake_category != "all" and cat != fake_category:
                    continue
            else:
                continue  # unknown category

            # walk to find roi+wav pairs
            for dirpath, _, fnames in os.walk(cat_dir):
                rois = {f[:-8] for f in fnames if f.endswith("_roi.mp4")}
                wavs = {f[:-4]  for f in fnames if f.endswith(".wav")}
                for stem in rois & wavs:
                    roi = os.path.join(dirpath, stem + "_roi.mp4")
                    wav = os.path.join(dirpath, stem + ".wav")
                    rel = os.path.relpath(roi, raw_root)
                    self.items.append((roi, wav, label, rel))

        print(f"[FakeAVCeleb_E2E_Dataset] found {len(self.items)} clips", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        roi_path, wav_path, label, rel = self.items[idx]
        try:
            video = load_video_frames(roi_path, self.transform)
            audio = load_audio_feats(wav_path)
            video, audio = _align_av(video, audio)
            if self.max_frames > 0:
                video = video[:self.max_frames]
                audio = audio[:self.max_frames]
        except Exception as e:
            print(f"[FakeAVCeleb_E2E_Dataset] WARNING: failed to load {roi_path}: {e}", flush=True)
            video = torch.zeros(1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
            audio = torch.zeros(1, STACK_ORDER_AUDIO * 26)
        return video, audio, label, rel


# ── Collate function ───────────────────────────────────────────────────────────

def e2e_collate_fn(batch):
    """
    Pad variable-length sequences to the max length in the batch.
    Skips items with T < 2.

    Returns (4-tuple):
        video  : FloatTensor [B, T_max, 1, H, W]   (added channel dim)
        audio  : FloatTensor [B, T_max, 104]
        labels : LongTensor  [B]
        paths  : list[str]

    Returns (5-tuple) when items include a domain/manip label:
        video, audio, labels, domain_labels : LongTensor [B], paths
    """
    batch = [b for b in batch if b[0].shape[0] >= 2]
    if not batch:
        return None

    has_domain = len(batch[0]) == 5

    T_max_v = max(b[0].shape[0] for b in batch)
    T_max_a = max(b[1].shape[0] for b in batch)
    T_max   = min(T_max_v, T_max_a)  # they should match after _align_av

    videos, audios, labels, domain_labels, paths = [], [], [], [], []
    for item in batch:
        if has_domain:
            video, audio, label, domain_label, path = item
        else:
            video, audio, label, path = item
            domain_label = -1
        T = min(video.shape[0], audio.shape[0], T_max)
        # pad video
        pad_v = T_max - T
        v = video[:T]                                     # [T, H, W]
        v = v.unsqueeze(1)                                # [T, 1, H, W]
        if pad_v > 0:
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad_v))  # pad time dim
        # pad audio
        a = audio[:T]                                     # [T, 104]
        if pad_v > 0:
            a = F.pad(a, (0, 0, 0, pad_v))
        videos.append(v)
        audios.append(a)
        labels.append(label)
        domain_labels.append(domain_label)
        paths.append(path)

    if has_domain:
        return (
            torch.stack(videos),
            torch.stack(audios),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(domain_labels, dtype=torch.long),
            paths,
        )
    return (
        torch.stack(videos),            # [B, T_max, 1, H, W]
        torch.stack(audios),            # [B, T_max, 104]
        torch.tensor(labels, dtype=torch.long),
        paths,
    )

# ── Full-path dataset (generated by generate_e2e_csv.py) ──────────────────────

class AV1M_E2E_FullPathDataset(Dataset):
    """
    E2E dataset that reads a CSV with absolute file paths generated by
    generate_e2e_csv.py.  Enables using ALL available preprocessed clips
    from multiple base directories without needing a single root path.

    CSV columns: roi_path, wav_path, label, manip_type
    Returns 4-tuple (video, audio, label, path) compatible with
    DomainLabeledDataset wrapper.
    """

    def __init__(self, csv_path: str, max_frames: int = 150):
        import pandas as pd
        self.max_frames = max_frames
        self.transform  = build_video_transform()

        df = pd.read_csv(csv_path)
        self.items = []
        missing = 0
        for _, row in df.iterrows():
            roi = str(row["roi_path"])
            wav = str(row["wav_path"])
            if os.path.isfile(roi) and os.path.isfile(wav):
                self.items.append((
                    roi, wav,
                    int(row["label"]),
                    int(row.get("manip_type", -1)),
                ))
            else:
                missing += 1
        print(
            f"[AV1M_E2E_FullPathDataset] {csv_path}: "
            f"{len(self.items)} clips ({missing} missing)",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        roi_path, wav_path, label, _manip_type = self.items[idx]
        try:
            video = load_video_frames(roi_path, self.transform)  # [T, H, W]
            audio = load_audio_feats(wav_path)                   # [T_a, 104]
            video, audio = _align_av(video, audio)
            if self.max_frames > 0:
                video = video[:self.max_frames]
                audio = audio[:self.max_frames]
        except Exception as e:
            print(
                f"[AV1M_E2E_FullPathDataset] WARNING: failed {roi_path}: {e}",
                flush=True,
            )
            video = torch.zeros(1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
            audio = torch.zeros(1, STACK_ORDER_AUDIO * 26)
        return video, audio, label, roi_path