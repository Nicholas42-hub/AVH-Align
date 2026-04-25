"""
SpeechForensics evaluation on FakeAVCeleb test split (per-category AUC/AP).

Reproduces the SpeechForensics (NeurIPS 2024) cosine-similarity score on our
FAVC 70/30 test split using pre-extracted mouth crops and wav files.

Usage:
    python eval_speechforensics_favc.py \
        --ckpt   checkpoints/large_vox_iter5.pt \
        --output_dir results_speechforensics

The script reads:
  • csv_metadata/favc/test_split.csv      (columns: source, category, full_path)
  • data/favc_preprocessed/<category>/...<stem>_roi.mp4   (mouth crops)
  • data/favc_preprocessed/<category>/...<stem>.wav       (audio)

Category codes in CSV:
    A = RealVideo-RealAudio  (real)
    B = RealVideo-FakeAudio  (fake)
    C = FakeVideo-RealAudio  (fake)
    D = FakeVideo-FakeAudio  (fake)
"""

import argparse
import csv
import os
import sys
import tempfile
import json

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# ─── AV-HuBERT path setup ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
AVHUBERT_ROOT = os.path.join(BASE, "..", "av_hubert", "avhubert")
FAIRSEQ_ROOT  = os.path.join(BASE, "..", "av_hubert", "fairseq")

for p in [AVHUBERT_ROOT, os.path.dirname(AVHUBERT_ROOT), FAIRSEQ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from python_speech_features import logfbank  # noqa: E402
from argparse import Namespace               # noqa: E402


# ─── Image transforms (copied from SpeechForensics utils.py) ───────────────
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std
    def __call__(self, frames):
        return (frames - self.mean) / self.std

class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        dh = int(round((h - th) / 2.))
        dw = int(round((w - tw) / 2.))
        return frames[:, dh:dh+th, dw:dw+tw]


def load_video(path):
    """Load greyscale frames from a video file. Returns [T, H, W] uint8."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames, axis=0)          # [T, H, W]


def stacker(feats, stack_order):
    """Concatenate consecutive audio frames (identical to SpeechForensics)."""
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        feats = np.concatenate([feats, np.zeros([res, feat_dim], dtype=feats.dtype)], axis=0)
    return feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)


def calc_cos_dist(feat1, feat2, vshift=15):
    """
    Compute audio-visual cosine distance grid (SpeechForensics metric).
    Returns the max cosine similarity across temporal shifts as the score.
    Higher score → more synchronous → more likely real.
    """
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    if len(feat1) != len(feat2):
        idx = np.linspace(0, len(feat1) - 1, len(feat2), dtype=int)
        feat1 = feat1[idx.tolist()]
    win_size = vshift * 2 + 1
    feat2p = F.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(
            F.cosine_similarity(
                feat1[[i]].expand(win_size, -1),
                feat2p[i:i + win_size]
            ).cpu().numpy()
        )
    return np.asarray(dists)


# ─── Feature extractors ─────────────────────────────────────────────────────

def extract_visual_feature(model, task, video_path, max_length, device):
    transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])
    frames = load_video(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps > 0 and len(frames) > fps * max_length:
        frames = frames[:int(fps * max_length)]
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        feature, _ = model.extract_finetune(
            source={'video': frames, 'audio': None},
            padding_mask=None, output_layer=None,
        )
        feature = feature.squeeze(0)
    return feature


def extract_audio_feature(model, wav_path, max_length, tmp_dir, device):
    wav, sr = sf.read(wav_path)
    if len(wav) > sr * max_length:
        tmp_wav = os.path.join(tmp_dir, 'audio.wav')
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        sf.write(tmp_wav, wav[:sr * max_length], sr)
        wav_path = tmp_wav

    sample_rate, wav_data = wavfile.read(wav_path)
    assert sample_rate == 16_000 and wav_data.ndim == 1, \
        f"Expected 16kHz mono, got {sample_rate}Hz shape={wav_data.shape}"
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
    audio_feats = stacker(audio_feats, 4)
    audio_t = torch.FloatTensor(audio_feats).to(device)
    with torch.no_grad():
        audio_t = F.layer_norm(audio_t, audio_t.shape[1:])
        audio_t = audio_t.T.unsqueeze(0)          # [1, F, T]
        feature, _ = model.extract_finetune(
            source={'video': None, 'audio': audio_t},
            padding_mask=None, output_layer=None,
        )
        feature = feature.squeeze(0)
    return feature


def compute_score(model, task, roi_path, wav_path, max_length, tmp_dir, device):
    vis_feat = extract_visual_feature(model, task, roi_path, max_length, device)
    aud_feat = extract_audio_feature(model, wav_path, max_length, tmp_dir, device)
    dists = calc_cos_dist(vis_feat.cpu(), aud_feat.cpu())
    return float(dists.mean(axis=0).max())


# ─── AUC helpers ────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    'A': 'RealVideo-RealAudio',   # real
    'B': 'RealVideo-FakeAudio',   # fake
    'C': 'FakeVideo-RealAudio',   # fake
    'D': 'FakeVideo-FakeAudio',   # fake
}
FAKE_CATS = {'B', 'C', 'D'}


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    # Higher score = more real, so negate to get "fake score"
    return roc_auc_score(y_true, [-s for s in y_score])


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true, [-s for s in y_score])


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', default='checkpoints/large_vox_iter5.pt',
                   help='AV-HuBERT pretrained checkpoint (large_vox_iter5.pt)')
    p.add_argument('--csv', default='avh_sup/csv_metadata/favc/test_split.csv',
                   help='FAVC test split CSV (columns: source, category, full_path)')
    p.add_argument('--preprocessed_dir', default='data/favc_preprocessed',
                   help='Root dir of mouth crops + wav (stem_roi.mp4 / stem.wav)')
    p.add_argument('--output_dir', default='results_speechforensics',
                   help='Directory to write results JSON + per-clip CSV')
    p.add_argument('--max_length', type=int, default=50,
                   help='Maximum video length (seconds)')
    p.add_argument('--trimmed', action='store_true',
                   help='Use trimmed features (favc_features_trimmed naming)')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    from fairseq import checkpoint_utils, utils as fairseq_utils
    from fairseq.dataclass.configs import GenerationConfig

    user_dir = os.path.dirname(os.path.abspath(__file__))
    # Register avhubert task/model
    fairseq_utils.import_user_module(Namespace(user_dir=AVHUBERT_ROOT))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.ckpt}  (device={device})")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.ckpt])
    model = models[0]
    if hasattr(model, 'decoder'):
        print("Checkpoint: fine-tuned — using encoder.w2v_model")
        model = model.encoder.w2v_model
    else:
        print("Checkpoint: pre-trained (no decoder)")
    model.to(device)
    model.eval()

    # ── Read test split ─────────────────────────────────────────────────────
    rows = []
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"Test clips: {len(rows)}")

    tmp_dir = tempfile.mkdtemp()

    all_scores, all_labels, all_cats, all_paths = [], [], [], []

    for row in tqdm(rows, desc='Scoring'):
        cat_code   = row['category']          # A/B/C/D
        full_path  = row['full_path']         # e.g. FakeAVCeleb/RealVideo-RealAudio/.../<stem>.mp4
        # Strip leading "FakeAVCeleb/" prefix if present
        rel = full_path.replace('FakeAVCeleb/', '', 1)
        stem = os.path.splitext(os.path.basename(rel))[0]
        subdir = os.path.dirname(rel)

        roi_path = os.path.join(args.preprocessed_dir, subdir, stem + '_roi.mp4')
        wav_path = os.path.join(args.preprocessed_dir, subdir, stem + '.wav')

        if not os.path.exists(roi_path) or not os.path.exists(wav_path):
            # Try trimmed suffix variants (won't change real names)
            continue

        try:
            score = compute_score(model, task, roi_path, wav_path,
                                  args.max_length, tmp_dir, device)
        except Exception as e:
            print(f"  SKIP {roi_path}: {e}")
            continue

        is_fake = int(cat_code in FAKE_CATS)
        all_scores.append(score)
        all_labels.append(is_fake)
        all_cats.append(cat_code)
        all_paths.append(full_path)

    print(f"Scored {len(all_scores)} / {len(rows)} clips")

    # ── Per-category AUC/AP ─────────────────────────────────────────────────
    results = {}

    # Overall: all fake vs all real
    results['overall'] = {
        'n': len(all_scores),
        'auc': safe_auc(all_labels, all_scores),
        'ap':  safe_ap(all_labels, all_scores),
    }

    # Per fake category vs real (A)
    real_idx    = [i for i, c in enumerate(all_cats) if c == 'A']
    real_scores = [all_scores[i] for i in real_idx]
    real_labels = [all_labels[i] for i in real_idx]

    category_names = {
        'B': 'RV-FA',
        'C': 'FV-RA',
        'D': 'FV-FA',
    }
    for code, name in category_names.items():
        fake_idx    = [i for i, c in enumerate(all_cats) if c == code]
        fake_scores = [all_scores[i] for i in fake_idx]
        fake_labels = [all_labels[i] for i in fake_idx]
        y_true  = real_labels + fake_labels
        y_score = real_scores + fake_scores
        results[name] = {
            'n_fake': len(fake_idx),
            'n_real': len(real_idx),
            'auc': safe_auc(y_true, y_score),
            'ap':  safe_ap(y_true, y_score),
        }

    # ── Print & save ────────────────────────────────────────────────────────
    print("\n===== SpeechForensics Results (FAVC test) =====")
    for k, v in results.items():
        auc = v['auc']
        ap  = v['ap']
        auc_s = f"{auc:.4f}" if auc is not None else "N/A"
        ap_s  = f"{ap:.4f}" if ap  is not None else "N/A"
        print(f"  {k:12s}  AUC={auc_s}  AP={ap_s}  (n={v.get('n', v.get('n_fake','?'))})")

    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")

    # Per-clip CSV
    import csv as csv_module
    perclip_path = os.path.join(args.output_dir, 'per_clip_scores.csv')
    with open(perclip_path, 'w', newline='') as f:
        w = csv_module.writer(f)
        w.writerow(['path', 'category', 'label', 'score'])
        for path, cat, lab, sc in zip(all_paths, all_cats, all_labels, all_scores):
            w.writerow([path, CATEGORY_MAP[cat], lab, sc])
    print(f"Per-clip scores saved to {perclip_path}")


if __name__ == '__main__':
    main()
