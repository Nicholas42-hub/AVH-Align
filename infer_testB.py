"""
Inference on preprocessed testB videos → prediction.txt
Outputs Codabench Task 1 format: filename.mp4;score (one per line)
Usage:
    python infer_testB.py --ckpt <path> --config <path> [--output prediction.txt]
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

# add avh_sup to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AVH_SUP    = os.path.join(_SCRIPT_DIR, "avh_sup")
_AVHUBERT_PKG = os.path.join(_SCRIPT_DIR, "av_hubert", "avhubert")
for _p in (_AVH_SUP, _AVHUBERT_PKG, os.path.dirname(_AVHUBERT_PKG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets_e2e import (
    load_video_frames, load_audio_feats, _align_av,
    build_video_transform, IMAGE_CROP_SIZE, STACK_ORDER_AUDIO,
    e2e_collate_fn,
)

PREPROCESSED_DIR = "/data/projects/punim2637/nnliang/Datasets/AVDeepfake1MPlusPlus/testB/preprocessed"


class TestBDataset(Dataset):
    """
    Loads preprocessed testB _roi.mp4 + .wav pairs from a flat directory.
    Returns (video, audio, 0, filename) — label is dummy 0.
    """
    def __init__(self, preprocessed_dir: str, max_frames: int = 150):
        self.preprocessed_dir = preprocessed_dir
        self.max_frames = max_frames
        self.transform  = build_video_transform()

        rois = {f[:-8] for f in os.listdir(preprocessed_dir) if f.endswith("_roi.mp4")}
        wavs = {f[:-4]  for f in os.listdir(preprocessed_dir) if f.endswith(".wav")}
        stems = sorted(rois & wavs)
        self.items = [(s + "_roi.mp4", s + ".wav", s + ".mp4") for s in stems]
        print(f"[TestBDataset] {len(self.items)} clips in {preprocessed_dir}", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        roi_name, wav_name, mp4_name = self.items[idx]
        roi_path = os.path.join(self.preprocessed_dir, roi_name)
        wav_path = os.path.join(self.preprocessed_dir, wav_name)
        try:
            video = load_video_frames(roi_path, self.transform)
            audio = load_audio_feats(wav_path)
            video, audio = _align_av(video, audio)
            if self.max_frames > 0:
                video = video[:self.max_frames]
                audio = audio[:self.max_frames]
        except Exception as e:
            print(f"[WARN] failed {roi_path}: {e}", flush=True)
            video = torch.zeros(1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE)
            audio = torch.zeros(1, STACK_ORDER_AUDIO * 26)
        return video, audio, 0, mp4_name


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    results = {}
    for batch in tqdm(dataloader, desc="Inference"):
        if batch is None:
            continue
        video, audio, _, fnames = batch
        video = video.to(device)
        audio = audio.to(device)
        video_feats, audio_feats = model.encoder(video, audio)
        scores = model.head.forward(video_feats, audio_feats, mode="causal")
        # sigmoid to convert logit → probability
        probs = torch.sigmoid(scores).cpu().numpy()
        for fname, prob in zip(fnames, probs):
            results[fname] = float(prob)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--preprocessed_dir", default=PREPROCESSED_DIR)
    parser.add_argument("--output",      default="prediction.txt")
    parser.add_argument("--max_frames",  type=int, default=150)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_module", default="train_e2e_a6")
    parser.add_argument("--model_class",  default="E2EModelA6")
    parser.add_argument("--limit",        type=int, default=0,
                        help="If >0, only process this many clips (for smoke-testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # Ensure avh_sup is on sys.path before importing model module
    avh_sup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avh_sup")
    if avh_sup_path not in sys.path:
        sys.path.insert(0, avh_sup_path)
    import importlib.util
    mod_path = os.path.join(avh_sup_path, args.model_module + ".py")
    spec = importlib.util.spec_from_file_location(args.model_module, mod_path)
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[args.model_module] = _mod
    spec.loader.exec_module(_mod)
    ModelCls = getattr(_mod, args.model_class)
    model = ModelCls.load_from_checkpoint(args.ckpt, config=config, strict=False)
    model.eval()
    model.to(device)
    print("Model loaded OK")

    # Dataset + loader
    ds = TestBDataset(args.preprocessed_dir, max_frames=args.max_frames)
    if args.limit > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))
        print(f"[--limit] Using first {len(ds)} clips for smoke test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=e2e_collate_fn)

    # Run inference
    results = run_inference(model, loader, device)
    print(f"\nScored {len(results)} clips.")

    # Write prediction.txt
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for fname in sorted(results.keys()):
            f.write(f"{fname};{results[fname]:.4f}\n")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
