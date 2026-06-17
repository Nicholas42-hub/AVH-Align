import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset


FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]


class FusionModel(nn.Module):
    def __init__(self, visual_dim=1024, audio_dim=1024, hidden_dim=1024):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim // 2)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
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

    def forward(self, visual_features, audio_features):
        visual_proj = self.visual_proj(visual_features)
        audio_proj = self.audio_proj(audio_features)
        return self.mlp(torch.cat((visual_proj, audio_proj), dim=-1))


def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_score)


def safe_ap(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_score)


def score_clip(model, video, audio, device):
    video = video.squeeze(0).to(device)
    audio = audio.squeeze(0).to(device)
    video = video / (torch.linalg.norm(video, ord=2, dim=-1, keepdim=True) + 1e-8)
    audio = audio / (torch.linalg.norm(audio, ord=2, dim=-1, keepdim=True) + 1e-8)
    output = model(video, audio)
    return torch.logsumexp(-output, dim=0).detach().cpu().item()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = FusionModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def per_category_metrics(paths, scores, labels):
    results = {
        "overall": {
            "auc": safe_auc(labels, scores),
            "ap": safe_ap(labels, scores),
            "n": len(labels),
            "n_real": labels.count(0),
            "n_fake": labels.count(1),
        }
    }
    for cat in FAKE_CATEGORIES:
        mask = [
            (cat in p) or (FakeAVCeleb_NPZ_Dataset.REAL_CATEGORY in p)
            for p in paths
        ]
        cat_scores = [s for s, m in zip(scores, mask) if m]
        cat_labels = [l for l, m in zip(labels, mask) if m]
        results[cat] = {
            "auc": safe_auc(cat_labels, cat_scores),
            "ap": safe_ap(cat_labels, cat_scores),
            "n": len(cat_labels),
        }
    return results


def write_line(text, fh):
    print(text, flush=True)
    fh.write(text + "\n")


def build_dataset(args):
    if args.dataset == "favc":
        config = {
            "root_path": args.features_path,
            "csv_root_path": args.csv_root_path,
            "apply_l2": False,
        }
        return FakeAVCeleb_NPZ_Dataset(config, split=args.split)

    config = {
        "root_path": args.features_path,
        "csv_root_path": args.csv_root_path,
        "apply_l2": False,
    }
    return AV1M_trainval_dataset(config, split=args.split)


def main():
    parser = argparse.ArgumentParser(description="Evaluate original unsupervised AVH-Align checkpoint.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset", choices=["favc", "av1m"], required=True)
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--csv_root_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint_path, device)
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    paths, scores, labels = [], [], []
    with torch.no_grad():
        for video, audio, label, path in tqdm.tqdm(loader, desc=f"Evaluating {args.dataset}"):
            scores.append(score_clip(model, video, audio, device))
            labels.append(int(label.item() if torch.is_tensor(label) else label))
            paths.append(path[0] if isinstance(path, (list, tuple)) else path)

    pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
        os.path.join(args.output_dir, "predictions.csv"), index=False
    )

    out_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(out_file, "w") as fh:
        write_line("Unsupervised AVH-Align Evaluation", fh)
        write_line(f"Checkpoint: {args.checkpoint_path}", fh)
        write_line(f"Dataset   : {args.dataset}", fh)
        write_line(f"Features  : {args.features_path}", fh)
        write_line(f"Split     : {args.split}", fh)
        write_line("=" * 60, fh)

        if args.dataset == "favc":
            metrics = per_category_metrics(paths, scores, labels)
            for key, m in metrics.items():
                auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
                ap = f"{m['ap']:.4f}" if m["ap"] is not None else "  N/A "
                write_line(f"  {key:<30}  AUC={auc}  AP={ap}  N={m['n']}", fh)
        else:
            auc = safe_auc(labels, scores)
            ap = safe_ap(labels, scores)
            write_line(f"  overall                         AUC={auc:.4f}  AP={ap:.4f}  N={len(labels)}", fh)

        write_line("=" * 60, fh)
        write_line(f"Results saved to: {args.output_dir}", fh)


if __name__ == "__main__":
    main()
