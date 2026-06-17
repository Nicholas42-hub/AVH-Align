import argparse
import csv
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from datasets import AV1M_trainval_dataset, FakeAVCeleb_NPZ_Dataset
from mlp_fcd import CCD, SDA, URD, _make_mlp, _sinkhorn_divergence
from mlp_fcd_triroute import CrossModalInconsistency


FAKE_CATEGORIES = [
    "RealVideo-FakeAudio",
    "FakeVideo-RealAudio",
    "FakeVideo-FakeAudio",
]


class TriRouteUnsupSync(nn.Module):
    def __init__(self, config):
        super().__init__()
        hp = config.get("model_hparams", {})
        feat_dim = int(hp.get("feat_dim", 1024))
        syn_dim = int(hp.get("syn_dim", 256))
        spec_dim = int(hp.get("spec_dim", 256))
        sda_dim = int(hp.get("sda_dim", 128))
        incon_dim = int(hp.get("incon_dim", 256))

        self.ccd_v = CCD(feat_dim, syn_dim, spec_dim)
        self.ccd_a = CCD(feat_dim, syn_dim, spec_dim)
        self.urd_v = URD(spec_dim)
        self.urd_a = URD(spec_dim)
        self.sda = SDA(syn_dim, sda_dim)
        self.cross_incon = CrossModalInconsistency(feat_dim, incon_dim)
        self.sync_head = _make_mlp((syn_dim + spec_dim) * 2 + incon_dim)
        self.modality_head = nn.Sequential(
            nn.Linear(spec_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.lambda_sda = float(hp.get("lambda_sda", 0.5))
        self.lambda_dis = float(hp.get("lambda_dis", 1.0))
        self.lambda_orth = float(hp.get("lambda_orth", 0.1))
        self.lambda_delta_sparse = float(hp.get("lambda_delta_sparse", 0.01))
        self.sda_eps = float(hp.get("sda_eps", 0.05))
        self.sda_n_iter = int(hp.get("sda_n_iter", 5))

    def encode(self, video_feats, audio_feats):
        s_v, h_v = self.ccd_v(video_feats)
        s_a, h_a = self.ccd_a(audio_feats)
        u_v, r_v = self.urd_v(h_v)
        u_a, r_a = self.urd_a(h_a)
        u_delta = self.cross_incon(video_feats, audio_feats)
        task = torch.cat([s_v, u_v, u_delta, s_a, u_a], dim=-1)
        return task, {"s_v": s_v, "s_a": s_a, "u_v": u_v, "u_a": u_a, "r_v": r_v, "r_a": r_a, "u_delta": u_delta}

    def logits(self, video_feats, audio_feats):
        task, _ = self.encode(video_feats, audio_feats)
        return self.sync_head(task)[..., 0]

    @staticmethod
    def orth_loss(u, r):
        u_n = F.normalize(u.reshape(-1, u.shape[-1]), dim=-1)
        r_n = F.normalize(r.reshape(-1, r.shape[-1]), dim=-1)
        return ((u_n * r_n).sum(dim=-1) ** 2).mean()

    def regularizers(self, comp):
        s_v, s_a = comp["s_v"], comp["s_a"]
        u_v, u_a = comp["u_v"], comp["u_a"]
        r_v, r_a = comp["r_v"], comp["r_a"]
        u_delta = comp["u_delta"]

        s_v_proj = self.sda(s_v.mean(1))
        s_a_proj = self.sda(s_a.mean(1))
        loss_sda = _sinkhorn_divergence(
            s_v_proj, s_a_proj, eps=self.sda_eps, n_iter=self.sda_n_iter
        )

        r_v_pool = r_v.mean(1)
        r_a_pool = r_a.mean(1)
        mod_score = self.modality_head(torch.cat([r_v_pool, r_a_pool], dim=0)).squeeze(-1)
        mod_labels = torch.cat([
            torch.zeros(r_v_pool.shape[0], device=r_v_pool.device),
            torch.ones(r_a_pool.shape[0], device=r_a_pool.device),
        ])
        loss_dis = F.binary_cross_entropy_with_logits(mod_score, mod_labels)

        loss_orth = (1 / 3) * (
            self.orth_loss(u_v, r_v)
            + self.orth_loss(u_a, r_a)
            + 0.5 * (self.orth_loss(u_delta, r_v) + self.orth_loss(u_delta, r_a))
        )
        loss_delta_sparse = (u_delta ** 2).mean()
        return loss_sda, loss_dis, loss_orth, loss_delta_sparse


class AV1MRealSyncDataset(Dataset):
    def __init__(
        self,
        root_path,
        csv_root_path,
        split="train",
        tau=15,
        max_clips=None,
        frames_per_clip=1,
    ):
        df = pd.read_csv(os.path.join(csv_root_path, f"{split}_labels.csv"))
        df = df[df["label"] == 0].reset_index(drop=True)
        if max_clips is not None:
            df = df.iloc[:max_clips].reset_index(drop=True)
        self.items = []
        self.tau = tau
        self.frames_per_clip = int(frames_per_clip)
        # Support multi-root: colon-separated list of root_path values, each
        # tried in order until the relative .npz is found. This lets one
        # csv index features that are split across filesystems (scratch + gpfs).
        roots = root_path.split(":") if isinstance(root_path, str) else list(root_path)
        self.feats_dirs = [os.path.join(p, split) for p in roots]
        n_missing = 0
        for _, row in df.iterrows():
            rel = row["path"][:-4] + ".npz"
            for fd in self.feats_dirs:
                p = os.path.join(fd, rel)
                if os.path.exists(p):
                    self.items.append(p)
                    break
            else:
                n_missing += 1
        if not self.items:
            raise RuntimeError(f"No real AV1M clips found under any of {self.feats_dirs}")
        if n_missing:
            print(f"  [warn] {n_missing} csv rows had no matching .npz under any root", flush=True)
        print(
            f"AV1M real sync dataset [{split}]: {len(self.items)} clips, "
            f"frames_per_clip={self.frames_per_clip}, "
            f"samples/epoch={len(self.items) * self.frames_per_clip}",
            flush=True,
        )

    def __len__(self):
        return len(self.items) * self.frames_per_clip

    def __getitem__(self, idx):
        clip_idx = idx // self.frames_per_clip
        feats = np.load(self.items[clip_idx], allow_pickle=True)
        visual = feats["visual"].astype(np.float32)
        audio = feats["audio"].astype(np.float32)
        n = min(len(visual), len(audio))
        frame_idx = random.randrange(n)
        v = visual[frame_idx]
        window = []
        for t in range(frame_idx - self.tau, frame_idx + self.tau + 1):
            if 0 <= t < n:
                a = audio[t]
            else:
                a = np.zeros(audio.shape[-1], dtype=np.float32)
            window.append(a)
        v = torch.tensor(v).float()
        a = torch.tensor(np.stack(window)).float()
        v = v / (torch.linalg.norm(v, ord=2, dim=-1, keepdim=True) + 1e-8)
        a = a / (torch.linalg.norm(a, ord=2, dim=-1, keepdim=True) + 1e-8)
        return v, a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    config = {
        "model_hparams": {
            "feat_dim": 1024,
            "syn_dim": args.syn_dim,
            "spec_dim": args.spec_dim,
            "sda_dim": args.sda_dim,
            "incon_dim": args.incon_dim,
            "lambda_sda": args.lambda_sda,
            "lambda_dis": args.lambda_dis,
            "lambda_orth": args.lambda_orth,
            "lambda_delta_sparse": args.lambda_delta_sparse,
            "sda_eps": 0.05,
            "sda_n_iter": 5,
        }
    }
    device = torch.device(args.device)
    model = TriRouteUnsupSync(config).to(device)
    train_features = args.train_features or args.av1m_features
    train_csv_root = args.train_csv_root or args.av1m_csv_root
    dataset = AV1MRealSyncDataset(
        train_features,
        train_csv_root,
        split="train",
        tau=args.tau,
        max_clips=args.max_train_clips,
        frames_per_clip=args.frames_per_clip,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    else:
        scheduler = None
    os.makedirs(args.output_dir, exist_ok=True)
    dense_dir = os.path.join(args.output_dir, "ckpts_dense")
    os.makedirs(dense_dir, exist_ok=True)

    # Resume from existing checkpoint if present
    start_epoch = 0
    best_loss = float("inf")
    best_path = os.path.join(args.output_dir, "triroute_unsup_sync.pt")
    best_auc_path = os.path.join(args.output_dir, "triroute_unsup_sync_best_auc.pt")
    best_auc = 0.0
    if os.path.exists(best_auc_path):
        _auc_ckpt = torch.load(best_auc_path, map_location="cpu", weights_only=False)
        best_auc = _auc_ckpt.get("overall_auc", 0.0)
        print(f"Loaded best_auc={best_auc:.4f} from {best_auc_path}", flush=True)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from {best_path} (epoch={start_epoch}, best_loss={best_loss:.6f})", flush=True)
        if start_epoch >= args.epochs:
            print(f"Already completed {start_epoch}/{args.epochs} epochs, skipping training.", flush=True)
            return best_path

    favc_loader = maybe_build_favc_loader(args)
    if favc_loader is not None:
        print(
            f"periodic cross-domain eval enabled: every {args.cross_domain_eval_every} epochs",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = []
        for v, a in tqdm.tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            v = v.to(device)
            a = a.to(device)
            v_rep = v.unsqueeze(1).repeat(1, 2 * args.tau + 1, 1)
            logits = model.logits(v_rep, a)
            target = torch.full((logits.shape[0],), args.tau, dtype=torch.long, device=device)
            loss_sync = F.cross_entropy(logits, target)
            _, comp = model.encode(v_rep, a)
            loss_sda, loss_dis, loss_orth, loss_delta = model.regularizers(comp)
            loss = (
                loss_sync
                + model.lambda_sda * loss_sda
                + model.lambda_dis * loss_dis
                + model.lambda_orth * loss_orth
                + model.lambda_delta_sparse * loss_delta
            )
            if args.lambda_consistency > 0:
                v_aug = v + torch.randn_like(v) * args.cons_noise_std
                a_aug = a + torch.randn_like(a) * args.cons_noise_std
                v_rep_aug = v_aug.unsqueeze(1).repeat(1, 2 * args.tau + 1, 1)
                logits_aug = model.logits(v_rep_aug, a_aug)
                loss_cons = F.kl_div(
                    F.log_softmax(logits_aug, dim=-1),
                    F.softmax(logits.detach(), dim=-1),
                    reduction="batchmean",
                )
                loss = loss + args.lambda_consistency * loss_cons
            if args.lambda_entropy > 0:
                probs = F.softmax(logits, dim=-1)
                loss_ent = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                loss = loss + args.lambda_entropy * loss_ent
            opt.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            totals.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(totals))
        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch={epoch+1} train_loss={mean_loss:.6f} lr={cur_lr:.6e}", flush=True)
        if scheduler is not None:
            scheduler.step()
        dense_path = os.path.join(dense_dir, f"ep_{epoch+1:03d}.pt")
        torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch + 1, "loss": mean_loss}, dense_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch + 1, "loss": best_loss}, best_path)
            print(f"saved best: {best_path}", flush=True)
        if (
            favc_loader is not None
            and ((epoch + 1) % args.cross_domain_eval_every == 0 or (epoch + 1) == args.epochs)
        ):
            metrics = run_cross_domain_eval(model, favc_loader, device)
            overall_auc = metrics["overall"]["auc"]
            fvra_auc = metrics["FakeVideo-RealAudio"]["auc"]
            rvfa_auc = metrics["RealVideo-FakeAudio"]["auc"]
            fvfa_auc = metrics["FakeVideo-FakeAudio"]["auc"]
            print(
                "[cross-domain] "
                f"epoch={epoch+1} "
                f"overall_auc={overall_auc:.4f} "
                f"fvra_auc={fvra_auc:.4f} "
                f"rvfa_auc={rvfa_auc:.4f} "
                f"fvfa_auc={fvfa_auc:.4f}",
                flush=True,
            )
            if overall_auc > best_auc:
                best_auc = overall_auc
                torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch + 1, "loss": mean_loss, "overall_auc": best_auc}, best_auc_path)
                print(f"saved best_auc: {best_auc_path} (overall_auc={best_auc:.4f})", flush=True)
            append_cross_domain_history(args.output_dir, epoch + 1, metrics)
    return best_path


def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else None


def safe_ap(y_true, y_score):
    return average_precision_score(y_true, y_score) if len(set(y_true)) >= 2 else None


def score_clip(model, video, audio, device):
    video = video.to(device)
    audio = audio.to(device)
    logits = model.logits(video, audio)
    return torch.logsumexp(-logits, dim=-1).detach().cpu().numpy().tolist()


def eval_loader(model, loader, device):
    model.eval()
    paths, scores, labels = [], [], []
    with torch.no_grad():
        for video, audio, label, path in tqdm.tqdm(loader, desc="eval"):
            video = video.to(device)
            audio = audio.to(device)
            if video.ndim == 2:
                video = video.unsqueeze(0)
                audio = audio.unsqueeze(0)
            scores.extend(score_clip(model, video, audio, device))
            labels.extend(label.numpy().astype(int).tolist())
            paths.extend(list(path))
    return paths, scores, labels


def maybe_build_favc_loader(args):
    if args.cross_domain_eval_every <= 0:
        return None
    favc_cfg = {
        "root_path": args.favc_features,
        "csv_root_path": args.favc_csv_root,
        "apply_l2": True,
    }
    favc_ds = FakeAVCeleb_NPZ_Dataset(favc_cfg, split="test")
    return DataLoader(
        favc_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )


def per_category_metrics(paths, scores, labels):
    results = {"overall": {"auc": safe_auc(labels, scores), "ap": safe_ap(labels, scores), "n": len(labels)}}
    for cat in FAKE_CATEGORIES:
        mask = [(cat in p) or (FakeAVCeleb_NPZ_Dataset.REAL_CATEGORY in p) for p in paths]
        ys = [y for y, m in zip(labels, mask) if m]
        ss = [s for s, m in zip(scores, mask) if m]
        results[cat] = {"auc": safe_auc(ys, ss), "ap": safe_ap(ys, ss), "n": len(ys)}
    return results


def append_cross_domain_history(output_dir, epoch, metrics):
    path = os.path.join(output_dir, "cross_domain_history.csv")
    row = {
        "epoch": epoch,
        "overall_auc": metrics["overall"]["auc"],
        "overall_ap": metrics["overall"]["ap"],
        "fvra_auc": metrics["FakeVideo-RealAudio"]["auc"],
        "fvra_ap": metrics["FakeVideo-RealAudio"]["ap"],
        "rvfa_auc": metrics["RealVideo-FakeAudio"]["auc"],
        "rvfa_ap": metrics["RealVideo-FakeAudio"]["ap"],
        "fvfa_auc": metrics["FakeVideo-FakeAudio"]["auc"],
        "fvfa_ap": metrics["FakeVideo-FakeAudio"]["ap"],
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_results(path, title, metrics):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "eval_results.txt"), "w") as f:
        print(title, file=f)
        print("=" * 60, file=f)
        for key, m in metrics.items():
            auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A "
            ap = f"{m['ap']:.4f}" if m["ap"] is not None else "  N/A "
            print(f"  {key:<30}  AUC={auc}  AP={ap}  N={m['n']}", file=f)


def run_cross_domain_eval(model, loader, device):
    paths, scores, labels = eval_loader(model, loader, device)
    return per_category_metrics(paths, scores, labels)


def evaluate(args, ckpt_path):
    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TriRouteUnsupSync(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    favc_cfg = {"root_path": args.favc_features, "csv_root_path": args.favc_csv_root, "apply_l2": True}
    favc_ds = FakeAVCeleb_NPZ_Dataset(favc_cfg, split="test")
    favc_loader = DataLoader(favc_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    paths, scores, labels = eval_loader(model, favc_loader, device)
    favc_dir = os.path.join(args.output_dir, "results_favc_7030")
    os.makedirs(favc_dir, exist_ok=True)
    pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
        os.path.join(favc_dir, "predictions.csv"), index=False
    )
    write_results(favc_dir, "TriRoute-unsup-sync FAVC 70/30", per_category_metrics(paths, scores, labels))

    av1m_cfg = {"root_path": args.av1m_features, "csv_root_path": args.av1m_csv_root, "apply_l2": True}
    av1m_ds = AV1M_trainval_dataset(av1m_cfg, split="val")
    av1m_loader = DataLoader(av1m_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    paths, scores, labels = eval_loader(model, av1m_loader, device)
    av1m_dir = os.path.join(args.output_dir, "results_av1m_val")
    os.makedirs(av1m_dir, exist_ok=True)
    pd.DataFrame({"path": paths, "score": scores, "label": labels}).to_csv(
        os.path.join(av1m_dir, "predictions.csv"), index=False
    )
    write_results(av1m_dir, "TriRoute-unsup-sync AV1M val", {
        "overall": {"auc": safe_auc(labels, scores), "ap": safe_ap(labels, scores), "n": len(labels)}
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--av1m_features", default="/data/projects/punim2637/nnliang/AVH-Align/data/avh_features")
    parser.add_argument("--av1m_csv_root", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/av1m")
    parser.add_argument("--train_features", default=None, help="Optional feature root used for unsupervised sync training.")
    parser.add_argument("--train_csv_root", default=None, help="Optional CSV root used for unsupervised sync training.")
    parser.add_argument("--favc_features", default="/data/projects/punim2637/nnliang/AVH-Align/data/favc_features")
    parser.add_argument("--favc_csv_root", default="/data/projects/punim2637/nnliang/AVH-Align/avh_sup/csv_metadata/favc_7030_canonical")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tau", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6, help="cosine schedule floor")
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="max grad norm; 0 = off")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_train_clips", type=int, default=None)
    parser.add_argument("--frames_per_clip", type=int, default=1)
    parser.add_argument("--syn_dim", type=int, default=256)
    parser.add_argument("--spec_dim", type=int, default=256)
    parser.add_argument("--sda_dim", type=int, default=128)
    parser.add_argument("--incon_dim", type=int, default=256)
    parser.add_argument("--lambda_sda", type=float, default=0.5)
    parser.add_argument("--lambda_dis", type=float, default=1.0)
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--lambda_delta_sparse", type=float, default=0.01)
    parser.add_argument("--lambda_consistency", type=float, default=0.0,
                        help="Consistency-reg weight: KL(p(logits) || p(logits_aug)) under feature noise.")
    parser.add_argument("--cons_noise_std", type=float, default=0.05,
                        help="Std of Gaussian noise added to AV-HuBERT features for consistency aug.")
    parser.add_argument("--lambda_entropy", type=float, default=0.0,
                        help="Entropy-min weight on softmax(sync_logits). TENT-style aux loss.")
    parser.add_argument(
        "--cross_domain_eval_every",
        type=int,
        default=0,
        help="If >0, run FAVC cross-domain eval every N epochs during training.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not args.eval_only:
        ckpt_path = train(args)
    evaluate(args, ckpt_path)


if __name__ == "__main__":
    main()
