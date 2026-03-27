"""
Training and evaluation entry-point for the modal-balance causal model.

This script trains AVH_Causal_Modal — an extension of the A2 ablation base
(adversarial + stop-gradient) with three optional modal balance constraints:

  方向1 (use_modal_cls)   — per-modality classification loss
  方向2 (use_cross_modal) — cross-modal cosine alignment loss
  方向3 (use_modal_disc)  — modal discriminator adversarial on causal features

Configs: configs/B1.yaml   (方向1 only)
         configs/B2.yaml   (方向2 only)
         configs/B3.yaml   (方向3 only)
         configs/B_all.yaml (all three)

Usage:
  # Train
  python train_test_causal_modal.py --config_path configs/B1.yaml

  # Test (evaluate on AV1M test set)
  python train_test_causal_modal.py --config_path configs/B1.yaml --test

Evaluation outputs:
  results_modal.csv          — per-clip scores for causal & spurious modes
  eval_results_modal.txt     — AUC / AP summary and modal balance check
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import average_precision_score, roc_auc_score
import lightning as L

from datasets import load_data, AV1M_trainval_dataset
from mlp_causal_modal import AVH_Causal_Modal


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_callbacks(config: dict):
    log_cfg = config["callbacks"]["logger"]
    if log_cfg["name"] == "tensorboard":
        logger = TensorBoardLogger(log_cfg["log_path"])
    elif log_cfg["name"] == "csv":
        logger = CSVLogger(log_cfg["log_path"])
    else:
        raise ValueError(f"Logger '{log_cfg['name']}' not implemented")

    from torch.utils.data import DataLoader
    callbacks = []
    if "ckpt_args" in config["callbacks"]:
        ckpt_cfg = config["callbacks"]["ckpt_args"]
        callbacks.append(
            ModelCheckpoint(
                monitor=ckpt_cfg["metric"],
                dirpath=ckpt_cfg["ckpt_dir"],
                filename="model-{epoch:02d}",
                mode=ckpt_cfg["mode"],
                save_top_k=1,
            )
        )
    if "early_stopping" in config and config["early_stopping"] is not None:
        es_cfg = config["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg["metric"],
                mode=es_cfg["mode"],
                patience=es_cfg["patience"],
            )
        )
    return logger, callbacks


# ── Train ──────────────────────────────────────────────────────────────────────

def train(config: dict):
    from torch.utils.data import DataLoader

    data_cfg = config["data_info"]
    train_ds = AV1M_trainval_dataset(data_cfg, split="train")
    val_ds   = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}", flush=True)

    model = AVH_Causal_Modal(config=config)
    logger, callbacks = init_callbacks(config)

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ── Test ───────────────────────────────────────────────────────────────────────

def test(config: dict):
    test_dl = load_data(config=config["data_info"], test=True)

    model = AVH_Causal_Modal.load_from_checkpoint(config["ckpt_path"])
    model.to("cuda")
    model.eval()

    all_labels = []
    all_paths  = []
    scores_by_mode = {"causal": [], "spurious": []}

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dl):
            video_feats, audio_feats, labels, paths = batch
            video_feats = video_feats.to("cuda")
            audio_feats = audio_feats.to("cuda")

            all_labels.extend(labels.cpu().numpy().tolist())
            all_paths.extend(paths)

            for mode in ("causal", "spurious"):
                s = model.predict_scores(video_feats, audio_feats, mode=mode)
                scores_by_mode[mode].extend(s.cpu().numpy().tolist())

    labels_arr = np.array(all_labels)
    os.makedirs(config["output_path"], exist_ok=True)

    # ── Save per-clip scores ────────────────────────────────────────────────────
    pd.DataFrame({
        "path":           all_paths,
        "label":          labels_arr,
        "score_causal":   scores_by_mode["causal"],
        "score_spurious": scores_by_mode["spurious"],
    }).to_csv(os.path.join(config["output_path"], "results_modal.csv"), index=False)

    # ── Compute and print metrics ───────────────────────────────────────────────
    results_txt = []
    print("\n" + "=" * 68)
    print(f"  AVH-Align Modal Balance Evaluation  [{config.get('modal_balance_id', '?')}]")
    print("=" * 68)
    for mode in ("causal", "spurious"):
        scores = np.array(scores_by_mode[mode])
        auc = roc_auc_score(y_true=labels_arr, y_score=scores)
        ap  = average_precision_score(y_true=labels_arr, y_score=scores)
        line = f"  {mode:<10}  AUC={auc:.4f}  AP={ap:.4f}  N={len(labels_arr)}"
        print(line)
        results_txt.append(line)
    print("=" * 68)

    auc_causal   = roc_auc_score(labels_arr, np.array(scores_by_mode["causal"]))
    auc_spurious = roc_auc_score(labels_arr, np.array(scores_by_mode["spurious"]))
    check1 = f"  v_spurious near 0.5? = {auc_spurious:.4f}  ({'YES' if abs(auc_spurious - 0.5) < 0.05 else 'NO'})"
    print(check1)

    with open(os.path.join(config["output_path"], "eval_results_modal.txt"), "w") as f:
        f.write(f"AVH-Align Modal Balance Evaluation  [{config.get('modal_balance_id', '?')}]\n")
        f.write("=" * 68 + "\n")
        for line in results_txt:
            f.write(line + "\n")
        f.write(check1 + "\n")

    print(f"\nResults saved to {config['output_path']}")


# ── Entry-point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal balance causal model — train / test")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    parser.add_argument("--test", action="store_true", help="Run evaluation instead of training")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    if args.test:
        test(config=config)
    else:
        train(config=config)
