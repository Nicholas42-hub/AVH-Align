"""
Train / test entry-point for the Mutual Information (MINE) model.

Usage:
    python train_test_mi.py --config_path configs/train_config_av1m_mi.yaml
    python train_test_mi.py --config_path configs/train_config_av1m_mi.yaml --test
"""

import argparse
import os
import random
import tqdm
import yaml

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets import load_data
from mlp_mi import AVH_MI


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_callbacks(config: dict):
    # ── Logger ─────────────────────────────────────────────────────────────────
    logger_path = config["logger"]["log_path"]
    if config["logger"]["name"] == "tensorboard":
        logger = TensorBoardLogger(logger_path)
    elif config["logger"]["name"] == "csv":
        logger = CSVLogger(logger_path)
    else:
        raise ValueError(f"Unknown logger: {config['logger']['name']}")

    # ── Callbacks ──────────────────────────────────────────────────────────────
    callbacks = []

    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=config["ckpt_args"]["ckpt_dir"],
                filename="mi-model-{epoch:02d}",
                mode=config["ckpt_args"]["mode"],
                save_top_k=3,
            )
        )

    if "early_stopping" in config:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                patience=config["early_stopping"]["patience"],
            )
        )

    return logger, callbacks


# ── Train ──────────────────────────────────────────────────────────────────────

def train(config: dict):
    train_dl, val_dl = load_data(config=config["data_info"])
    model = AVH_MI(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


# ── Test ───────────────────────────────────────────────────────────────────────

def test(config: dict):
    test_dl = load_data(config=config["data_info"], test=True)
    model = AVH_MI.load_from_checkpoint(config["ckpt_path"])

    model.to("cuda")
    model.eval()

    all_scores = np.array([])
    all_labels = np.array([])
    all_paths  = np.array([])

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dl):
            video_feats, audio_feats, labels, paths = batch
            video_feats = video_feats.to("cuda")
            audio_feats = audio_feats.to("cuda")

            # predict_scores returns –T(z_v, z_a): higher ↔ more likely fake
            scores = model.predict_scores(video_feats, audio_feats)

            all_scores = np.concatenate((all_scores, scores.cpu().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels.numpy()), axis=0)
            all_paths  = np.concatenate((all_paths, paths), axis=0)

    os.makedirs(config["output_path"], exist_ok=True)

    pd.DataFrame({
        "paths":  all_paths,
        "scores": all_scores,
        "labels": all_labels,
    }).to_csv(os.path.join(config["output_path"], "results_mi.csv"), index=False)

    auc = roc_auc_score(y_score=all_scores, y_true=all_labels)
    ap  = average_precision_score(y_score=all_scores, y_true=all_labels)

    result_str = f"AUC: {auc:.6f}\nAP:  {ap:.6f}\n"
    print(result_str)

    with open(os.path.join(config["output_path"], "eval_results_mi.txt"), "w") as f:
        f.write(result_str)


# ── Entry-point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVH-Align MI training and testing")
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
