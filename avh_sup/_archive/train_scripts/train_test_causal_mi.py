"""
Training and evaluation entry-point for the MINE-based causal AVH-Align model.

Evaluation outputs three sets of AUC scores:
  v_full     — full_head   using causal_repr + spurious_repr
  v_causal   — causal_head using causal_repr   ONLY
  v_spurious — spurious_head using spurious_repr ONLY

Hypothesis (proven if):
  v_causal   ≈ v_full   (causal repr alone is self-sufficient)
  v_spurious ≈ 0.5      on AV1M  AND  FakeAVCeleb  ← key cross-domain test
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

from datasets import load_data
from mlp_causal_mi import AVH_Causal_MI


def set_seed(seed: int):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_callbacks(config: dict):
    log_cfg = config["logger"]
    if log_cfg["name"] == "tensorboard":
        logger = TensorBoardLogger(log_cfg["log_path"])
    elif log_cfg["name"] == "csv":
        logger = CSVLogger(log_cfg["log_path"])
    else:
        raise ValueError(f"Logger '{log_cfg['name']}' not implemented")

    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(
            ModelCheckpoint(
                monitor=config["ckpt_args"]["metric"],
                dirpath=config["ckpt_args"]["ckpt_dir"],
                filename="model-{epoch:02d}",
                mode=config["ckpt_args"]["mode"],
                save_top_k=1,
            )
        )
    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config["early_stopping"]["metric"],
                mode=config["early_stopping"]["mode"],
                patience=config["early_stopping"]["patience"],
            )
        )
    return logger, callbacks


def train(config: dict):
    train_dl, val_dl = load_data(config=config["data_info"])
    model = AVH_Causal_MI(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def test(config: dict):
    test_dl = load_data(config=config["data_info"], test=True)
    model = AVH_Causal_MI.load_from_checkpoint(config["ckpt_path"])
    model.to("cuda")
    model.eval()

    all_labels = []
    all_paths  = []
    scores_by_mode = {"full": [], "causal": [], "spurious": []}

    with torch.no_grad():
        for batch in tqdm.tqdm(test_dl):
            video_feats, audio_feats, labels, paths = batch
            video_feats = video_feats.to("cuda")
            audio_feats = audio_feats.to("cuda")

            all_labels.extend(labels.cpu().numpy().tolist())
            all_paths.extend(paths)

            for mode in ("full", "causal", "spurious"):
                s = model.predict_scores(video_feats, audio_feats, mode=mode)
                scores_by_mode[mode].extend(s.cpu().numpy().tolist())

    labels_arr = np.array(all_labels)
    os.makedirs(config["output_path"], exist_ok=True)

    pd.DataFrame({
        "path":           all_paths,
        "label":          labels_arr,
        "score_full":     scores_by_mode["full"],
        "score_causal":   scores_by_mode["causal"],
        "score_spurious": scores_by_mode["spurious"],
    }).to_csv(os.path.join(config["output_path"], "results_causal_mi.csv"), index=False)

    results_txt = []
    print("\n" + "=" * 60)
    print("  AVH-Align Causal+MINE Evaluation")
    print("=" * 60)
    for mode in ("full", "causal", "spurious"):
        scores = np.array(scores_by_mode[mode])
        auc = roc_auc_score(y_true=labels_arr, y_score=scores)
        ap  = average_precision_score(y_true=labels_arr, y_score=scores)
        line = f"  {mode:<10}  AUC={auc:.4f}  AP={ap:.4f}"
        print(line)
        results_txt.append(line)
    print("=" * 60)

    auc_full     = roc_auc_score(labels_arr, np.array(scores_by_mode["full"]))
    auc_causal   = roc_auc_score(labels_arr, np.array(scores_by_mode["causal"]))
    auc_spurious = roc_auc_score(labels_arr, np.array(scores_by_mode["spurious"]))
    delta = abs(auc_full - auc_causal)

    print("\nHypothesis check:")
    print(f"  |v_full - v_causal|  = {delta:.4f}  "
          f"({'SUPPORTS' if delta < 0.02 else 'DOES NOT support'} causal hypothesis)")
    print(f"  v_spurious near 0.5? = {auc_spurious:.4f}  "
          f"({'YES' if abs(auc_spurious - 0.5) < 0.05 else 'NO'}) "
          f"← cross-domain test: compare with baseline (0.26 on FAVC)")

    with open(os.path.join(config["output_path"], "eval_results_causal_mi.txt"), "w") as f:
        f.write("AVH-Align Causal+MINE Evaluation\n")
        f.write("=" * 60 + "\n")
        for line in results_txt:
            f.write(line + "\n")
        f.write("\nHypothesis check:\n")
        f.write(f"  |v_full - v_causal|  = {delta:.4f}\n")
        f.write(f"  v_spurious           = {auc_spurious:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal+MINE AVH-Align train/test")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    if args.test:
        test(config=config)
    else:
        train(config=config)
