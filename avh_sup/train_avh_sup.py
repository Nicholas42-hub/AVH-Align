"""
Train the plain AVH_Sup classifier baseline on frozen AV-HuBERT features.

This entry point is intentionally small and mirrors the archived baseline
trainer, with two practical additions used for paper baselines:
  * CLI seed override
  * CLI output-dir override

The model itself is defined in mlp.py and supports:
  model_hparams.input_type in {"both", "audio", "video"}.
"""

import argparse
import os
import random

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from datasets import load_data
from mlp import AVH_Sup


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
    log_cfg = config["logger"]
    if log_cfg["name"] == "tensorboard":
        logger = TensorBoardLogger(log_cfg["log_path"])
    elif log_cfg["name"] == "csv":
        logger = CSVLogger(log_cfg["log_path"])
    else:
        raise ValueError(f"Logger '{log_cfg['name']}' not implemented")

    callbacks = []
    if "ckpt_args" in config:
        callbacks.append(ModelCheckpoint(
            monitor=config["ckpt_args"]["metric"],
            dirpath=config["ckpt_args"]["ckpt_dir"],
            filename="model-{epoch:02d}",
            mode=config["ckpt_args"]["mode"],
            save_top_k=1,
        ))
    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(EarlyStopping(
            monitor=config["early_stopping"]["metric"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        ))
    return logger, callbacks


def train(config: dict):
    train_dl, val_dl = load_data(config=config["data_info"])
    model = AVH_Sup(config=config)
    logger, callbacks = init_callbacks(config=config["callbacks"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


def apply_overrides(config: dict, args: argparse.Namespace):
    if args.seed is not None:
        config["seed"] = args.seed
    if args.input_type is not None:
        config["model_hparams"]["input_type"] = args.input_type
    if args.output_dir is not None:
        base = args.output_dir
        config["callbacks"]["logger"]["log_path"] = os.path.join(base, "logs")
        config["callbacks"]["ckpt_args"]["ckpt_dir"] = os.path.join(base, "ckpts")
        os.makedirs(os.path.join(base, "logs"), exist_ok=True)
        os.makedirs(os.path.join(base, "ckpts"), exist_ok=True)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AVH_Sup baseline")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input_type", choices=["both", "audio", "video"], default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = apply_overrides(config, args)

    set_seed(config.get("seed", 42))

    mh = config["model_hparams"]
    print(f"\n{'=' * 68}")
    print("  Plain AVH_Sup baseline")
    print(f"  model_type={mh['model_type']} input_type={mh['input_type']}")
    print(f"  seed={config.get('seed', 42)} epochs={config['epochs']}")
    print(f"{'=' * 68}\n")

    train(config)
