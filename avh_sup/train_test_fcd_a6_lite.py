"""
A6-lite-trainreal — entry-point.

FCD + Light Domain Adversarial (lambda=0.2, warmup=5 epochs) on FAVC train-real.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import lightning as L

from datasets import AV1M_trainval_dataset
from mlp_fcd_a6_lite import AVH_FCD_A6_Lite

# Reuse the domain dataset classes from the A6 leakage-control script
from train_test_fcd_a6 import AV1M_DomainWrapper, FAVC_RealDomainDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}", flush=True)


def load_data(config: dict):
    data_cfg = config["data_info"]
    favc_cfg = config["favc_domain_info"]

    av1m_train = AV1M_DomainWrapper(AV1M_trainval_dataset(data_cfg, split="train"))
    favc_real  = FAVC_RealDomainDataset(favc_cfg)
    train_ds   = ConcatDataset([av1m_train, favc_real])
    val_ds     = AV1M_trainval_dataset(data_cfg, split="val")

    train_dl = DataLoader(train_ds, shuffle=True,  batch_size=1)
    val_dl   = DataLoader(val_ds,   shuffle=False, batch_size=1)

    print(
        f"Train: {len(av1m_train)} AV1M + {len(favc_real)} FAVC-real "
        f"= {len(train_ds)} total   Val(AV1M): {len(val_ds)}",
        flush=True,
    )
    return train_dl, val_dl


def init_callbacks(config: dict):
    log_cfg = config["logger"]
    logger  = CSVLogger(log_cfg["log_path"])
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
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model  = AVH_FCD_A6_Lite(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override logger/ckpt/results base directory")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    if args.output_dir is not None:
        base = args.output_dir
        config["callbacks"]["logger"]["log_path"] = os.path.join(base, "logs")
        config["callbacks"]["ckpt_args"]["ckpt_dir"] = os.path.join(base, "ckpts")
        config["output_path"] = os.path.join(base, "results")
        os.makedirs(os.path.join(base, "logs"), exist_ok=True)
        os.makedirs(os.path.join(base, "ckpts"), exist_ok=True)
        os.makedirs(os.path.join(base, "results"), exist_ok=True)

    hp = config.get("model_hparams", {})
    print("\n" + "=" * 68)
    print("  A6-lite-trainreal — FCD + Light Domain Adversarial")
    print(f"  seed={config.get('seed', 43)}")
    print(f"  lambda_dadv={hp.get('lambda_dadv', 0.2)}")
    print(f"  lambda_ddis={hp.get('lambda_ddis', 0.2)}")
    print(f"  warmup_epochs={hp.get('warmup_epochs', 5)}")
    if args.output_dir is not None:
        print(f"  output_dir={args.output_dir}")
    print("=" * 68 + "\n")

    if args.mode == "train":
        train(config)
