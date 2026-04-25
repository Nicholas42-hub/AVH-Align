"""
A7-FAVC-PseudoDomain — FCD + Domain Push-Pull + GRL task-adversarial head on Z_s.

Identical to A6-FAVC-PseudoDomain but uses AVH_FCD_A7 which adds:
    L_tadv_s = BCE(task_adv_head_s(GRL(Z_s_pool)), fake_label)  (AV1M clips only)

This expels fake/real information from the spurious branch, pushing spurious AUC → 0.5.
New hyperparameter: lambda_tadv_s (default 0.5)
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
import lightning as L

from datasets import FakeAVCeleb_NPZ_Dataset
from mlp_fcd_a7 import AVH_FCD_A7

# Re-use dataset and helper functions from A6 script
from train_test_fcd_a6_favc_pseudodomain import (
    set_seed,
    FAVC_SpeakerGroupDomainDataset,
    load_data,
    init_callbacks,
)


def train(config: dict):
    set_seed(config.get("seed", 42))
    train_dl, val_dl = load_data(config)
    model  = AVH_FCD_A7(config=config)
    logger, callbacks = init_callbacks(config["callbacks"])
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--seed",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed

    print(f"\n{'='*68}")
    print(f"  A7-FAVC-PseudoDomain — A6 + GRL task-adv head on Z_s")
    print(f"  syn_dim={config['model_hparams']['syn_dim']}  "
          f"spec_dim={config['model_hparams']['spec_dim']}  "
          f"lr={config['model_hparams']['lr']}  "
          f"lambda_tadv_s={config['model_hparams'].get('lambda_tadv_s', 0.5)}")
    print(f"{'='*68}\n")

    train(config)
