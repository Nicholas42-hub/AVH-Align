"""
A6-AV1M-PseudoDomain-v2 — POLICY-CLEAN training variant for SWA.

Differences from train_test_fcd_a6_av1m_pseudodomain_v2.py:

  1. Adds a SECOND ModelCheckpoint that saves every epoch's weights
     to ckpts/dense/ep-{epoch:02d}.ckpt  (save_top_k=-1, save_weights_only=True).
     This gives a dense set of training-time snapshots for SWA, with NO
     target signal involved in the saving rule.

  2. Disables the FAVCFvraCallback (target-side) so no target labels touch
     the training loop at all.

  3. Keeps the original source-val "best" save and last.ckpt unchanged.

The averaging window for SWA can then be defined a priori from source val_auc
(e.g. epochs >= E* where E* = first epoch source val_auc >= threshold) using
metrics.csv only — fully policy-compliant.
"""

import argparse
import os
import yaml

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# Import everything else from the original script
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_test_fcd_a6_av1m_pseudodomain_v2 import (
    set_seed,
    load_data,
    AVH_FCD_A6,
)


def init_callbacks_clean(config: dict):
    """Like init_callbacks but with dense every-epoch save and no FAVC callback."""
    log_cfg = config["logger"]
    logger = CSVLogger(log_cfg["log_path"])
    callbacks = []

    if "ckpt_args" in config:
        # Source-val best ckpt (policy-clean)
        callbacks.append(ModelCheckpoint(
            monitor=config["ckpt_args"]["metric"],
            dirpath=config["ckpt_args"]["ckpt_dir"],
            filename="model-{epoch:02d}",
            mode=config["ckpt_args"]["mode"],
            save_top_k=1,
            save_last=True,
        ))
        # Dense every-epoch save (weights only) for SWA
        dense_dir = os.path.join(config["ckpt_args"]["ckpt_dir"], "dense")
        callbacks.append(ModelCheckpoint(
            dirpath=dense_dir,
            filename="ep-{epoch:02d}",
            every_n_epochs=1,
            save_top_k=-1,
            save_weights_only=True,
            save_on_train_epoch_end=True,
        ))

    if "early_stopping" in config and config["early_stopping"] is not None:
        callbacks.append(EarlyStopping(
            monitor=config["early_stopping"]["metric"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        ))

    return logger, callbacks


def train_clean(config: dict):
    set_seed(config.get("seed", 43))
    train_dl, val_dl = load_data(config)
    model = AVH_FCD_A6(config=config)
    logger, callbacks = init_callbacks_clean(config["callbacks"])
    print(f"[train-clean] dense every-epoch ckpt save enabled; FAVC callback disabled", flush=True)
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=config.get("ckpt_path"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt_path", default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path

    ablation_id = config.get("ablation_id", "A6_av1m_pseudodomain_v2_clean")
    seed = config.get("seed", 43)

    print("\n" + "=" * 68)
    print(f"  {ablation_id} — POLICY-CLEAN training (dense ckpt save)")
    print(f"  seed={seed}")
    print(f"  every-epoch save: ckpt_dir/dense/ep-XX.ckpt (weights only)")
    print(f"  FAVC FV-RA callback: DISABLED")
    if config.get("ckpt_path"):
        print(f"  resume:        {config['ckpt_path']}")
    print("=" * 68 + "\n")

    train_clean(config)
