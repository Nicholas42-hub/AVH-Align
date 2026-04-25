"""
Generate A6 routing-ablation configs.

The grid is designed to answer the closest reviewer questions:
  noudelta          : paper-clean TriRoute, residual excluded, no u_delta
  expose_residual  : same model, but the task head can also read residual factors
  ddis_only         : residual domain-attraction only
  dadv_only         : task-branch domain-adversarial pressure only
  udelta100        : clean u_delta comparison with the same 100-epoch budget
"""

from pathlib import Path

import yaml


SUP = Path(__file__).absolute().parent
ROOT = SUP.parent
CONFIG_DIR = SUP / "configs" / "routing_ablation"
SEEDS = (43, 44, 45)

VARIANTS = {
    "noudelta": {
        "incon_dim": 0,
        "lambda_dadv": 1.0,
        "lambda_ddis": 1.0,
        "expose_residual_to_task_head": False,
    },
    "expose_residual": {
        "incon_dim": 0,
        "lambda_dadv": 1.0,
        "lambda_ddis": 1.0,
        "expose_residual_to_task_head": True,
    },
    "ddis_only": {
        "incon_dim": 0,
        "lambda_dadv": 0.0,
        "lambda_ddis": 1.0,
        "expose_residual_to_task_head": False,
    },
    "dadv_only": {
        "incon_dim": 0,
        "lambda_dadv": 1.0,
        "lambda_ddis": 0.0,
        "expose_residual_to_task_head": False,
    },
    "udelta100": {
        "incon_dim": 256,
        "lambda_dadv": 1.0,
        "lambda_ddis": 1.0,
        "expose_residual_to_task_head": False,
    },
}


def _base_config(variant: str, seed: int, settings: dict) -> dict:
    out_dir = SUP / f"outputs_A6_trainvalreal_{variant}_seed{seed}"
    hparams = {
        "feat_dim": 1024,
        "syn_dim": 256,
        "spec_dim": 256,
        "sda_dim": 128,
        "mi_dim": 128,
        "incon_dim": settings["incon_dim"],
        "lambda_mi": 1.0,
        "lambda_sda": 0.5,
        "lambda_dis": 1.0,
        "lambda_orth": 0.1,
        "lambda_dadv": settings["lambda_dadv"],
        "lambda_ddis": settings["lambda_ddis"],
        "grl_alpha": 1.0,
        "sda_eps": 0.05,
        "sda_n_iter": 5,
        "lr": 1.0e-3,
    }
    if settings["expose_residual_to_task_head"]:
        hparams["expose_residual_to_task_head"] = True

    return {
        "ablation_id": f"A6_trainvalreal_{variant}_seed{seed}",
        "data_info": {
            "name": "AV1M",
            "root_path": str(ROOT / "data" / "avh_features"),
            "csv_root_path": str(SUP / "csv_metadata" / "av1m"),
            "apply_l2": True,
        },
        "favc_domain_info": {
            "root_path": str(ROOT / "data" / "favc_features"),
            "csv_root_path": str(SUP / "csv_metadata" / "favc_npz_matched"),
            "split": "trainval",
            "real_only": True,
            "apply_l2": True,
        },
        "model_hparams": hparams,
        "callbacks": {
            "logger": {
                "log_path": str(out_dir / "logs"),
                "name": "csv",
            },
            "ckpt_args": {
                "ckpt_dir": str(out_dir / "ckpts"),
                "metric": "val_auc_causal",
                "mode": "max",
            },
        },
        "epochs": 20,
        "seed": seed,
        "early_stopping": {
            "metric": "val_auc_causal",
            "mode": "max",
            "patience": 20,
        },
    }


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for variant, settings in VARIANTS.items():
        for seed in SEEDS:
            cfg = _base_config(variant, seed, settings)
            out_path = CONFIG_DIR / f"A6_trainvalreal_{variant}_seed{seed}.yaml"
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(out_path)


if __name__ == "__main__":
    main()
