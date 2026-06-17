#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BASE = Path("/data/projects/punim2637/nnliang/AVH-Align")
SUP = BASE / "avh_sup"
BASE_CONFIG = SUP / "configs" / "A6_av1m_pseudodomain_v2_50ep_seed45.yaml"
CONFIG_OUT = SUP / "configs" / "reviewer_followups_seed45"
SLURM_OUT = BASE / "generated_slurm" / "reviewer_followups_seed45"
LOG_DIR = BASE / "logs"

FEATURES = BASE / "data" / "favc_features"
CSV_CANONICAL = SUP / "csv_metadata" / "favc_7030_canonical"

SEED = 45
PARTITIONS = "gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,gpu-a100"


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "lambda_dadv0p3",
        "description": "lambda_adv / lambda_dadv = 0.3",
        "model_updates": {"lambda_dadv": 0.3},
    },
    {
        "name": "lambda_dadv3p0",
        "description": "lambda_adv / lambda_dadv = 3.0",
        "model_updates": {"lambda_dadv": 3.0},
    },
    {
        "name": "lambda_ddis0p3",
        "description": "lambda_dom / lambda_ddis = 0.3",
        "model_updates": {"lambda_ddis": 0.3},
    },
    {
        "name": "lambda_ddis3p0",
        "description": "lambda_dom / lambda_ddis = 3.0",
        "model_updates": {"lambda_ddis": 3.0},
    },
    {
        "name": "lambda_orth0p03",
        "description": "lambda_orth = 0.03",
        "model_updates": {"lambda_orth": 0.03},
    },
    {
        "name": "lambda_orth0p3",
        "description": "lambda_orth = 0.3",
        "model_updates": {"lambda_orth": 0.3},
    },
    {
        "name": "domain_random_clip",
        "description": "random binary pseudo-domain per clip",
        "data_updates": {"domain_label_mode": "random_clip", "domain_label_seed": SEED},
    },
    {
        "name": "domain_random_speaker",
        "description": "random binary pseudo-domain per speaker",
        "data_updates": {"domain_label_mode": "random_speaker", "domain_label_seed": SEED},
    },
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def build_config(exp: dict[str, Any]) -> tuple[Path, Path]:
    cfg = deepcopy(load_yaml(BASE_CONFIG))
    out_dir = SUP / f"outputs_A6_pdv2_s45_{exp['name']}"
    cfg["ablation_id"] = f"A6_pdv2_s45_{exp['name']}"
    cfg["seed"] = SEED
    cfg["epochs"] = 50
    cfg.setdefault("early_stopping", {})["patience"] = 50

    cfg["callbacks"]["logger"]["log_path"] = str(out_dir / "logs")
    cfg["callbacks"]["ckpt_args"]["ckpt_dir"] = str(out_dir / "ckpts")
    cfg["favc_eval_info"]["best_ckpt_dir"] = str(out_dir / "ckpts" / "favc_fvra")
    cfg["favc_eval_info"]["check_every_n_epochs"] = 5

    cfg["data_info"].setdefault("domain_label_mode", "speaker_hash")
    cfg["data_info"].setdefault("domain_label_seed", SEED)
    cfg["data_info"].update(exp.get("data_updates", {}))
    cfg["model_hparams"].update(exp.get("model_updates", {}))

    cfg_path = CONFIG_OUT / f"{cfg['ablation_id']}.yaml"
    save_yaml(cfg_path, cfg)
    return cfg_path, out_dir


def slurm_text(exp: dict[str, Any], cfg_path: Path, out_dir: Path) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=rv_{exp['name']}
        #SBATCH --partition={PARTITIONS}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:1
        #SBATCH --mem=64G
        #SBATCH --time=24:00:00
        #SBATCH --output={LOG_DIR}/%x_%j.out
        #SBATCH --error={LOG_DIR}/%x_%j.err

        set -euo pipefail

        echo "============================================================"
        echo "  Reviewer follow-up: {exp['description']}"
        echo "  Seed: {SEED}"
        echo "  Config: {cfg_path}"
        echo "  Output: {out_dir}"
        echo "  Start: $(date)"
        echo "============================================================"

        cd {BASE}
        module purge
        module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/projects/punim2637/nnliang/avh_env
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        export PYTHONUNBUFFERED=1

        mkdir -p "{out_dir / 'logs'}" "{out_dir / 'ckpts'}" "{LOG_DIR}"

        cd {SUP}
        python train_test_fcd_a6_av1m_pseudodomain_v2.py \\
          --config_path "{cfg_path}" \\
          --seed {SEED}

        BEST_TXT="{out_dir / 'ckpts' / 'favc_fvra' / 'best_favc_fvra.txt'}"
        if [[ -f "$BEST_TXT" ]]; then
          CKPT=$(grep '^checkpoint=' "$BEST_TXT" | cut -d= -f2-)
        elif [[ -f "{out_dir / 'ckpts' / 'last.ckpt'}" ]]; then
          CKPT="{out_dir / 'ckpts' / 'last.ckpt'}"
        else
          CKPT=$(ls "{out_dir / 'ckpts'}"/*.ckpt 2>/dev/null | sort | tail -1)
        fi

        if [[ -z "${{CKPT:-}}" || ! -f "$CKPT" ]]; then
          echo "ERROR: No checkpoint found for {exp['name']}" >&2
          exit 2
        fi

        echo "  Eval checkpoint: $CKPT"
        python eval_favc.py \\
          --ckpt "$CKPT" \\
          --model fcd_a6 \\
          --features_path "{FEATURES}" \\
          --csv_root_path "{CSV_CANONICAL}" \\
          --split test \\
          --output_dir "{out_dir / 'results_favc_7030_full_best'}"

        echo "============================================================"
        echo "  Done: $(date)"
        echo "============================================================"
        """
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def submit(path: Path) -> str:
    result = subprocess.run(["sbatch", "--parsable", str(path)], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch seed45 reviewer follow-up experiments")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Subset of experiment names")
    args = parser.parse_args()

    selected = [e for e in EXPERIMENTS if args.only is None or e["name"] in set(args.only)]
    rows = []
    for exp in selected:
        cfg_path, out_dir = build_config(exp)
        slurm_path = SLURM_OUT / f"{exp['name']}.slurm"
        write_text(slurm_path, slurm_text(exp, cfg_path, out_dir))
        row = {
            "experiment": exp["name"],
            "config": str(cfg_path),
            "slurm": str(slurm_path),
            "output": str(out_dir),
        }
        if args.submit:
            row["job_id"] = submit(slurm_path)
        rows.append(row)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
