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
CONFIG_TEMPLATE = SUP / "configs" / "DANN.yaml"
CONFIG_OUT = SUP / "configs" / "alpha_sweep"
SLURM_OUT = BASE / "generated_slurm" / "dann_alpha_sweep"
LOG_DIR = BASE / "logs"

FEATURES = BASE / "data" / "favc_features"
CSV_CANONICAL = SUP / "csv_metadata" / "favc_7030_canonical"

ALPHAS = [0.1, 0.3, 0.5, 1.0, 3.0]
SEEDS = [43, 44, 45]
DEFAULT_TRAIN_PARTITIONS = "gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,deeplearn,feit-gpu-a100,gpu-a100"
DEFAULT_EVAL_PARTITIONS = "gpu-l40s-preempt,gpu-a100-preempt,gpu-l40s,gpu-a100-short,gpu-a100"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def build_config(alpha: float, seed: int) -> tuple[str, Path, Path]:
    tag = alpha_tag(alpha)
    exp_name = f"DANN_alpha{tag}_seed{seed}"
    output_root = SUP / f"outputs_{exp_name}"

    cfg = deepcopy(load_yaml(CONFIG_TEMPLATE))
    cfg["seed"] = seed
    cfg["model_hparams"]["grl_alpha"] = float(alpha)
    cfg["callbacks"]["logger"]["log_path"] = str(output_root / "logs")
    cfg["callbacks"]["ckpt_args"]["ckpt_dir"] = str(output_root / "ckpts")

    cfg_path = CONFIG_OUT / f"{exp_name}.yaml"
    save_yaml(cfg_path, cfg)
    return exp_name, cfg_path, output_root


def train_script(exp_name: str, cfg_path: Path, output_root: Path, partitions: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=tr_{exp_name}
        #SBATCH --partition={partitions}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:1
        #SBATCH --mem=64G
        #SBATCH --time=24:00:00
        #SBATCH --output={LOG_DIR}/%x_%j.out
        #SBATCH --error={LOG_DIR}/%x_%j.err

        set -euo pipefail
        cd {BASE}
        module purge
        module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/projects/punim2637/nnliang/avh_env
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        export PYTHONUNBUFFERED=1

        mkdir -p {output_root / 'logs'} {output_root / 'ckpts'} {LOG_DIR}
        cd {SUP}
        python train_da_baselines.py \
            --method dann \
            --config_path "{cfg_path}" \
            --seed {load_yaml(cfg_path)['seed']} \
            --output_dir "{output_root}"
        """
    )


def eval_script(exp_name: str, output_root: Path, partitions: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name=ev_{exp_name}
        #SBATCH --partition={partitions}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --gres=gpu:1
        #SBATCH --mem=32G
        #SBATCH --time=01:00:00
        #SBATCH --output={LOG_DIR}/%x_%j.out
        #SBATCH --error={LOG_DIR}/%x_%j.err

        set -euo pipefail
        cd {BASE}
        module purge
        module load Miniconda3/23.10.0-1 GCC/11.3.0 CUDA/12.2.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/projects/punim2637/nnliang/avh_env
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        export PYTHONUNBUFFERED=1

        cd {SUP}
        CKPT=$(ls "{output_root / 'ckpts'}"/*.ckpt 2>/dev/null | sort | tail -1)
        if [[ -z "$CKPT" ]]; then
            echo "No checkpoint found for {exp_name}" >&2
            exit 2
        fi

        python eval_favc.py \
            --ckpt "$CKPT" \
            --model dann \
            --features_path "{FEATURES}" \
            --csv_root_path "{CSV_CANONICAL}" \
            --split test \
            --output_dir "{output_root / 'results_favc_7030_full'}"
        """
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def submit(path: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(path))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch isolated DANN GRL alpha sweep")
    parser.add_argument("--submit", action="store_true", help="Submit jobs with sbatch")
    parser.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--train_partitions", default=DEFAULT_TRAIN_PARTITIONS)
    parser.add_argument("--eval_partitions", default=DEFAULT_EVAL_PARTITIONS)
    args = parser.parse_args()

    summary: list[dict[str, str | float | int]] = []
    for alpha in args.alphas:
        for seed in args.seeds:
            exp_name, cfg_path, output_root = build_config(alpha, seed)
            train_slurm = SLURM_OUT / f"train_{exp_name}.slurm"
            eval_slurm = SLURM_OUT / f"eval_{exp_name}.slurm"
            write_text(train_slurm, train_script(exp_name, cfg_path, output_root, args.train_partitions))
            write_text(eval_slurm, eval_script(exp_name, output_root, args.eval_partitions))

            record: dict[str, str | float | int] = {
                "experiment": exp_name,
                "alpha": alpha,
                "seed": seed,
                "config": str(cfg_path),
                "train_slurm": str(train_slurm),
                "eval_slurm": str(eval_slurm),
            }
            if args.submit:
                train_job = submit(train_slurm)
                eval_job = submit(eval_slurm, dependency=f"afterok:{train_job}")
                record["train_job"] = train_job
                record["eval_job"] = eval_job
            summary.append(record)

    for row in summary:
        print(row)


if __name__ == "__main__":
    main()