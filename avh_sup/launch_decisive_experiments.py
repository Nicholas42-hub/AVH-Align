#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BASE = Path("/data/projects/punim2637/nnliang/AVH-Align")
SUP = BASE / "avh_sup"
ENV_PYTHON = Path("/data/projects/punim2637/nnliang/avh_env/bin/python")
CONFIG_OUT = SUP / "configs" / "decisive"
SLURM_OUT = BASE / "generated_slurm" / "decisive"
LOG_DIR = BASE / "logs"

AV1M_RAW = "/scratch/punim2637/nnliang/avd1m_preprocessed/val"
AV1M_CSV = str(SUP / "csv_metadata" / "av1m_e2e")
FAVC_RAW = "/scratch/punim2637/nnliang/favc_preprocessed"
FULL_MAX_CLIPS = 1000000


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def output_paths(exp_name: str) -> dict[str, str]:
    base = SUP / f"outputs_{exp_name}"
    return {
        "output_root": str(base),
        "ckpt_dir": str(base / "ckpts"),
        "log_path": str(base / "logs"),
        "output_path": str(base / "results"),
    }


def build_config(
    base_config_name: str,
    exp_name: str,
    seed: int,
    overrides: dict[str, Any],
) -> Path:
    cfg = load_yaml(SUP / "configs" / base_config_name)
    paths = output_paths(exp_name)
    deep_update(
        cfg,
        {
            "ablation_id": exp_name,
            "seed": seed,
            "callbacks": {
                "logger": {"log_path": paths["log_path"]},
                "ckpt_args": {"ckpt_dir": paths["ckpt_dir"]},
            },
            "output_path": paths["output_path"],
        },
    )
    deep_update(cfg, deepcopy(overrides))
    out_path = CONFIG_OUT / f"{exp_name}.yaml"
    save_yaml(out_path, cfg)
    return out_path


def write_slurm(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def submit(script_path: Path, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(script_path))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def train_script(
    job_name: str,
    train_py: str,
    config_path: Path,
    partition: str,
    mem_gb: int,
    hours: int,
) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition={partition}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:1
        #SBATCH --mem={mem_gb}G
        #SBATCH --time={hours}:00:00
        #SBATCH --output={LOG_DIR}/%x_%j.out
        #SBATCH --error={LOG_DIR}/%x_%j.err

        cd {BASE}
        module purge
        module load Miniconda3/23.10.0-1
        module load GCC/11.3.0
        module load CUDA/12.2.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/projects/punim2637/nnliang/avh_env
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        export PYTHONUNBUFFERED=1
        export CUDA_VISIBLE_DEVICES=0
        PYTHON={ENV_PYTHON}

        mkdir -p {LOG_DIR}
        cd {SUP}
        echo ">>> {job_name} starting on $(hostname) at $(date)"
        $PYTHON {train_py} --config_path "{config_path}"
        echo ">>> {job_name} finished at $(date)"
        """
    )


def eval_script(
    job_name: str,
    config_path: Path,
    ckpt_dir: Path,
    model_module: str,
    model_class: str,
    output_dir: Path,
    eval_kind: str,
    max_clips: int,
) -> str:
    eval_py = "eval_e2e_auc.py" if eval_kind == "auc" else "eval_e2e_domain.py"
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --partition=gpu-a100
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --gres=gpu:1
        #SBATCH --mem=32G
        #SBATCH --time=06:00:00
        #SBATCH --output={LOG_DIR}/%x_%j.out
        #SBATCH --error={LOG_DIR}/%x_%j.err

        cd {BASE}
        module purge
        module load Miniconda3/23.10.0-1
        module load GCC/11.3.0
        module load CUDA/12.2.0
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate /data/projects/punim2637/nnliang/avh_env
        export PYTHONNOUSERSITE=1
        unset PYTHONPATH
        export PYTHONUNBUFFERED=1
        export CUDA_VISIBLE_DEVICES=0
        PYTHON={ENV_PYTHON}

        mkdir -p {output_dir} {LOG_DIR}
        cd {SUP}
        BEST=$(ls -t "{ckpt_dir}"/model-epoch=*.ckpt 2>/dev/null | grep -v last | head -1)
        CKPT="${{BEST:-{ckpt_dir}/last.ckpt}}"
        echo ">>> {job_name} using checkpoint: $CKPT"
        $PYTHON {eval_py} \\
            --model_module {model_module} \\
            --model_class {model_class} \\
            --use_fullpath_csv \\
            --ckpt "$CKPT" \\
            --config "{config_path}" \\
            --av1m_raw {AV1M_RAW} \\
            --av1m_csv "{AV1M_CSV}" \\
            --favc_raw {FAVC_RAW} \\
            --output_dir "{output_dir}" \\
            --max_clips {max_clips} \\
            --batch_size 4 \\
            --num_workers 2
        """
    )


def train_family(
    exp_name: str,
    base_config: str,
    train_py: str,
    model_module: str,
    model_class: str,
    seeds: list[int],
    overrides: dict[str, Any],
    mem_gb: int = 64,
    hours: int = 24,
    partition: str = "gpu-a100",
) -> list[dict[str, str]]:
    submitted: list[dict[str, str]] = []
    for seed in seeds:
        full_name = f"{exp_name}_seed{seed}"
        cfg_path = build_config(base_config, full_name, seed, overrides)
        ckpt_dir = Path(output_paths(full_name)["ckpt_dir"])
        train_slurm = SLURM_OUT / f"train_{full_name}.slurm"
        write_slurm(
            train_slurm,
            train_script(
                job_name=f"tr_{exp_name}_{seed}",
                train_py=train_py,
                config_path=cfg_path,
                partition=partition,
                mem_gb=mem_gb,
                hours=hours,
            ),
        )
        train_job = submit(train_slurm)

        eval_jobs = {}
        for kind, clips, suffix in [
            ("auc", 500, "probe500_auc"),
            ("domain", 500, "probe500_domain"),
            ("auc", FULL_MAX_CLIPS, "full_auc"),
        ]:
            eval_slurm = SLURM_OUT / f"eval_{full_name}_{suffix}.slurm"
            eval_out = SUP / f"outputs_{full_name}" / f"results_{suffix}"
            write_slurm(
                eval_slurm,
                eval_script(
                    job_name=f"ev_{exp_name}_{seed}_{suffix}",
                    config_path=cfg_path,
                    ckpt_dir=ckpt_dir,
                    model_module=model_module,
                    model_class=model_class,
                    output_dir=eval_out,
                    eval_kind=kind,
                    max_clips=clips,
                ),
            )
            dep = f"afterok:{train_job}"
            eval_jobs[suffix] = submit(eval_slurm, dependency=dep)

        submitted.append(
            {
                "experiment": full_name,
                "config": str(cfg_path),
                "train_job": train_job,
                **{f"eval_{k}": v for k, v in eval_jobs.items()},
            }
        )
    return submitted


def eval_existing(
    exp_name: str,
    config_path: Path,
    ckpt_dir: Path,
    model_module: str,
    model_class: str,
) -> dict[str, str]:
    submitted: dict[str, str] = {"experiment": exp_name}
    for kind, clips, suffix in [
        ("auc", 500, "probe500_auc"),
        ("domain", 500, "probe500_domain"),
        ("auc", FULL_MAX_CLIPS, "full_auc"),
    ]:
        eval_slurm = SLURM_OUT / f"eval_existing_{exp_name}_{suffix}.slurm"
        eval_out = SUP / f"outputs_{exp_name}" / f"results_{suffix}"
        write_slurm(
            eval_slurm,
            eval_script(
                job_name=f"ev_{exp_name}_{suffix}",
                config_path=config_path,
                ckpt_dir=ckpt_dir,
                model_module=model_module,
                model_class=model_class,
                output_dir=eval_out,
                eval_kind=kind,
                max_clips=clips,
            ),
        )
        submitted[f"eval_{suffix}"] = submit(eval_slurm)
    return submitted


def main() -> None:
    CONFIG_OUT.mkdir(parents=True, exist_ok=True)
    SLURM_OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    submitted: list[dict[str, str]] = []

    submitted.append(
        eval_existing(
            exp_name="A40_e2e",
            config_path=SUP / "configs" / "A40_e2e.yaml",
            ckpt_dir=SUP / "outputs_A40_e2e" / "ckpts",
            model_module="train_e2e_a40",
            model_class="E2EModelA40",
        )
    )
    submitted.append(
        eval_existing(
            exp_name="A2_e2e_dadv",
            config_path=SUP / "configs" / "A2_e2e_dadv.yaml",
            ckpt_dir=SUP / "outputs_A2_e2e_dadv" / "ckpts",
            model_module="train_e2e_a1_dadv",
            model_class="E2EModelA1DADV",
        )
    )
    submitted.append(
        eval_existing(
            exp_name="A34_e2e",
            config_path=SUP / "configs" / "A34_e2e.yaml",
            ckpt_dir=SUP / "outputs_A34_e2e" / "ckpts",
            model_module="train_e2e_a34",
            model_class="E2EModelA34",
        )
    )
    submitted.append(
        eval_existing(
            exp_name="A41_e2e",
            config_path=SUP / "configs" / "A41_e2e.yaml",
            ckpt_dir=SUP / "outputs_A41_e2e" / "ckpts",
            model_module="train_e2e_a41",
            model_class="E2EModelA41",
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_baseline_samepipe",
            base_config="A40_e2e.yaml",
            train_py="train_e2e_a40.py",
            model_module="train_e2e_a40",
            model_class="E2EModelA40",
            seeds=[42, 43, 44],
            overrides={
                "model_hparams": {
                    "lambda_ladv_ds": 0.0,
                    "lambda_mean_zc": 0.0,
                },
            },
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_A40",
            base_config="A40_e2e.yaml",
            train_py="train_e2e_a40.py",
            model_module="train_e2e_a40",
            model_class="E2EModelA40",
            seeds=[42, 44],
            overrides={},
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_A41lite1",
            base_config="A41_e2e.yaml",
            train_py="train_e2e_a41.py",
            model_module="train_e2e_a41",
            model_class="E2EModelA41",
            seeds=[42, 43, 44],
            overrides={"model_hparams": {"lambda_adv": 1.0}},
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_A41lite3",
            base_config="A41_e2e.yaml",
            train_py="train_e2e_a41.py",
            model_module="train_e2e_a41",
            model_class="E2EModelA41",
            seeds=[42, 43, 44],
            overrides={"model_hparams": {"lambda_adv": 3.0}},
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_A2e2e_dadv",
            base_config="A2_e2e_dadv.yaml",
            train_py="train_e2e_a1_dadv.py",
            model_module="train_e2e_a1_dadv",
            model_class="E2EModelA1DADV",
            seeds=[42, 44],
            overrides={},
            mem_gb=80,
            hours=48,
        )
    )

    submitted.extend(
        train_family(
            exp_name="decisive_A34e2e",
            base_config="A34_e2e.yaml",
            train_py="train_e2e_a34.py",
            model_module="train_e2e_a34",
            model_class="E2EModelA34",
            seeds=[42, 44],
            overrides={},
        )
    )

    summary_path = SLURM_OUT / "submitted_jobs.json"
    with summary_path.open("w") as fh:
        json.dump(submitted, fh, indent=2)

    print(f"Wrote submission summary to {summary_path}")
    for item in submitted:
        print(json.dumps(item, sort_keys=True))


if __name__ == "__main__":
    main()
