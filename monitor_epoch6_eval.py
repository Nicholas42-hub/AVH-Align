#!/usr/bin/env python3
"""
Monitor decisive training jobs; submit probe500_auc + probe500_domain + full_auc
evaluations as soon as each experiment finishes epoch 6 (log marker [Epoch 05]).

Usage (run in background):
    nohup python monitor_epoch6_eval.py &> logs/monitor_epoch6_eval.log &
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE      = Path("/data/projects/punim2637/nnliang/AVH-Align")
SUP       = BASE / "avh_sup"
ENV_PYTHON = Path("/data/projects/punim2637/nnliang/avh_env/bin/python")
LOG_DIR   = BASE / "logs"
SLURM_OUT = BASE / "generated_slurm" / "decisive_ep6"
STATE_FILE = BASE / "generated_slurm" / "decisive_ep6" / "submitted_ep6.json"

AV1M_RAW  = "/scratch/punim2637/nnliang/avd1m_preprocessed/val"
AV1M_CSV  = str(SUP / "csv_metadata" / "av1m_e2e")
FAVC_RAW  = "/scratch/punim2637/nnliang/favc_preprocessed"

# Epoch index that triggers eval (0-based; 5 = "6 epochs done")
TARGET_EPOCH_IDX = 5
EPOCH_MARKER     = f"[Epoch {TARGET_EPOCH_IDX:02d}]"

POLL_INTERVAL = 120  # seconds between checks

# Map experiment family prefix → (model_module, model_class)
MODEL_MAP = {
    "decisive_baseline_samepipe": ("train_e2e_a40",    "E2EModelA40"),
    "decisive_A40":               ("train_e2e_a40",    "E2EModelA40"),
    "decisive_A41lite1":          ("train_e2e_a41",    "E2EModelA41"),
    "decisive_A41lite3":          ("train_e2e_a41",    "E2EModelA41"),
    "decisive_A2e2e_dadv":        ("train_e2e_a1_dadv","E2EModelA1DADV"),
    "decisive_A34e2e":            ("train_e2e_a34",    "E2EModelA34"),
}


# ── helpers ────────────────────────────────────────────────────────────────

def parse_exp(full_exp_name: str):
    """Return (family, seed_int) from e.g. 'decisive_A40_seed42'."""
    m = re.match(r"^(.+)_seed(\d+)$", full_exp_name)
    if m:
        return m.group(1), int(m.group(2))
    return full_exp_name, None


def find_log(family: str, seed: int, job_id: str) -> Path | None:
    """Reconstruct the training log path."""
    # Log is named: tr_{family}_{seed}_{job_id}.out
    candidate = LOG_DIR / f"tr_{family}_{seed}_{job_id}.out"
    if candidate.exists():
        return candidate
    # Fallback: glob for any file containing the job id
    matches = list(LOG_DIR.glob(f"*_{job_id}.out"))
    return matches[0] if matches else None


def log_has_epoch(log_path: Path, marker: str) -> bool:
    """Return True if marker appears in the log file."""
    try:
        with log_path.open() as fh:
            for line in fh:
                if marker in line:
                    return True
    except OSError:
        pass
    return False


def get_model(family: str):
    for prefix, pair in MODEL_MAP.items():
        if family.startswith(prefix):
            return pair
    raise ValueError(f"No model mapping for family: {family}")


def write_eval_slurm(
    full_exp_name: str,
    config_path: Path,
    ckpt_dir: Path,
    model_module: str,
    model_class: str,
    output_dir: Path,
    eval_kind: str,
    max_clips: int,
    suffix: str,
    epoch_idx: int,
) -> Path:
    """Write a SLURM eval script that targets the epoch-6 checkpoint."""
    family, seed = parse_exp(full_exp_name)
    job_name = f"ev_{family}_{seed}_ep{epoch_idx+1}_{suffix}"[:60]
    eval_py  = "eval_e2e_auc.py" if eval_kind == "auc" else "eval_e2e_domain.py"

    # Checkpoint selection: prefer epoch=05 checkpoint, fall back to last.ckpt
    ckpt_glob = f"model-epoch={epoch_idx:02d}-*.ckpt"

    script = textwrap.dedent(f"""\
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
        CKPT=$(ls "{ckpt_dir}"/""" + ckpt_glob + """ 2>/dev/null | head -1)
        CKPT="${CKPT:-""" + str(ckpt_dir / "last.ckpt") + """}"
        echo ">>> """ + job_name + """ using checkpoint: $CKPT"
        $PYTHON """ + eval_py + """ \\
            --model_module """ + model_module + """ \\
            --model_class """ + model_class + """ \\
            --use_fullpath_csv \\
            --ckpt "$CKPT" \\
            --config \"""" + str(config_path) + """\" \\
            --av1m_raw """ + AV1M_RAW + """ \\
            --av1m_csv \"""" + AV1M_CSV + """\" \\
            --favc_raw """ + FAVC_RAW + """ \\
            --output_dir \"""" + str(output_dir) + """\" \\
            --max_clips """ + str(max_clips) + """ \\
            --batch_size 4 \\
            --num_workers 2
        """)

    script_path = SLURM_OUT / f"eval_{full_exp_name}_ep{epoch_idx+1}_{suffix}.slurm"
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def submit(script_path: Path) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open() as fh:
            return json.load(fh)
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as fh:
        json.dump(state, fh, indent=2)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    SLURM_OUT.mkdir(parents=True, exist_ok=True)

    jobs_file = BASE / "generated_slurm" / "decisive" / "submitted_jobs.json"
    with jobs_file.open() as fh:
        submitted_jobs = json.load(fh)

    # Keep only experiments that have a training job
    train_exps = [e for e in submitted_jobs if "train_job" in e]
    print(f"Monitoring {len(train_exps)} training experiments for {EPOCH_MARKER}")

    state = load_state()  # tracks which exps have had ep6 evals submitted

    while True:
        remaining = [e for e in train_exps if e["experiment"] not in state]
        if not remaining:
            print("All experiments have had epoch-6 evals submitted. Exiting.")
            break

        for exp in remaining:
            full_name = exp["experiment"]
            train_job = exp["train_job"]
            config_path = Path(exp["config"])

            family, seed = parse_exp(full_name)
            log_path = find_log(family, seed, train_job)

            if log_path is None:
                print(f"[{full_name}] Log not found yet, skipping...")
                continue

            if not log_has_epoch(log_path, EPOCH_MARKER):
                print(f"[{full_name}] Not yet at {EPOCH_MARKER}")
                continue

            # Epoch 6 reached – submit evals
            print(f"\n[{full_name}] {EPOCH_MARKER} detected! Submitting epoch-6 evals...")
            model_module, model_class = get_model(family)
            ckpt_dir = SUP / f"outputs_{full_name}" / "ckpts"
            out_base  = SUP / f"outputs_{full_name}"

            submitted = {"experiment": full_name}
            for eval_kind, max_clips, suffix in [
                ("auc",    500,        "ep6_probe500_auc"),
                ("domain", 500,        "ep6_probe500_domain"),
                ("auc",    1_000_000,  "ep6_full_auc"),
            ]:
                script = write_eval_slurm(
                    full_exp_name=full_name,
                    config_path=config_path,
                    ckpt_dir=ckpt_dir,
                    model_module=model_module,
                    model_class=model_class,
                    output_dir=out_base / f"results_{suffix}",
                    eval_kind=eval_kind,
                    max_clips=max_clips,
                    suffix=suffix,
                    epoch_idx=TARGET_EPOCH_IDX,
                )
                job_id = submit(script)
                submitted[f"eval_{suffix}"] = job_id
                print(f"  Submitted {suffix}: job {job_id}")

            state[full_name] = submitted
            save_state(state)

        remaining_count = len([e for e in train_exps if e["experiment"] not in state])
        if remaining_count == 0:
            print("All experiments covered. Exiting.")
            break

        print(f"\n[{remaining_count} remaining] Sleeping {POLL_INTERVAL}s...\n")
        time.sleep(POLL_INTERVAL)

    # Print summary
    print("\n=== Epoch-6 eval submission summary ===")
    for exp_name, info in state.items():
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
