#!/usr/bin/env python3
"""Train the locked national models sequentially and preserve every run."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "vendor" / "neuralhydrology_literature_adapter"
PYTHON = Path(sys.executable)
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)


def matching(model: str, seed: int) -> tuple[list[Path], list[Path]]:
    prefix = f"national_literature_{model}_seed{seed}_"
    complete, partial = [], []
    for path in sorted((ROOT / "runs").glob(prefix + "*")):
        checkpoint = path / "model_epoch030.pt"
        (complete if checkpoint.exists() else partial).append(path)
    return complete, partial


def train(model: str, seed: int) -> None:
    complete, partial = matching(model, seed)
    if len(complete) > 1:
        raise RuntimeError(f"Multiple complete runs for {model} seed {seed}: {complete}")
    if complete:
        print(f"SKIP {model} seed {seed}: {complete[0]}", flush=True)
        return
    if partial:
        raise RuntimeError(f"Inspect partial run before continuing: {partial}")
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 10:
        raise RuntimeError(f"Only {free:.1f} GiB free; stopping before training")
    config = ROOT / "configs" / f"full_{model}_seed{seed}.yml"
    loaded = yaml.safe_load(config.read_text())
    if loaded["seed"] != seed:
        raise RuntimeError(f"Config seed mismatch: {config}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ADAPTER)
    environment["MPLCONFIGDIR"] = "/private/tmp/nh_mpl_cache"
    if model == "author_dhbv":
        environment["NGEN_SILENT"] = "1"
    command = [
        str(PYTHON), "-m", "neuralhydrology.nh_run", "train",
        "--config-file", str(config),
    ]
    print(f"START {model} seed {seed}; free {free:.1f} GiB", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    complete, partial = matching(model, seed)
    if len(complete) != 1:
        raise RuntimeError(
            f"Training ended without one epoch-30 run: complete={complete}, partial={partial}"
        )
    print(f"DONE {model} seed {seed}: {complete[0]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all",) + MODELS, default="all")
    parser.add_argument("--seed", choices=("11", "29", "47", "all"), default="11")
    args = parser.parse_args()
    models = MODELS if args.model == "all" else (args.model,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)
    for model in models:
        for seed in seeds:
            train(model, seed)


if __name__ == "__main__":
    main()
