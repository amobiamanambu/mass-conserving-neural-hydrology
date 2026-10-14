#!/usr/bin/env python3
"""Run the frozen training, evaluation, and assessment matrix reproducibly.

The scientific work is implemented in scripts 03--07.  This launcher only
enumerates the prespecified architectures, observation rungs, and seeds and
runs them sequentially.  Sequential execution avoids platform-specific
OpenMP conflicts and does not alter model configuration or numerical results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
ARCHITECTURES = ("dhbv", "fluxgate")
RUNGS = ("q", "q_sm", "q_et", "q_sm_et")
SEEDS = (11, 29, 47)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("train", "evaluate", "assess", "validate", "all"),
        default="all",
    )
    parser.add_argument(
        "--threads", type=int, choices=(1,), default=1,
        help="Pinned to one on the archived macOS environment to avoid an OpenMP runtime conflict.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.update({
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
    })
    subprocess.run(command, cwd=PROJECT.parent, env=environment, check=True)


def training() -> None:
    for architecture in ARCHITECTURES:
        for rung in RUNGS:
            for seed in SEEDS:
                run([
                    sys.executable, str(SCRIPTS / "04_train_daily_models.py"),
                    "--architecture", architecture, "--rung", rung,
                    "--seed", str(seed), "--threads", str(args.threads),
                ])


def evaluation() -> None:
    for architecture in ARCHITECTURES:
        for rung in RUNGS:
            for seed in SEEDS:
                run([
                    sys.executable, str(SCRIPTS / "05_predict_and_intervene.py"),
                    "--run", f"{architecture}_{rung}_seed{seed}",
                    "--threads", str(args.threads),
                ])


def assessment() -> None:
    run([sys.executable, str(SCRIPTS / "06_assess_observability.py")])


def validation() -> None:
    run([sys.executable, str(SCRIPTS / "07_validate_archive.py")])


if __name__ == "__main__":
    args = arguments()
    if args.stage in ("train", "all"):
        training()
    if args.stage in ("evaluate", "all"):
        evaluation()
    if args.stage in ("assess", "all"):
        assessment()
    if args.stage in ("validate", "all"):
        validation()
