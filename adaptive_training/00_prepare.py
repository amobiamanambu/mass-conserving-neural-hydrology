#!/usr/bin/env python3
"""Prepare adaptive training from the archived epoch-30 runs.

The release archive places the model-ready data, vendored model code, initial
runs, and frozen MC-LSTM products at the repository root. This script links
those shared inputs and copies each admitted lineage's epoch-30 state into the
isolated adaptive-training tree. It never overwrites an existing file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        return
    link_path.symlink_to(os.path.relpath(target, link_path.parent), target_is_directory=target.is_dir())


def copy_run(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name.startswith("continue_training_from_epoch"):
            continue
        if item.is_dir():
            shutil.copytree(item, destination / item.name)
        else:
            shutil.copy2(item, destination / item.name)


def source_run(model: str, seed: int) -> Path:
    candidates = [
        path
        for path in sorted((PROJECT / "runs").glob(f"national_literature_{model}_seed{seed}_*"))
        if (path / "model_epoch030.pt").exists()
        and (path / "optimizer_state_epoch030.pt").exists()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one complete epoch-30 run for {model} seed {seed}; found {candidates}"
        )
    return candidates[0]


def main() -> None:
    link(PROJECT / "data", HERE / "data")
    link(PROJECT / "vendor", HERE / "vendor")

    frozen = HERE / "outputs/frozen"
    for path in (PROJECT / "outputs/frozen").glob("mclstm_seed*"):
        link(path, frozen / path.name)
    for path in (PROJECT / "outputs/sensitivity").glob("mclstm_seed11_batch_sensitivity.*"):
        link(path, HERE / "outputs/sensitivity" / path.name)
    for run in (PROJECT / "runs").glob("national_literature_mclstm_seed*"):
        link(run, HERE / "runs" / run.name)

    receipt = {"status": "prepared", "source": "runs/ archived epoch-30 states", "models": {}}
    for model in MODELS:
        receipt["models"][model] = {}
        for seed in SEEDS:
            source = source_run(model, seed)
            destination = HERE / "runs_adaptive" / model / f"seed{seed}" / "base_run"
            copy_run(source, destination)
            for name in ("model_epoch030.pt", "optimizer_state_epoch030.pt"):
                if sha256(source / name) != sha256(destination / name):
                    raise RuntimeError(f"Epoch-30 state differs after copy: {destination / name}")
            receipt["models"][model][str(seed)] = {
                "epoch30_checkpoint_sha256": sha256(destination / "model_epoch030.pt"),
                "epoch30_optimizer_sha256": sha256(destination / "optimizer_state_epoch030.pt"),
            }
    (HERE / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Prepared {HERE}")


if __name__ == "__main__":
    main()
