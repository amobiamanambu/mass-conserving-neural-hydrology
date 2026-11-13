#!/usr/bin/env python3
"""Freeze the development-only checkpoint selection for the analysis chain."""

from __future__ import annotations

import hashlib
import json
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


def checkpoint_run(model: str, seed: int, epoch: int) -> Path:
    base = HERE / "runs_adaptive" / model / f"seed{seed}" / "base_run"
    found = [p.parent for p in base.glob(f"**/model_epoch{epoch:03d}.pt")]
    if len(found) != 1:
        raise RuntimeError(f"Expected one epoch-{epoch} checkpoint for {model} seed {seed}; found {found}")
    return found[0]


def main() -> None:
    receipt = {
        "selection_data": "200 development gauges, 2015-2018",
        "target_population": "3,982 unseen gauges, 2019-2024",
        "target_discharge_accessed_for_stopping_or_selection": False,
        "target_discharge_accessed": False,
        "post_opening_sensitivity": True,
        "selection_seed": 11,
        "selection_rule": "adaptive learning-rate rule of PROTOCOL.md; highest median development KGE, later epoch breaks a tie",
        "architectures": {},
    }
    for model in MODELS:
        adaptive = json.loads((HERE / f"adaptive_receipt_{model}.json").read_text())
        if adaptive["status"] == "running":
            raise RuntimeError(f"Seed-11 training for {model} has not finished")
        epoch = int(adaptive["selected_epoch"])
        runs = {}
        for seed in SEEDS:
            run = checkpoint_run(model, seed, epoch)
            checkpoint = run / f"model_epoch{epoch:03d}.pt"
            runs[str(seed)] = {"run_dir": run.relative_to(HERE).as_posix(), "checkpoint": checkpoint.relative_to(HERE).as_posix(), "checkpoint_sha256": sha256(checkpoint)}
        receipt["architectures"][model] = {
            "adaptive_receipt": f"adaptive_receipt_{model}.json",
            "selected_epoch": epoch,
            "stopping_epoch": int(adaptive["stopping_epoch"]),
            "selected_optimizer_updates": epoch * 64,
            "stopping_optimizer_updates": int(adaptive["stopping_epoch"]) * 64,
            "selected_median_development_kge": adaptive["selected_median_development_kge"],
            "learning_rate_schedule": adaptive["learning_rate_schedule"],
            "halvings": adaptive["halvings"],
            "plateau_reached": adaptive["plateau_reached"],
            "runs": runs,
        }
    # MC-LSTM is descriptive only and stays at its epoch-30 checkpoint; its entry
    # is carried over unchanged from the released selection receipt.
    path = HERE / "provenance/development_epoch_selection.json"
    previous = json.loads(path.read_text())
    receipt["architectures"]["mclstm"] = previous["architectures"]["mclstm"]
    receipt["all_three_seeds_complete"] = True
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({m: (v["selected_epoch"], v.get("plateau_reached")) for m, v in receipt["architectures"].items() if m != "mclstm"}, indent=2))


if __name__ == "__main__":
    main()
