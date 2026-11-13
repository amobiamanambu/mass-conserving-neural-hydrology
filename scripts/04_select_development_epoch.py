#!/usr/bin/env python3
"""Select one checkpoint per architecture from seed-11 development KGE only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
EPOCHS = (10, 20, 30)
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_run(model: str, seed: int, required: bool = True) -> Path | None:
    prefix = f"national_literature_{model}_seed{seed}_"
    candidates: list[Path] = []
    for path in sorted((ROOT / "runs").glob(prefix + "*")):
        config_path = path / "config.yml"
        checkpoint = path / "model_epoch030.pt"
        if not config_path.exists() or not checkpoint.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if (
            int(config.get("seed", -1)) == seed
            and int(config.get("epochs", -1)) == 30
            and str(config.get("experiment_name", ""))
            == f"national_literature_{model}_seed{seed}"
        ):
            candidates.append(path)
    if len(candidates) > 1 or (required and len(candidates) != 1):
        raise RuntimeError(
            f"Expected {'exactly one' if required else 'at most one'} completed run "
            f"for {model} seed {seed}; found {len(candidates)}: {candidates}"
        )
    return candidates[0] if candidates else None


def main() -> None:
    receipt: dict[str, object] = {
        "selection_data": "200 development gauges, 2015-2018",
        "target_population": "3,982 unseen gauges, 2019-2024",
        "target_discharge_accessed": False,
        "selection_seed": 11,
        "candidate_epochs": list(EPOCHS),
        "selection_rule": "largest median development KGE; later epoch breaks an exact tie",
        "architectures": {},
    }
    all_seeds_complete = True
    for model in MODELS:
        seed11 = completed_run(model, 11)
        assert seed11 is not None
        candidates = []
        for epoch in EPOCHS:
            metrics_path = (
                seed11 / "validation" / f"model_epoch{epoch:03d}"
                / "validation_metrics.csv"
            )
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            metrics = pd.read_csv(metrics_path, dtype={"basin": str})
            if len(metrics) != 200:
                raise RuntimeError(
                    f"{model} epoch {epoch} has {len(metrics)} development gauges; expected 200"
                )
            candidates.append({
                "epoch": epoch,
                "median_kge": float(np.nanmedian(metrics["KGE"])),
                "mean_kge": float(np.nanmean(metrics["KGE"])),
                "fraction_kge_ge_0_30": float(np.nanmean(metrics["KGE"] >= 0.30)),
                "metrics": str(metrics_path.relative_to(ROOT)),
                "metrics_sha256": sha256(metrics_path),
            })
        selected = max(candidates, key=lambda item: (item["median_kge"], item["epoch"]))
        runs = {}
        for seed in SEEDS:
            run = completed_run(model, seed, required=(seed == 11))
            if run is None:
                all_seeds_complete = False
                continue
            checkpoint = run / f"model_epoch{selected['epoch']:03d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            runs[str(seed)] = {
                "run_dir": str(run.relative_to(ROOT)),
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
            }
        receipt["architectures"][model] = {
            "candidates": candidates,
            "selected_epoch": selected["epoch"],
            "runs": runs,
        }
    receipt["all_three_seeds_complete"] = all_seeds_complete
    filename = (
        "development_epoch_selection.json"
        if all_seeds_complete
        else "development_epoch_selection_seed11.json"
    )
    target = ROOT / "provenance" / filename
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
