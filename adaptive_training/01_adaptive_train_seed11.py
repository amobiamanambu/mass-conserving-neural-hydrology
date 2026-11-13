#!/usr/bin/env python3
"""Seed-11 adaptive learning-rate training (see PROTOCOL.md).

Resumable: every decision is rebuilt from the saved development-validation
files, so a stopped job can simply be started again.
Usage: python 01_adaptive_train_seed11.py --model hybrid_shm
"""

from __future__ import annotations

import argparse
import json
import tempfile
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ADAPTER = PROJECT / "vendor" / "neuralhydrology_literature_adapter"
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
START_EPOCH = 30
BLOCK = 10
MATERIAL = 0.005
PATIENCE_HALVE = 3
MAX_HALVINGS = 3
PATIENCE_STOP = 5
MAX_EPOCH = 400


def base_run(model: str, seed: int = 11) -> Path:
    return HERE / "runs_adaptive" / model / f"seed{seed}" / "base_run"


def peak_rate(model: str) -> float:
    config = yaml.safe_load((base_run(model) / "config.yml").read_text())
    return float({int(k): v for k, v in config["learning_rate"].items()}[0])


def checkpoint_epochs(run: Path) -> list[int]:
    return sorted(int(p.stem[-3:]) for p in run.glob("model_epoch*.pt"))


def latest_leaf(base: Path) -> Path:
    candidates = [base] + [p for p in base.glob("**/continue_training_from_epoch*") if p.is_dir()]
    usable = [(max(checkpoint_epochs(p), default=-1), len(p.parts), p) for p in candidates]
    return max(usable)[2]


def development_history(base: Path) -> pd.DataFrame:
    rows = []
    for path in base.glob("**/validation/model_epoch*/validation_metrics.csv"):
        metrics = pd.read_csv(path, dtype={"basin": "string"})
        if len(metrics) != 200:
            raise RuntimeError(f"{path} has {len(metrics)} rows, expected 200")
        epoch = int(path.parent.name[-3:])
        rows.append({"epoch": epoch, "updates": epoch * 64, "median_kge": float(np.nanmedian(metrics["KGE"])),
                     "mean_kge": float(np.nanmean(metrics["KGE"])),
                     "fraction_kge_ge_0_30": float(np.nanmean(metrics["KGE"] >= 0.30)),
                     "metrics_file": str(path.relative_to(HERE))})
    if not rows:
        raise RuntimeError(f"No validation history under {base}")
    return pd.DataFrame(rows).sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)


def replay(history: pd.DataFrame, peak: float) -> dict:
    """Apply the frozen rule to the validation history in epoch order."""
    rows = history[history.epoch >= START_EPOCH].sort_values("epoch")
    if rows.empty or int(rows.iloc[0].epoch) != START_EPOCH:
        raise RuntimeError("Epoch-30 development validation is missing")
    best = float(rows.iloc[0].median_kge)
    since, halvings, rate = 0, 0, peak
    schedule = {START_EPOCH + 1: peak}
    status, stop_epoch = "running", int(rows.iloc[-1].epoch)
    learning_rate_by_epoch = {START_EPOCH: None}
    for row in rows.iloc[1:].itertuples(index=False):
        learning_rate_by_epoch[int(row.epoch)] = rate
        if row.median_kge >= best + MATERIAL:
            best, since = float(row.median_kge), 0
        else:
            since += 1
        if halvings < MAX_HALVINGS and since >= PATIENCE_HALVE:
            rate, halvings, since = rate / 2.0, halvings + 1, 0
            schedule[int(row.epoch) + 1] = rate
        elif halvings >= MAX_HALVINGS and since >= PATIENCE_STOP:
            status, stop_epoch = "converged", int(row.epoch)
            break
        if int(row.epoch) >= MAX_EPOCH:
            status, stop_epoch = "maximum_epoch", int(row.epoch)
            break
    return {"rate": rate, "halvings": halvings, "schedule": schedule, "status": status,
            "stop_epoch": stop_epoch, "rate_by_epoch": learning_rate_by_epoch}


def continue_block(leaf: Path, model: str, rate: float, last_epoch: int) -> None:
    override = HERE / "overrides" / f"{model}_seed11_block_from_{last_epoch:03d}.yml"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(yaml.safe_dump({
        "epochs": BLOCK,
        "learning_rate": {0: rate, last_epoch + 1: rate},
        "validate_every": BLOCK, "validate_n_random_basins": 200,
        "save_weights_every": BLOCK, "save_validation_results": True, "verbose": 0,
    }, sort_keys=False))
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ADAPTER), "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "nh_mpl_cache"), "NGEN_SILENT": "1",
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"), "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "2")})
    subprocess.run([sys.executable, "-m", "neuralhydrology.nh_run", "continue_training", "--run-dir", str(leaf),
                    "--config-file", str(override)], cwd=PROJECT, env=env, check=True, stdin=subprocess.DEVNULL)


def write_receipt(model: str, history: pd.DataFrame, state: dict) -> dict:
    window = history[(history.epoch >= START_EPOCH) & (history.epoch <= state["stop_epoch"])]
    best = window.sort_values(["median_kge", "epoch"]).iloc[-1]
    history = history.copy()
    history["learning_rate"] = history.epoch.map(state["rate_by_epoch"])
    history_path = HERE / f"development_history_{model}_seed11.csv"
    history.to_csv(history_path, index=False)
    receipt = {
        "model": model, "seed": 11, "status": state["status"], "target_discharge_accessed": False,
        "rule": {"block_epochs": BLOCK, "material_improvement": MATERIAL, "halve_after_blocks": PATIENCE_HALVE,
                 "maximum_halvings": MAX_HALVINGS, "stop_after_blocks_at_final_rate": PATIENCE_STOP,
                 "maximum_epoch": MAX_EPOCH},
        "peak_learning_rate": peak_rate(model),
        "learning_rate_schedule": {str(k): v for k, v in state["schedule"].items()},
        "halvings": state["halvings"],
        "latest_epoch": int(history.epoch.max()),
        "stopping_epoch": state["stop_epoch"],
        "plateau_reached": state["status"] == "converged",
        "selected_epoch": int(best.epoch),
        "selected_median_development_kge": float(best.median_kge),
        "epoch30_median_development_kge": float(history.loc[history.epoch == START_EPOCH, "median_kge"].iloc[0]),
        "history_csv": history_path.name,
    }
    (HERE / f"adaptive_receipt_{model}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    base = base_run(args.model)
    if not base.exists():
        raise FileNotFoundError("Run 00_prepare.py first")
    peak = peak_rate(args.model)
    while True:
        history = development_history(base)
        state = replay(history, peak)
        write_receipt(args.model, history, state)
        if state["status"] != "running":
            print(json.dumps(json.loads((HERE / f"adaptive_receipt_{args.model}.json").read_text()), indent=2), flush=True)
            return
        leaf = latest_leaf(base)
        last = max(checkpoint_epochs(leaf))
        if last != int(history.epoch.max()):
            raise RuntimeError(f"Checkpoint epoch {last} and validation epoch {int(history.epoch.max())} disagree in {leaf}")
        print(f"{args.model}: epoch {last}, learning rate {state['rate']:.6g}, halvings {state['halvings']}", flush=True)
        continue_block(leaf, args.model, state["rate"], last)


if __name__ == "__main__":
    main()
