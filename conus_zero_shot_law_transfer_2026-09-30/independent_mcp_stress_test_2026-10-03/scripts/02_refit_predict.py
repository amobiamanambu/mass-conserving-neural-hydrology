#!/usr/bin/env python3
"""Refit MCP on 500 donors and predict the original 3,982 target gauges."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mcp_model import RegionalMCP, parameter_count
from importlib.machinery import SourceFileLoader


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
COMMON_ANCHOR_PATH = (
    PROJECT / "outputs/identification/strengthened_test/common_anchors.npz"
)
QUANTILES = np.linspace(0.05, 0.95, 12, dtype=np.float64)

# Reuse the exact archived discharge loss rather than maintaining a divergent copy.
selection_path = Path(__file__).resolve().parent / "01_select_on_development.py"
selection_module = SourceFileLoader(
    "mcp_development_selection",
    str(selection_path),
).load_module()
discharge_loss = selection_module.discharge_loss
load_et = selection_module.load_et


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", choices=["q", "qet"], required=True)
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_storage_anchors(
    state: np.ndarray, forcing: np.ndarray, eval_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _, state_size = state.shape
    chosen_states = np.empty((n, len(QUANTILES), state_size), dtype=np.float32)
    chosen_time = np.empty((n, len(QUANTILES)), dtype=np.int16)
    relaxation = np.zeros(n, dtype=np.int8)
    months = pd.DatetimeIndex(eval_dates.astype("datetime64[ns]")).month.to_numpy()
    warm_season = (months >= 6) & (months <= 9)
    for basin in range(n):
        storage = state[basin].sum(axis=1)
        snow_free = state[basin, :, 0] < 1.0
        warm = forcing[basin, :, 2] > 5.0
        candidates = np.flatnonzero(warm_season & snow_free & warm & np.isfinite(storage))
        if len(candidates) < len(QUANTILES):
            candidates = np.flatnonzero(warm_season & snow_free & np.isfinite(storage))
            relaxation[basin] = 1
        if len(candidates) < len(QUANTILES):
            candidates = np.flatnonzero(warm_season & np.isfinite(storage))
            relaxation[basin] = 2
        if len(candidates) < len(QUANTILES):
            raise RuntimeError(f"Insufficient MCP antecedent states for local basin {basin}")
        targets = np.quantile(storage[candidates], QUANTILES)
        used: set[int] = set()
        selected: list[int] = []
        for target in targets:
            order = candidates[np.argsort(np.abs(storage[candidates] - target), kind="stable")]
            pick = next((int(value) for value in order if int(value) not in used), int(order[0]))
            used.add(pick)
            selected.append(pick)
        selected_array = np.asarray(selected, dtype=np.int16)
        chosen_states[basin] = state[basin, selected_array]
        chosen_time[basin] = selected_array
    return chosen_states, chosen_time, relaxation


def capture_common_states(
    state: np.ndarray, common_time: np.ndarray
) -> np.ndarray:
    captured = np.full(
        (state.shape[0], common_time.shape[1], state.shape[2]), np.nan, dtype=np.float32
    )
    for basin in range(state.shape[0]):
        valid = common_time[basin] >= 0
        captured[basin, valid] = state[basin, common_time[basin, valid]]
    return captured


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    development_summary_path = (
        EXPERIMENT / f"runs/development/mcp_{args.rung}_seed11/summary.json"
    )
    if not development_summary_path.exists():
        raise FileNotFoundError(development_summary_path)
    development_summary = json.loads(development_summary_path.read_text())
    epochs = int(development_summary["best_epoch"])

    processed = PROJECT / "data/processed"
    stations = pd.read_csv(processed / "station_index.csv", dtype={"GAGE_ID": str})
    dates = np.load(processed / "dates.npy", mmap_mode="r")
    forcing = np.load(processed / "forcing.npy", mmap_mode="r")
    static = np.load(processed / "static.npy", mmap_mode="r")
    q = np.load(processed / "q_donor_development.npy", mmap_mode="r")
    et, year_to_et = load_et(stations)
    donor = stations.loc[stations.role.eq("donor")]
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    donor_rows = donor.row.to_numpy(int)
    donor_q_rows = donor.q_row.to_numpy(int)
    target_rows = target.row.to_numpy(int)

    common_anchor = np.load(COMMON_ANCHOR_PATH)
    if not np.array_equal(common_anchor["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Common-anchor station order does not match the target population")
    common_time = common_anchor["eval_time_index"].astype(np.int16)

    model = RegionalMCP(n_static=static.shape[1])
    learning_rate = 0.0015
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 20.0
    )
    run_name = f"mcp_{args.rung}_seed{args.seed}"
    run_dir = EXPERIMENT / "runs/frozen_refit" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "outputs/predictions").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "outputs/antecedent_states").mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    training_years = np.arange(2005, 2019)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.permutation(len(donor_rows))
        losses: list[float] = []
        q_losses: list[float] = []
        et_losses: list[float] = []
        for first in range(0, len(shuffled), args.batch_size):
            local = shuffled[first : first + args.batch_size]
            rows = donor_rows[local]
            q_rows = donor_q_rows[local]
            year = int(np.random.choice(training_years))
            start = int(np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01")))
            stop = int(np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right"))
            target_start = int(np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01")))
            x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            observed = torch.from_numpy(
                np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])
            ).float()
            result = model(x, s)
            q_loss = discharge_loss(result["q"][:, target_start:], observed)
            et_loss = torch.zeros((), dtype=q_loss.dtype)
            if args.rung == "qet":
                season = (dates[start:stop] >= np.datetime64(f"{year}-04-01")) & (
                    dates[start:stop] <= np.datetime64(f"{year}-10-31")
                )
                predicted_et = result["et"][:, torch.from_numpy(season)].sum(dim=1)
                target_et = torch.from_numpy(et[local, year_to_et[year]]).float()
                scaled = (predicted_et - target_et) / (0.20 * target_et + 50.0)
                et_loss = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled))
            loss = q_loss + (0.20 * et_loss if args.rung == "qet" else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            q_losses.append(float(q_loss.detach()))
            et_losses.append(float(et_loss.detach()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "training_q_loss": float(np.mean(q_losses)),
            "training_et_loss": float(np.mean(et_losses)),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(json.dumps({"run": run_name, **record}), flush=True)

    history_path = run_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "mcp",
            "rung": args.rung,
            "seed": args.seed,
            "n_static": int(static.shape[1]),
            "n_cells": model.n_cells,
            "epochs": epochs,
            "refit_end": "2018-12-31",
        },
        checkpoint_path,
    )

    original_dtype = next(model.parameters()).dtype
    model = model.double().eval()
    with torch.no_grad():
        audit = model(
            torch.from_numpy(np.asarray(forcing[donor_rows[:4], :730])).double(),
            torch.from_numpy(np.asarray(static[donor_rows[:4]])).double(),
            return_states=True,
        )
    mass_balance = {
        "daily_max_abs_mm": float(audit["daily_residual"].abs().max()),
        "cumulative_max_abs_mm": float(audit["cumulative_residual"].abs().max()),
    }
    model.to(dtype=original_dtype).eval()

    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    prediction_path = (
        EXPERIMENT / f"outputs/predictions/{run_name}_target_q_2019_2024.npy"
    )
    prediction = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(target_rows), len(eval_dates)),
    )
    storage_state = np.empty(
        (len(target_rows), len(QUANTILES), model.state_size), dtype=np.float32
    )
    storage_time = np.empty((len(target_rows), len(QUANTILES)), dtype=np.int16)
    storage_relaxation = np.empty(len(target_rows), dtype=np.int8)
    common_state = np.full(
        (len(target_rows), len(QUANTILES), model.state_size), np.nan, dtype=np.float32
    )

    with torch.no_grad():
        for first in range(0, len(target_rows), args.batch_size):
            stop = min(first + args.batch_size, len(target_rows))
            rows = target_rows[first:stop]
            x_np = np.asarray(forcing[rows, :eval_end])
            result = model(
                torch.from_numpy(x_np).float(),
                torch.from_numpy(np.asarray(static[rows])).float(),
                return_states=True,
            )
            prediction[first:stop] = result["q"][:, eval_start:eval_end].cpu().numpy()
            state = result["state"][:, eval_start:eval_end].cpu().numpy()
            selected_state, selected_time, relaxation = select_storage_anchors(
                state, x_np[:, eval_start:eval_end], eval_dates
            )
            storage_state[first:stop] = selected_state
            storage_time[first:stop] = selected_time
            storage_relaxation[first:stop] = relaxation
            common_state[first:stop] = capture_common_states(
                state, common_time[first:stop]
            )
            print(f"{run_name}: predicted targets {stop:,}/{len(target_rows):,}", flush=True)
    prediction.flush()

    model_specific_path = (
        EXPERIMENT / f"outputs/antecedent_states/{run_name}_model_specific.npz"
    )
    np.savez_compressed(
        model_specific_path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        quantiles=QUANTILES.astype(np.float32),
        state=storage_state,
        eval_time_index=storage_time,
        relaxation_code=storage_relaxation,
    )
    common_path = EXPERIMENT / f"outputs/antecedent_states/{run_name}_common_date.npz"
    np.savez_compressed(
        common_path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        quantiles=QUANTILES.astype(np.float32),
        state=common_state,
        eval_time_index=common_time,
        relaxation_code=common_anchor["snow_threshold_relaxation_code"],
    )
    unique, counts = np.unique(storage_relaxation, return_counts=True)
    receipt = {
        "run": run_name,
        "architecture": "mcp",
        "rung": args.rung,
        "seed": args.seed,
        "epochs": epochs,
        "parameter_count": parameter_count(model),
        "donor_gauges": int(len(donor_rows)),
        "target_gauges": int(len(target_rows)),
        "target_period": ["2019-01-01", "2024-12-31"],
        "mass_balance": mass_balance,
        "checkpoint": str(checkpoint_path.relative_to(EXPERIMENT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction": str(prediction_path.relative_to(EXPERIMENT)),
        "prediction_sha256": sha256(prediction_path),
        "model_specific_states": str(model_specific_path.relative_to(EXPERIMENT)),
        "model_specific_states_sha256": sha256(model_specific_path),
        "common_date_states": str(common_path.relative_to(EXPERIMENT)),
        "common_date_states_sha256": sha256(common_path),
        "model_specific_relaxation_counts": {
            str(int(code)): int(count) for code, count in zip(unique, counts)
        },
        "target_discharge_accessed_by_script": False,
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
