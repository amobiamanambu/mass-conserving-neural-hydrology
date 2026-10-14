#!/usr/bin/env python3
"""Frozen donor refit and outcome-blind target prediction/state extraction.

This script never opens target discharge. It fits one selected model on the
500 donors through 2018, saves 2019-2024 predictions for all 3,982 targets,
and, for constrained models, saves warm-season snow-free antecedent states for
the prespecified zero-input interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model, parameter_count  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
QUANTILES = np.linspace(0.05, 0.95, 12, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["ealstm", "dhbv", "fluxgate"], required=True)
    parser.add_argument("--rung", choices=["q", "qet"], required=True)
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_moments(values: torch.Tensor, mask: torch.Tensor):
    count = mask.sum(dim=1).clamp(min=2)
    mean = (values * mask).sum(dim=1) / count
    centered = (values - mean[:, None]) * mask
    variance = centered.square().sum(dim=1) / (count - 1)
    return mean, torch.sqrt(variance + 1e-6), count


def discharge_loss(prediction: torch.Tensor, observation: torch.Tensor):
    mask = torch.isfinite(observation).to(prediction.dtype)
    observed = torch.nan_to_num(observation, nan=0.0)
    pred_mean, pred_std, count = masked_moments(prediction, mask)
    obs_mean, obs_std, _ = masked_moments(observed, mask)

    def normalized_error(predicted_transform, observed_transform):
        observed_mean = (observed_transform * mask).sum(dim=1) / count
        error = ((predicted_transform - observed_transform).square() * mask).sum(dim=1)
        variance = (((observed_transform - observed_mean[:, None]) * mask).square()).sum(dim=1)
        return error / (variance + 0.05 * count)

    raw_error = normalized_error(prediction, observed)
    sqrt_error = normalized_error(torch.sqrt(prediction + 0.01), torch.sqrt(observed + 0.01))
    log_error = normalized_error(torch.log1p(prediction), torch.log1p(observed))
    bias = ((pred_mean - obs_mean) / (obs_mean + 0.10)).square()
    spread = torch.log((pred_std + 0.05) / (obs_std + 0.05)).square()
    return (0.35 * raw_error + 0.30 * sqrt_error + 0.10 * log_error + 0.10 * bias + 0.15 * spread).mean()


def load_selection(architecture: str, rung: str) -> dict:
    selection = json.loads((PROJECT / "config/frozen_model_selection.json").read_text())
    chosen = [m for m in selection["models"] if m["architecture"] == architecture and m["rung"] == rung]
    if len(chosen) != 1:
        raise RuntimeError(f"No unique frozen setting for {architecture}/{rung}")
    if architecture == "ealstm" and rung != "q":
        raise ValueError("The predictive LSTM benchmark is Q-only")
    return chosen[0]


def load_et(stations: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    table = pd.read_parquet(PROJECT / "data/processed/donor_ssebop_et_2003_2018.parquet")
    table.GAGE_ID = table.GAGE_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    years = sorted(table.year.unique())
    donor_ids = stations.loc[stations.role.eq("donor"), "GAGE_ID"].tolist()
    pivot = table.pivot(index="GAGE_ID", columns="year", values="ssebop_apr_oct_et_mm").reindex(index=donor_ids, columns=years)
    if pivot.isna().any().any():
        raise RuntimeError("Missing donor SSEBop values")
    return pivot.to_numpy(np.float32), {int(year): position for position, year in enumerate(years)}


def select_anchors(
    state: np.ndarray,
    forcing: np.ndarray,
    eval_dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select nearest observed states to 12 storage quantiles per basin."""
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
            raise RuntimeError(f"Insufficient antecedent states for local basin {basin}")
        target_storage = np.quantile(storage[candidates], QUANTILES)
        used: set[int] = set()
        selected = []
        for target in target_storage:
            order = candidates[np.argsort(np.abs(storage[candidates] - target), kind="stable")]
            pick = next((int(value) for value in order if int(value) not in used), int(order[0]))
            used.add(pick)
            selected.append(pick)
        selected_array = np.asarray(selected, dtype=np.int16)
        chosen_states[basin] = state[basin, selected_array]
        chosen_time[basin] = selected_array
    return chosen_states, chosen_time, relaxation


def mass_audit(model: torch.nn.Module, forcing: np.ndarray, static: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    if not getattr(model, "conserves_mass", False):
        return {"daily_max_abs_mm": np.nan, "cumulative_max_abs_mm": np.nan}
    original_dtype = next(model.parameters()).dtype
    model = model.double().eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(np.asarray(forcing[rows, :730])).double(),
            torch.from_numpy(np.asarray(static[rows])).double(),
            return_states=True,
        )
    result = {
        "daily_max_abs_mm": float(output["daily_residual"].abs().max()),
        "cumulative_max_abs_mm": float(output["cumulative_residual"].abs().max()),
    }
    model.to(dtype=original_dtype)
    return result


def main() -> None:
    args = parse_args()
    frozen = load_selection(args.architecture, args.rung)
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    processed = PROJECT / "data/processed"
    stations = pd.read_csv(processed / "station_index.csv", dtype={"GAGE_ID": str})
    dates = np.load(processed / "dates.npy", mmap_mode="r")
    forcing = np.load(processed / "forcing.npy", mmap_mode="r")
    static = np.load(processed / "static.npy", mmap_mode="r")
    q = np.load(processed / "q_donor_development.npy", mmap_mode="r")
    donor = stations[stations.role.eq("donor")]
    target = stations[stations.role.eq("target")]
    donor_rows = donor.row.to_numpy(int)
    donor_q_rows = donor.q_row.to_numpy(int)
    target_rows = target.row.to_numpy(int)
    et, year_to_et = load_et(stations)

    defaults = {
        "ealstm": {"learning_rate": 0.0010, "weight_decay": 1e-5},
        "dhbv": {"learning_rate": 0.0020, "weight_decay": 1e-5},
        "fluxgate": {"learning_rate": 0.0015, "weight_decay": 1e-5},
    }[args.architecture]
    epochs = int(frozen["epochs"])
    model = make_model(args.architecture, static.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=defaults["learning_rate"], weight_decay=defaults["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=defaults["learning_rate"] / 20.0
    )

    run_name = f"{args.architecture}_{args.rung}_seed{args.seed}"
    run_dir = PROJECT / "runs/frozen_refit" / run_name
    prediction_dir = PROJECT / "outputs/predictions"
    anchor_dir = PROJECT / "outputs/antecedent_states"
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir.mkdir(parents=True, exist_ok=True)

    history = []
    training_years = np.arange(2005, 2019)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.permutation(len(donor_rows))
        losses, q_losses, et_losses = [], [], []
        for first in range(0, len(shuffled), args.batch_size):
            local = shuffled[first:first + args.batch_size]
            rows = donor_rows[local]
            q_rows = donor_q_rows[local]
            year = int(np.random.choice(training_years))
            start = np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01"))
            stop = np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right")
            target_start = np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01"))
            x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            observed = torch.from_numpy(np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])).float()
            result = model(x, s)
            q_loss = discharge_loss(result["q"][:, target_start:], observed)
            et_loss = torch.zeros((), dtype=q_loss.dtype)
            if args.rung == "qet":
                season = (dates[start:stop] >= np.datetime64(f"{year}-04-01")) & (dates[start:stop] <= np.datetime64(f"{year}-10-31"))
                predicted_et = result["et"][:, torch.from_numpy(season)].sum(dim=1)
                target_et = torch.from_numpy(et[local, year_to_et[year]]).float()
                scaled = (predicted_et - target_et) / (0.20 * target_et + 50.0)
                et_loss = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled))
            loss = q_loss + (0.20 * et_loss if args.rung == "qet" else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach())); q_losses.append(float(q_loss.detach())); et_losses.append(float(et_loss.detach()))
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

    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    checkpoint_path = run_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(), "architecture": args.architecture,
        "rung": args.rung, "seed": args.seed, "n_static": int(static.shape[1]),
        "epochs": epochs, "refit_end": "2018-12-31",
    }, checkpoint_path)
    audit = mass_audit(model, forcing, static, donor_rows[:4])

    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    prediction_path = prediction_dir / f"{run_name}_target_q_2019_2024.npy"
    prediction = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(target_rows), len(eval_dates))
    )
    anchor_path = None
    anchor_relaxation_counts = None
    if getattr(model, "conserves_mass", False):
        state_size = int(model.state_size)
        all_anchor_state = np.empty((len(target_rows), len(QUANTILES), state_size), dtype=np.float32)
        all_anchor_time = np.empty((len(target_rows), len(QUANTILES)), dtype=np.int16)
        all_relaxation = np.empty(len(target_rows), dtype=np.int8)

    model.eval()
    with torch.no_grad():
        for first in range(0, len(target_rows), args.batch_size):
            stop = min(first + args.batch_size, len(target_rows))
            rows = target_rows[first:stop]
            x_np = np.asarray(forcing[rows, :eval_end])
            result = model(
                torch.from_numpy(x_np).float(),
                torch.from_numpy(np.asarray(static[rows])).float(),
                return_states=getattr(model, "conserves_mass", False),
            )
            prediction[first:stop] = result["q"][:, eval_start:eval_end].cpu().numpy()
            if getattr(model, "conserves_mass", False):
                state = result["state"][:, eval_start:eval_end].cpu().numpy()
                selected_state, selected_time, relaxation = select_anchors(
                    state, x_np[:, eval_start:eval_end], eval_dates
                )
                all_anchor_state[first:stop] = selected_state
                all_anchor_time[first:stop] = selected_time
                all_relaxation[first:stop] = relaxation
            print(f"{run_name}: predicted targets {stop:,}/{len(target_rows):,}", flush=True)
    prediction.flush()

    if getattr(model, "conserves_mass", False):
        anchor_path = anchor_dir / f"{run_name}_antecedent_states.npz"
        np.savez_compressed(
            anchor_path,
            gage_ids=target.GAGE_ID.to_numpy(str),
            quantiles=QUANTILES.astype(np.float32),
            state=all_anchor_state,
            eval_time_index=all_anchor_time,
            relaxation_code=all_relaxation,
        )
        unique, counts = np.unique(all_relaxation, return_counts=True)
        anchor_relaxation_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}

    receipt = {
        "run": run_name,
        "architecture": args.architecture,
        "rung": args.rung,
        "seed": args.seed,
        "epochs": epochs,
        "parameter_count": parameter_count(model),
        "donor_gauges": int(len(donor_rows)),
        "target_gauges": int(len(target_rows)),
        "target_period": ["2019-01-01", "2024-12-31"],
        "target_discharge_accessed": False,
        "mass_balance": audit,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction": str(prediction_path.relative_to(PROJECT)),
        "prediction_sha256": sha256(prediction_path),
        "antecedent_states": str(anchor_path.relative_to(PROJECT)) if anchor_path else None,
        "antecedent_states_sha256": sha256(anchor_path) if anchor_path else None,
        "anchor_relaxation_counts": anchor_relaxation_counts,
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=True) + "\n")
    print(json.dumps(receipt, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
