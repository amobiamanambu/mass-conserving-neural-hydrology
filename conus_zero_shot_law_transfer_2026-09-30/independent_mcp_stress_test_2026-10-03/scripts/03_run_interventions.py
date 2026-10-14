#!/usr/bin/env python3
"""Run the paper's zero-input intervention on frozen MCP water states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mcp_model import RegionalMCP


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
PRIMARY_Q0 = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="For example, mcp_qet_seed11")
    parser.add_argument(
        "--anchor-design", choices=["model_specific", "common_date"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_basin(q: np.ndarray, storage: np.ndarray) -> dict[str, float | int]:
    q0 = q[:, 0]
    valid = np.isfinite(q0) & (q0 >= PRIMARY_Q0)

    def retention(day: int) -> np.ndarray:
        value = np.full(len(q), np.nan, dtype=np.float64)
        value[valid] = q[valid, day] / q0[valid]
        return value

    r7, r30, r90 = retention(7), retention(30), retention(90)
    half_life = np.full(len(q), np.nan, dtype=np.float64)
    monotonicity = np.full(len(q), np.nan, dtype=np.float64)
    for anchor in np.flatnonzero(valid):
        crossing = np.flatnonzero(q[anchor] <= 0.5 * q0[anchor])
        half_life[anchor] = float(crossing[0]) if len(crossing) else 91.0
        tolerance = max(1e-8, 1e-5 * q0[anchor])
        monotonicity[anchor] = float(np.mean(np.diff(q[anchor]) <= tolerance))

    positive = np.isfinite(q) & np.isfinite(storage) & (q >= 1e-5) & (storage > 1e-5)
    exponent = intercept = r2 = np.nan
    if positive.sum() >= 60:
        x = np.log(storage[positive])
        y = np.log(q[positive])
        design = np.column_stack([np.ones(len(x)), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficients
        denominator = np.sum((y - y.mean()) ** 2)
        intercept, exponent = map(float, coefficients)
        r2 = (
            float(1.0 - np.sum((y - fitted) ** 2) / denominator)
            if denominator > 0
            else np.nan
        )

    def median(value: np.ndarray) -> float:
        return float(np.nanmedian(value)) if np.isfinite(value).any() else np.nan

    return {
        "valid_anchors": int(valid.sum()),
        "median_q0_mm_day": median(np.where(valid, q0, np.nan)),
        "median_r7": median(r7),
        "median_r30": median(r30),
        "median_r90": median(r90),
        "median_half_life_days": median(half_life),
        "median_monotonic_fraction": median(monotonicity),
        "storage_discharge_log_intercept": intercept,
        "storage_discharge_exponent": exponent,
        "storage_discharge_r2": r2,
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    run_dir = EXPERIMENT / "runs/frozen_refit" / args.run
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    state_key = (
        "model_specific_states"
        if args.anchor_design == "model_specific"
        else "common_date_states"
    )

    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    static = np.load(PROJECT / "data/processed/static.npy", mmap_mode="r")
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    anchor = np.load(EXPERIMENT / receipt[state_key])
    if not np.array_equal(anchor["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("MCP antecedent-state station order mismatch")

    checkpoint = torch.load(
        EXPERIMENT / receipt["checkpoint"], map_location="cpu", weights_only=False
    )
    model = RegionalMCP(n_static=static.shape[1], n_cells=int(checkpoint["n_cells"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_gauges, n_anchors, state_size = anchor["state"].shape
    n_samples = n_gauges * n_anchors
    state_flat = anchor["state"].reshape(n_samples, state_size)
    time_flat = anchor["eval_time_index"].reshape(n_samples)
    basin_flat = np.repeat(np.arange(n_gauges), n_anchors)
    valid_flat = (time_flat >= 0) & np.isfinite(state_flat).all(axis=1)
    valid_positions = np.flatnonzero(valid_flat)
    q_curve = np.full((n_samples, 91), np.nan, dtype=np.float32)
    storage_curve = np.full((n_samples, 91), np.nan, dtype=np.float32)
    residual_max = 0.0

    with torch.no_grad():
        for first in range(0, len(valid_positions), args.batch_size):
            stop = min(first + args.batch_size, len(valid_positions))
            position = valid_positions[first:stop]
            base_date = eval_dates[time_flat[position]].astype("datetime64[D]")
            experiment_date = base_date[:, None] + np.arange(1, 92).astype(
                "timedelta64[D]"
            )[None, :]
            day_of_year = pd.DatetimeIndex(
                experiment_date.reshape(-1).astype("datetime64[ns]")
            ).dayofyear.to_numpy().reshape(len(position), 91)
            forcing = np.zeros((len(position), 91, 5), dtype=np.float32)
            forcing[:, :, 2] = 10.0
            forcing[:, :, 3] = np.sin(2 * np.pi * day_of_year / 365.25)
            forcing[:, :, 4] = np.cos(2 * np.pi * day_of_year / 365.25)
            rows = target.row.to_numpy(int)[basin_flat[position]]
            result = model(
                torch.from_numpy(forcing),
                torch.from_numpy(np.asarray(static[rows])).float(),
                initial_state=torch.from_numpy(state_flat[position]).float(),
                return_states=True,
            )
            q_curve[position] = result["q"].cpu().numpy()
            storage_curve[position] = result["storage"].cpu().numpy()
            residual_max = max(
                residual_max, float(result["daily_residual"].abs().max())
            )
            print(
                f"{args.run}/{args.anchor_design}: interventions "
                f"{stop:,}/{len(valid_positions):,}",
                flush=True,
            )

    q_curve = q_curve.reshape(n_gauges, n_anchors, 91)
    storage_curve = storage_curve.reshape(n_gauges, n_anchors, 91)
    records = []
    for basin in range(n_gauges):
        records.append(
            {
                "GAGE_ID": target.GAGE_ID.iloc[basin],
                "architecture": "mcp",
                "rung": receipt["rung"],
                "seed": int(receipt["seed"]),
                "anchor_design": args.anchor_design,
                **summarize_basin(q_curve[basin], storage_curve[basin]),
            }
        )

    output_dir = EXPERIMENT / "outputs/interventions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.run}_{args.anchor_design}"
    curve_path = output_dir / f"{prefix}_curves.npz"
    table_path = output_dir / f"{prefix}_fingerprints.parquet"
    np.savez_compressed(
        curve_path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        quantiles=anchor["quantiles"],
        antecedent_time_index=anchor["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    pd.DataFrame(records).to_parquet(table_path, index=False)
    result_receipt = {
        "run": args.run,
        "anchor_design": args.anchor_design,
        "primary_q0_threshold_mm_day": PRIMARY_Q0,
        "gauges": n_gauges,
        "gauges_with_complete_anchors": int(
            np.all(np.isfinite(q_curve[:, :, 0]), axis=1).sum()
        ),
        "anchors_per_gauge": n_anchors,
        "days_per_intervention": 91,
        "maximum_daily_mass_residual_mm": residual_max,
        "curves": str(curve_path.relative_to(EXPERIMENT)),
        "curves_sha256": sha256(curve_path),
        "fingerprints": str(table_path.relative_to(EXPERIMENT)),
        "fingerprints_sha256": sha256(table_path),
    }
    receipt_out = output_dir / f"{prefix}_receipt.json"
    receipt_out.write_text(json.dumps(result_receipt, indent=2) + "\n")
    print(json.dumps(result_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
