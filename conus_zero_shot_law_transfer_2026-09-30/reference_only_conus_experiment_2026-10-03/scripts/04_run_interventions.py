#!/usr/bin/env python3
"""Run identical zero-input drainage interventions on frozen reference models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
MODELS = ("dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr")
Q0_THRESHOLDS = (0.0001, 0.001, 0.01)


sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "multimodel_neural_conservation_stress_test_2026-10-03/scripts"))
import hydro_models  # noqa: E402
import conserved_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], required=True)
    parser.add_argument("--anchor-design", choices=["common_date", "model_specific"], required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def make_model(name: str, n_static: int):
    if name in {"dhbv", "fluxgate"}:
        return hydro_models.make_model(name, n_static)
    return conserved_models.make_model(name, n_static)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_median(values: np.ndarray) -> float:
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def summarize(q: np.ndarray, storage: np.ndarray) -> dict[str, float | int]:
    q0 = q[:, 0]
    result: dict[str, float | int] = {}
    for threshold in Q0_THRESHOLDS:
        valid = np.isfinite(q0) & (q0 >= threshold)
        suffix = str(threshold).replace("0.", "")
        result[f"valid_anchors_q0_{suffix}"] = int(valid.sum())
        for day in (7, 30, 90):
            retention = np.full(len(q), np.nan)
            retention[valid] = q[valid, day] / q0[valid]
            result[f"median_r{day}_q0_{suffix}"] = safe_median(retention)
    primary = np.isfinite(q0) & (q0 >= 0.001)
    half_life = np.full(len(q), np.nan)
    for index in np.flatnonzero(primary):
        crossing = np.flatnonzero(q[index] <= 0.5 * q0[index])
        half_life[index] = float(crossing[0]) if len(crossing) else 91.0
    result["median_q0_mm_day"] = safe_median(np.where(primary, q0, np.nan))
    result["median_half_life_days"] = safe_median(half_life)
    result["median_release_7d_mm"] = safe_median(np.where(primary, q[:, :8].sum(axis=1), np.nan))
    result["median_release_30d_mm"] = safe_median(np.where(primary, q[:, :31].sum(axis=1), np.nan))
    result["median_initial_storage_mm"] = safe_median(np.where(primary, storage[:, 0], np.nan))
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    run_name = f"{args.model}_qet_seed{args.seed}"
    run_dir = EXPERIMENT / f"runs/frozen_refit/{run_name}"
    receipt = json.loads((run_dir / "receipt.json").read_text())
    if receipt.get("target_discharge_accessed") is not False:
        raise RuntimeError("Outcome-blind model receipt is not intact")

    stations = pd.read_csv(EXPERIMENT / "data/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    target_rows = target.row.to_numpy(int)
    static = np.load(EXPERIMENT / "data/static.npy", mmap_mode="r")
    dates = np.load(EXPERIMENT / "data/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    state_key = "common_date_states" if args.anchor_design == "common_date" else "model_specific_states"
    anchor = np.load(EXPERIMENT / receipt[state_key])
    if not np.array_equal(anchor["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Target order mismatch")

    checkpoint = torch.load(EXPERIMENT / receipt["checkpoint"], map_location="cpu", weights_only=False)
    model = make_model(args.model, static.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    model = model.double().eval()
    n_gauges, n_anchors, state_size = anchor["state"].shape
    n_samples = n_gauges * n_anchors
    state_flat = anchor["state"].reshape(n_samples, state_size)
    time_flat = anchor["eval_time_index"].reshape(n_samples)
    basin_flat = np.repeat(np.arange(n_gauges), n_anchors)
    valid = (time_flat >= 0) & np.isfinite(state_flat).all(axis=1)
    positions = np.flatnonzero(valid)
    q_curve = np.full((n_samples, 91), np.nan, dtype=np.float32)
    storage_curve = np.full_like(q_curve, np.nan)
    residual_max = 0.0
    with torch.no_grad():
        for first in range(0, len(positions), args.batch_size):
            stop = min(first + args.batch_size, len(positions))
            position = positions[first:stop]
            base_date = eval_dates[time_flat[position]].astype("datetime64[D]")
            experiment_date = base_date[:, None] + np.arange(1, 92).astype("timedelta64[D]")[None]
            day_of_year = pd.DatetimeIndex(
                experiment_date.reshape(-1).astype("datetime64[ns]")
            ).dayofyear.to_numpy().reshape(len(position), 91)
            forcing = np.zeros((len(position), 91, 5), dtype=np.float64)
            forcing[:, :, 2] = 10.0
            forcing[:, :, 3] = np.sin(2 * np.pi * day_of_year / 365.25)
            forcing[:, :, 4] = np.cos(2 * np.pi * day_of_year / 365.25)
            rows = target_rows[basin_flat[position]]
            output = model(
                torch.from_numpy(forcing),
                torch.from_numpy(np.asarray(static[rows])).double(),
                initial_state=torch.from_numpy(state_flat[position]).double(),
                return_states=True,
            )
            q_curve[position] = output["q"].cpu().numpy()
            storage_curve[position] = output["storage"].cpu().numpy()
            residual_max = max(residual_max, float(output["daily_residual"].abs().max()))
            print(f"{run_name}/{args.anchor_design}: {stop}/{len(positions)} interventions", flush=True)

    q_curve = q_curve.reshape(n_gauges, n_anchors, 91)
    storage_curve = storage_curve.reshape(n_gauges, n_anchors, 91)
    records = []
    for basin in range(n_gauges):
        records.append(
            {
                "GAGE_ID": target.GAGE_ID.iloc[basin],
                "model": args.model,
                "seed": args.seed,
                "anchor_design": args.anchor_design,
                **summarize(q_curve[basin], storage_curve[basin]),
            }
        )
    output_dir = EXPERIMENT / "outputs/interventions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_name}_{args.anchor_design}"
    table_path = output_dir / f"{prefix}_fingerprints.csv.gz"
    curve_path = output_dir / f"{prefix}_curves.npz"
    pd.DataFrame(records).to_csv(table_path, index=False, compression="gzip")
    np.savez_compressed(
        curve_path, gage_ids=target.GAGE_ID.to_numpy(str), quantiles=anchor["quantiles"],
        antecedent_time_index=anchor["eval_time_index"], q_mm_day=q_curve, storage_mm=storage_curve,
    )
    result_receipt = {
        "run": run_name, "model": args.model, "seed": args.seed,
        "anchor_design": args.anchor_design, "gauges": n_gauges,
        "anchors_per_gauge": n_anchors, "intervention_days": 91,
        "forcing": "P=0, PET=0, T=10 C with calendar harmonics retained",
        "q0_thresholds_mm_day": list(Q0_THRESHOLDS),
        "maximum_daily_mass_residual_mm": residual_max,
        "target_discharge_accessed": False,
        "fingerprints": str(table_path.relative_to(EXPERIMENT)),
        "fingerprints_sha256": sha256(table_path),
        "curves": str(curve_path.relative_to(EXPERIMENT)),
        "curves_sha256": sha256(curve_path),
    }
    (output_dir / f"{prefix}_receipt.json").write_text(json.dumps(result_receipt, indent=2) + "\n")
    print(json.dumps(result_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
