#!/usr/bin/env python3
"""Run a post-evaluation zero-rain sensitivity with climatological PET.

The primary intervention sets precipitation and PET to zero. This sensitivity
keeps precipitation at zero but supplies each target gauge with its own
day-of-year median GridMET PET from 2004-2018. It uses the archived antecedent
states and frozen model weights and never reads target discharge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_Q0 = 0.001
CLIMATOLOGY_START = np.datetime64("2004-01-01")
CLIMATOLOGY_END = np.datetime64("2018-12-31")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Frozen Q+ET run, e.g. dhbv_qet_seed11")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


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
        r2 = float(1.0 - np.sum((y - fitted) ** 2) / denominator) if denominator > 0 else np.nan

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


def build_pet_climatology(
    forcing: np.ndarray,
    dates: np.ndarray,
    target_rows: np.ndarray,
    output_dir: Path,
    batch_size: int = 256,
) -> tuple[np.ndarray, Path]:
    output_path = output_dir / "target_pet_climatology_2004_2018.npy"
    if output_path.exists():
        climatology = np.load(output_path)
        if climatology.shape != (len(target_rows), 366):
            raise RuntimeError("Existing PET climatology has an unexpected shape")
        return climatology, output_path

    start = int(np.searchsorted(dates, CLIMATOLOGY_START))
    stop = int(np.searchsorted(dates, CLIMATOLOGY_END, side="right"))
    local_dates = np.asarray(dates[start:stop])
    day_of_year = pd.DatetimeIndex(local_dates.astype("datetime64[ns]")).dayofyear.to_numpy()
    climatology = np.full((len(target_rows), 366), np.nan, dtype=np.float32)
    for first in range(0, len(target_rows), batch_size):
        last = min(first + batch_size, len(target_rows))
        rows = target_rows[first:last]
        pet = np.asarray(forcing[rows, start:stop, 1], dtype=np.float32)
        for day in range(1, 367):
            selected = day_of_year == day
            if selected.any():
                climatology[first:last, day - 1] = np.nanmedian(pet[:, selected], axis=1)
        print(f"PET climatology {last:,}/{len(target_rows):,}", flush=True)
    if not np.isfinite(climatology).all():
        raise RuntimeError("PET climatology contains missing values")
    np.save(output_path, climatology)
    return climatology, output_path


def main() -> None:
    args = parse_args()
    if "_qet_seed" not in args.run:
        raise ValueError("This reviewer sensitivity is defined for Q+ET runs")
    torch.set_num_threads(args.threads)

    run_dir = PROJECT / "runs/frozen_refit" / args.run
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt["architecture"] not in {"dhbv", "fluxgate"}:
        raise ValueError("Sensitivity is only defined for constrained models")

    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations[stations.role.eq("target")].reset_index(drop=True)
    target_rows = target.row.to_numpy(int)
    static = np.load(PROJECT / "data/processed/static.npy", mmap_mode="r")
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    forcing_archive = np.load(PROJECT / "data/processed/forcing.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])

    anchor_file = PROJECT / receipt["antecedent_states"]
    anchor = np.load(anchor_file)
    if not np.array_equal(anchor["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Antecedent-state station order mismatch")

    output_dir = PROJECT / "outputs/law_interventions_pet_climatology"
    output_dir.mkdir(parents=True, exist_ok=True)
    pet_climatology, climatology_path = build_pet_climatology(
        forcing_archive, dates, target_rows, output_dir
    )

    checkpoint = torch.load(PROJECT / receipt["checkpoint"], map_location="cpu", weights_only=False)
    model = make_model(receipt["architecture"], static.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    # Double precision keeps the mass-residual audit below the prespecified
    # numerical tolerance for high-storage flux-gate states.
    model.double().eval()

    n_gauges, n_anchors, state_size = anchor["state"].shape
    n_samples = n_gauges * n_anchors
    state_flat = anchor["state"].reshape(n_samples, state_size)
    time_flat = anchor["eval_time_index"].reshape(n_samples)
    basin_flat = np.repeat(np.arange(n_gauges), n_anchors)
    q_curve = np.empty((n_samples, 91), dtype=np.float32)
    storage_curve = np.empty((n_samples, 91), dtype=np.float32)
    pet_curve = np.empty((n_samples, 91), dtype=np.float32)
    residual_max = 0.0

    with torch.no_grad():
        for first in range(0, n_samples, args.batch_size):
            stop = min(first + args.batch_size, n_samples)
            base_date = eval_dates[time_flat[first:stop]].astype("datetime64[D]")
            experiment_date = base_date[:, None] + np.arange(1, 92).astype("timedelta64[D]")[None, :]
            day_of_year = pd.DatetimeIndex(
                experiment_date.reshape(-1).astype("datetime64[ns]")
            ).dayofyear.to_numpy().reshape(stop - first, 91)
            local_basins = basin_flat[first:stop]
            local_pet = pet_climatology[local_basins[:, None], day_of_year - 1]
            forcing = np.zeros((stop - first, 91, 5), dtype=np.float32)
            forcing[:, :, 1] = local_pet
            forcing[:, :, 2] = 10.0
            forcing[:, :, 3] = np.sin(2 * np.pi * day_of_year / 365.25)
            forcing[:, :, 4] = np.cos(2 * np.pi * day_of_year / 365.25)
            rows = target_rows[local_basins]
            result = model(
                torch.from_numpy(forcing).double(),
                torch.from_numpy(np.asarray(static[rows])).double(),
                initial_state=torch.from_numpy(state_flat[first:stop]).double(),
                return_states=True,
            )
            q_curve[first:stop] = result["q"].cpu().numpy()
            storage_curve[first:stop] = result["storage"].cpu().numpy()
            pet_curve[first:stop] = local_pet
            residual_max = max(residual_max, float(result["daily_residual"].abs().max()))
            print(f"{args.run}: climatological-PET interventions {stop:,}/{n_samples:,}", flush=True)

    q_curve = q_curve.reshape(n_gauges, n_anchors, 91)
    storage_curve = storage_curve.reshape(n_gauges, n_anchors, 91)
    pet_curve = pet_curve.reshape(n_gauges, n_anchors, 91)
    records = []
    for basin in range(n_gauges):
        records.append({
            "GAGE_ID": target.GAGE_ID.iloc[basin],
            "architecture": receipt["architecture"],
            "rung": receipt["rung"],
            "seed": int(receipt["seed"]),
            "median_first_7d_pet_mm": float(np.median(pet_curve[basin, :, :7].sum(axis=1))),
            "median_91d_pet_mm": float(np.median(pet_curve[basin].sum(axis=1))),
            **summarize_basin(q_curve[basin], storage_curve[basin]),
        })

    curve_path = output_dir / f"{args.run}_curves.npz"
    table_path = output_dir / f"{args.run}_fingerprints.csv"
    np.savez_compressed(
        curve_path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        quantiles=anchor["quantiles"],
        antecedent_time_index=anchor["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
        pet_mm_day=pet_curve,
    )
    pd.DataFrame(records).to_csv(table_path, index=False)
    sensitivity_receipt = {
        "run": args.run,
        "analysis_phase": "post_evaluation_reviewer_sensitivity",
        "uses_target_discharge": False,
        "counterfactual": "zero precipitation with target-gauge day-of-year climatological PET",
        "pet_climatology_period": [str(CLIMATOLOGY_START), str(CLIMATOLOGY_END)],
        "temperature_c": 10.0,
        "primary_q0_threshold_mm_day": PRIMARY_Q0,
        "gauges": n_gauges,
        "anchors_per_gauge": n_anchors,
        "days_per_intervention": 91,
        "maximum_daily_mass_residual_mm": residual_max,
        "pet_climatology": str(climatology_path.relative_to(PROJECT)),
        "pet_climatology_sha256": sha256(climatology_path),
        "curves": str(curve_path.relative_to(PROJECT)),
        "curves_sha256": sha256(curve_path),
        "fingerprints": str(table_path.relative_to(PROJECT)),
        "fingerprints_sha256": sha256(table_path),
    }
    receipt_out = output_dir / f"{args.run}_receipt.json"
    receipt_out.write_text(json.dumps(sensitivity_receipt, indent=2) + "\n")
    print(json.dumps(sensitivity_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
