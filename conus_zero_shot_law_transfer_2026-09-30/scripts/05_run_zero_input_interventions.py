#!/usr/bin/env python3
"""Generate frozen zero-input law fingerprints before target-Q access."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Frozen run name, e.g. dhbv_q_seed11")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="Numerical precision for the intervention replay (default: float64).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--table-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Fingerprint table format (CSV is useful in minimal PyTorch environments).",
    )
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
        x = np.log(storage[positive]); y = np.log(q[positive])
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


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch_dtype = torch.float64 if args.precision == "float64" else torch.float32
    numpy_dtype = np.float64 if args.precision == "float64" else np.float32
    run_dir = PROJECT / "runs/frozen_refit" / args.run
    receipt_path = run_dir / "receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt["architecture"] not in {"dhbv", "fluxgate"}:
        raise ValueError("Interventions are only defined for constrained models")
    if receipt.get("target_discharge_accessed") is not False:
        raise RuntimeError("Outcome-blind receipt is not intact")

    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations[stations.role.eq("target")].reset_index(drop=True)
    static = np.load(PROJECT / "data/processed/static.npy", mmap_mode="r")
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    anchor_file = PROJECT / receipt["antecedent_states"]
    anchor = np.load(anchor_file)
    if not np.array_equal(anchor["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Antecedent-state station order mismatch")

    checkpoint = torch.load(PROJECT / receipt["checkpoint"], map_location="cpu", weights_only=False)
    model = make_model(receipt["architecture"], static.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(dtype=torch_dtype)
    model.eval()

    n_gauges, n_anchors, state_size = anchor["state"].shape
    n_samples = n_gauges * n_anchors
    state_flat = anchor["state"].reshape(n_samples, state_size).astype(numpy_dtype, copy=False)
    time_flat = anchor["eval_time_index"].reshape(n_samples)
    basin_flat = np.repeat(np.arange(n_gauges), n_anchors)
    q_curve = np.empty((n_samples, 91), dtype=numpy_dtype)
    storage_curve = np.empty((n_samples, 91), dtype=numpy_dtype)
    residual_max = 0.0

    with torch.no_grad():
        for first in range(0, n_samples, args.batch_size):
            stop = min(first + args.batch_size, n_samples)
            base_date = eval_dates[time_flat[first:stop]].astype("datetime64[D]")
            experiment_date = base_date[:, None] + np.arange(1, 92).astype("timedelta64[D]")[None, :]
            day_of_year = pd.DatetimeIndex(experiment_date.reshape(-1).astype("datetime64[ns]")).dayofyear.to_numpy().reshape(stop - first, 91)
            forcing = np.zeros((stop - first, 91, 5), dtype=numpy_dtype)
            forcing[:, :, 2] = 10.0
            forcing[:, :, 3] = np.sin(2 * np.pi * day_of_year / 365.25)
            forcing[:, :, 4] = np.cos(2 * np.pi * day_of_year / 365.25)
            rows = target.row.to_numpy(int)[basin_flat[first:stop]]
            result = model(
                torch.from_numpy(forcing),
                torch.from_numpy(np.asarray(static[rows], dtype=numpy_dtype)),
                initial_state=torch.from_numpy(state_flat[first:stop]),
                return_states=True,
            )
            q_curve[first:stop] = result["q"].cpu().numpy()
            storage_curve[first:stop] = result["storage"].cpu().numpy()
            residual_max = max(residual_max, float(result["daily_residual"].abs().max()))
            print(f"{args.run}: interventions {stop:,}/{n_samples:,}", flush=True)

    q_curve = q_curve.reshape(n_gauges, n_anchors, 91)
    storage_curve = storage_curve.reshape(n_gauges, n_anchors, 91)
    records = []
    for basin in range(n_gauges):
        records.append({
            "GAGE_ID": target.GAGE_ID.iloc[basin],
            "architecture": receipt["architecture"],
            "rung": receipt["rung"],
            "seed": int(receipt["seed"]),
            **summarize_basin(q_curve[basin], storage_curve[basin]),
        })

    output_dir = args.output_dir or (PROJECT / "outputs/law_interventions")
    if not output_dir.is_absolute():
        output_dir = PROJECT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / f"{args.run}_curves.npz"
    table_path = output_dir / f"{args.run}_fingerprints.{args.table_format}"
    np.savez_compressed(
        curve_path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        quantiles=anchor["quantiles"],
        antecedent_time_index=anchor["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    fingerprints = pd.DataFrame(records)
    if args.table_format == "parquet":
        fingerprints.to_parquet(table_path, index=False)
    else:
        fingerprints.to_csv(table_path, index=False)
    law_receipt = {
        "run": args.run,
        "target_discharge_accessed": False,
        "primary_q0_threshold_mm_day": PRIMARY_Q0,
        "gauges": n_gauges,
        "anchors_per_gauge": n_anchors,
        "days_per_intervention": 91,
        "numerical_precision": args.precision,
        "maximum_daily_mass_residual_mm": residual_max,
        "curves": str(curve_path.relative_to(PROJECT)),
        "curves_sha256": sha256(curve_path),
        "fingerprints": str(table_path.relative_to(PROJECT)),
        "fingerprints_sha256": sha256(table_path),
    }
    receipt_out = output_dir / f"{args.run}_receipt.json"
    receipt_out.write_text(json.dumps(law_receipt, indent=2) + "\n")
    print(json.dumps(law_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
