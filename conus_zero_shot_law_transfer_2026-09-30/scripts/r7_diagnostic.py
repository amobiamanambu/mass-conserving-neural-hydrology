#!/usr/bin/env python3
"""Portable zero-input drainage-response diagnostic.

The module accepts NumPy arrays or CPU/GPU PyTorch tensors containing modeled
discharge trajectories. The last dimension is elapsed day and must include day
0. The command-line interface accepts an NPZ archive with ``q_mm_day`` shaped
as ``(gauges, antecedent_states, elapsed_days)`` and writes one fingerprint per
gauge without requiring project-specific model code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_Q0_MM_DAY = 0.001
DEFAULT_HORIZONS = (7, 30, 90)


def _as_numpy(values: Any) -> np.ndarray:
    """Convert NumPy-like input or a PyTorch tensor to a float array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def persistence_class(r7: Any) -> np.ndarray:
    """Classify seven-day retention using 7- and 14-day half-life breaks."""
    values = _as_numpy(r7)
    classes = np.full(values.shape, -1, dtype=np.int8)
    finite = np.isfinite(values)
    classes[finite & (values < 0.5)] = 0
    classes[finite & (values >= 0.5) & (values < np.sqrt(0.5))] = 1
    classes[finite & (values >= np.sqrt(0.5))] = 2
    return classes


def trajectory_metrics(
    q_mm_day: Any,
    *,
    initial_flow_threshold: float = PRIMARY_Q0_MM_DAY,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, np.ndarray]:
    """Calculate retention, discharge half-life, and monotonicity per curve."""
    q = _as_numpy(q_mm_day)
    if q.ndim < 2:
        raise ValueError("q_mm_day must have at least a curve and elapsed-day dimension")
    if min(horizons) < 1 or max(horizons) >= q.shape[-1]:
        raise ValueError("Each requested horizon must be represented in q_mm_day")

    q0 = q[..., 0]
    valid = np.isfinite(q0) & (q0 >= initial_flow_threshold)
    result: dict[str, np.ndarray] = {"q0_mm_day": np.where(valid, q0, np.nan)}
    for horizon in horizons:
        ratio = np.full(q0.shape, np.nan, dtype=np.float64)
        ratio[valid] = q[..., horizon][valid] / q0[valid]
        result[f"r{horizon}"] = ratio

    below_half = q <= 0.5 * q0[..., None]
    has_crossing = np.any(below_half, axis=-1) & valid
    first_crossing = np.argmax(below_half, axis=-1).astype(np.float64)
    half_life = np.full(q0.shape, np.nan, dtype=np.float64)
    half_life[has_crossing] = first_crossing[has_crossing]
    half_life[valid & ~has_crossing] = float(q.shape[-1])
    result["discharge_half_life_days"] = half_life

    tolerance = np.maximum(1e-8, 1e-5 * q0)
    monotonic = np.mean(np.diff(q, axis=-1) <= tolerance[..., None], axis=-1)
    result["monotonic_fraction"] = np.where(valid, monotonic, np.nan)
    result["valid"] = valid
    return result


def gauge_fingerprints(
    q_mm_day: Any,
    *,
    initial_flow_threshold: float = PRIMARY_Q0_MM_DAY,
) -> dict[str, np.ndarray]:
    """Aggregate ``(gauges, antecedent_states, days)`` curves by state median."""
    q = _as_numpy(q_mm_day)
    if q.ndim != 3:
        raise ValueError("q_mm_day must have shape (gauges, antecedent_states, elapsed_days)")
    metrics = trajectory_metrics(q, initial_flow_threshold=initial_flow_threshold)
    output: dict[str, np.ndarray] = {
        "valid_anchors": np.sum(metrics["valid"], axis=1).astype(np.int32)
    }
    for key, values in metrics.items():
        if key == "valid":
            continue
        with np.errstate(all="ignore"):
            output[f"median_{key}"] = np.nanmedian(values, axis=1)
    output["persistence_class"] = persistence_class(output["median_r7"])
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", nargs="?", type=Path)
    parser.add_argument("output_csv", nargs="?", type=Path)
    parser.add_argument("--q-key", default="q_mm_day")
    parser.add_argument("--gage-key", default="gage_ids")
    parser.add_argument("--initial-flow-threshold", type=float, default=PRIMARY_Q0_MM_DAY)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _self_test() -> None:
    days = np.arange(91, dtype=np.float64)
    curve = 2.0 * np.power(0.5, days / 7.0)
    q = np.stack([curve, curve * 1.5], axis=0)[None, ...]
    result = gauge_fingerprints(q)
    if not np.isclose(result["median_r7"][0], 0.5, atol=1e-12):
        raise AssertionError("R7 self-test failed")
    if not np.isclose(result["median_discharge_half_life_days"][0], 7.0):
        raise AssertionError("Half-life self-test failed")
    if result["valid_anchors"][0] != 2:
        raise AssertionError("Validity self-test failed")
    print("R7 diagnostic self-test passed")


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    if args.input_npz is None or args.output_csv is None:
        raise SystemExit("input_npz and output_csv are required unless --self-test is used")

    with np.load(args.input_npz, allow_pickle=False) as archive:
        q = archive[args.q_key]
        if args.gage_key in archive:
            gage_ids = archive[args.gage_key].astype(str)
        else:
            gage_ids = np.arange(q.shape[0]).astype(str)
    metrics = gauge_fingerprints(q, initial_flow_threshold=args.initial_flow_threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["GAGE_ID", *metrics.keys()]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, gage_id in enumerate(gage_ids):
            writer.writerow({"GAGE_ID": gage_id, **{key: values[index] for key, values in metrics.items()}})
    print(args.output_csv)


if __name__ == "__main__":
    main()
