#!/usr/bin/env python3
"""Harmonize daily process observations with the frozen model calendar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
DERIVED = PROJECT / "data" / "derived"
MODEL = PROJECT / "data" / "model_ready"
PROVENANCE = PROJECT / "provenance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nldas_relative_saturation(values: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, dict]:
    dt = pd.DatetimeIndex(dates.astype("datetime64[ns]"))
    train = (dt.year >= 2005) & (dt.year <= 2014)
    baseline = values[:, train]
    lower = np.nanpercentile(baseline, 5, axis=1).astype(np.float32)
    upper = np.nanpercentile(baseline, 95, axis=1).astype(np.float32)
    span = np.maximum(upper - lower, 20.0).astype(np.float32)
    saturation = ((values - lower[:, None]) / span[:, None]).astype(np.float32)
    saturation = np.clip(saturation, 0.0, 1.0)
    saturation[~np.isfinite(values)] = np.nan
    return saturation, {
        "baseline_years": [2005, 2014],
        "lower_percentile": 5,
        "upper_percentile": 95,
        "minimum_percentile_span_mm": 20.0,
        "lower_mm_median": float(np.nanmedian(lower)),
        "upper_mm_median": float(np.nanmedian(upper)),
    }


def load_nldas(stations: pd.DataFrame, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root = np.full((len(stations), len(dates)), np.nan, np.float32)
    coverage = np.full_like(root, np.nan)
    row = {gage: i for i, gage in enumerate(stations.GAGE_ID)}
    time = {date: i for i, date in enumerate(dates)}
    files = sorted((DERIVED / "nldas_daily").glob("nldas_daily_*.npz"))
    if not files:
        raise FileNotFoundError("No annual NLDAS archive files")
    for path in files:
        with np.load(path) as data:
            local_ids = data["gage_ids"].astype(str)
            if local_ids.tolist() != stations.GAGE_ID.tolist():
                raise RuntimeError(f"Gauge order mismatch in {path}")
            local_dates = data["dates"].astype("datetime64[D]")
            columns = np.asarray([time.get(date, -1) for date in local_dates], int)
            keep = columns >= 0
            root[:, columns[keep]] = data["values"][:, keep, 1]
            coverage[:, columns[keep]] = data["valid_fraction"][:, keep, 1]
    root[coverage < 0.75] = np.nan
    return root, coverage


def load_ssebop(stations: pd.DataFrame, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = stations[stations.role.isin(["donor", "development"])].copy()
    files = sorted((DERIVED / "ssebop_monthly").glob("ssebop_monthly_*.npz"))
    if not files:
        raise FileNotFoundError("No annual SSEBop archive files")
    month_parts, eta_parts, coverage_parts = [], [], []
    for path in files:
        with np.load(path) as data:
            local_ids = data["gage_ids"].astype(str)
            if local_ids.tolist() != selected.GAGE_ID.tolist():
                raise RuntimeError(f"Gauge order mismatch in {path}")
            month_parts.append(data["dates"].astype("datetime64[D]"))
            eta_parts.append(data["eta_mm_month"].astype(np.float32))
            coverage_parts.append(data["valid_fraction"].astype(np.float32))
    months = np.concatenate(month_parts)
    eta = np.concatenate(eta_parts, axis=1)
    coverage = np.concatenate(coverage_parts, axis=1)
    order = np.argsort(months)
    months, eta, coverage = months[order], eta[:, order], coverage[:, order]
    eta[coverage < 0.75] = np.nan
    starts = np.searchsorted(dates, months).astype(np.int32)
    next_month = (pd.DatetimeIndex(months.astype("datetime64[ns]")) + pd.offsets.MonthBegin(1)).to_numpy(dtype="datetime64[D]")
    stops = np.searchsorted(dates, next_month).astype(np.int32)
    return eta, starts, stops, months


def main() -> None:
    MODEL.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(SOURCE / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    dates = np.load(SOURCE / "data/processed/dates.npy", mmap_mode="r").astype("datetime64[D]")

    rootzone, coverage = load_nldas(stations, dates)
    saturation, operator_meta = nldas_relative_saturation(rootzone, dates)
    np.save(MODEL / "nldas_rootzone_saturation.npy", saturation)
    np.save(MODEL / "nldas_rootzone_coverage.npy", coverage.astype(np.float16))
    del saturation, coverage

    eta, starts, stops, labels = load_ssebop(stations, dates)
    selected = stations[stations.role.isin(["donor", "development"])]
    np.savez_compressed(
        MODEL / "ssebop_monthly.npz", gage_ids=selected.GAGE_ID.to_numpy(str),
        block_start=starts, block_stop=stops, block_date=labels,
        eta_month_mm=eta,
    )

    receipts = []
    for path in sorted(MODEL.glob("*")):
        if path.is_file():
            receipts.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(receipts).to_csv(PROVENANCE / "model_ready_checksums.csv", index=False)
    summary = {
        "gauges": len(stations), "days": len(dates),
        "nldas_rootzone_finite_fraction": float(np.isfinite(rootzone).mean()),
        "nldas_observation_operator": operator_meta,
        "ssebop_gauges": len(selected), "ssebop_months": int(eta.shape[1]),
        "ssebop_complete_fraction": float(np.isfinite(eta).mean()),
    }
    (PROVENANCE / "process_tensor_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
