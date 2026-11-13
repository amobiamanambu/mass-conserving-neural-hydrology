#!/usr/bin/env python3
"""Outcome-blind calendar-day PET climatology for intervention sensitivities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
TIME_SERIES = ROOT / "data/generic_blind/time_series"
PET_ARCHIVE = ROOT / "derived/pet_climatology_2003_2018.npz"
CLIMATOLOGY_START = "2003-10-05"
CLIMATOLOGY_END = "2018-12-31"
# Store fixed-width Unicode labels so the NPZ archive remains readable with
# NumPy's safe default (allow_pickle=False).
CALENDAR_KEYS = np.asarray(
    pd.date_range("2000-01-01", "2000-12-31", freq="D").strftime("%m-%d"),
    dtype="U5",
)
CALENDAR_KEY_INDEX = {key: index for index, key in enumerate(CALENDAR_KEYS)}


@lru_cache(maxsize=1)
def _archive() -> tuple[dict[str, int], np.ndarray] | None:
    if not PET_ARCHIVE.exists():
        return None
    loaded = np.load(PET_ARCHIVE, allow_pickle=False)
    gauges = [str(value).zfill(8) for value in loaded["gauge_ids"]]
    keys = loaded["calendar_keys"].astype(str)
    if not np.array_equal(keys, CALENDAR_KEYS):
        raise RuntimeError("PET climatology archive calendar-key mismatch")
    values = loaded["pet_mm_day"].astype(np.float64)
    if values.shape != (len(gauges), len(CALENDAR_KEYS)):
        raise RuntimeError("PET climatology archive shape mismatch")
    return {gauge: index for index, gauge in enumerate(gauges)}, values


@lru_cache(maxsize=64)
def _calendar_day_means(gauge: str) -> dict[str, float]:
    """Return pre-test mean PET by month-day for one gauge."""
    path = TIME_SERIES / f"{str(gauge).zfill(8)}.nc"
    with xr.open_dataset(path) as dataset:
        pet = dataset["pet_mm"].sel(
            date=slice(CLIMATOLOGY_START, CLIMATOLOGY_END)
        ).to_series()
    pet = pet[np.isfinite(pet.to_numpy(float))]
    if pet.empty:
        raise RuntimeError(f"No finite pre-test PET values for gauge {gauge}")
    keys = pd.DatetimeIndex(pet.index).strftime("%m-%d")
    means = pet.groupby(keys).mean().to_dict()
    if len(means) < 365:
        raise RuntimeError(
            f"Incomplete calendar-day PET climatology for gauge {gauge}: {len(means)} days"
        )
    return {str(key): float(value) for key, value in means.items()}


def seasonal_pet_climatology(gauge: str, future_dates: np.ndarray) -> np.ndarray:
    """Map dates to the gauge's 2003-2018 calendar-day mean PET (mm d-1)."""
    dates = np.asarray(future_dates).astype("datetime64[D]")
    flat_keys = pd.DatetimeIndex(
        dates.reshape(-1).astype("datetime64[ns]")
    ).strftime("%m-%d")
    gauge = str(gauge).zfill(8)
    archive = _archive()
    if archive is None:
        means = _calendar_day_means(gauge)
        values = np.asarray([means[key] for key in flat_keys], dtype=np.float64)
    else:
        indices, matrix = archive
        if gauge not in indices:
            raise KeyError(f"Gauge {gauge} is absent from the PET climatology archive")
        values = matrix[indices[gauge], [CALENDAR_KEY_INDEX[key] for key in flat_keys]]
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError(f"Invalid PET climatology generated for gauge {gauge}")
    return values.reshape(dates.shape)
