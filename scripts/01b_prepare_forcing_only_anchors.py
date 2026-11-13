#!/usr/bin/env python3
"""Freeze outcome-blind national intervention anchors from meteorological forcing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "source_inputs" / "processed"
OUTPUT = ROOT / "data" / "forcing_only_anchors.npz"
RECEIPT = ROOT / "provenance" / "forcing_only_anchor_receipt.json"
QUANTILES = np.linspace(0.05, 0.95, 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() or RECEIPT.exists():
        raise FileExistsError("Refusing to overwrite the frozen forcing-only anchors")

    target_path = ROOT / "data" / "basin_lists" / "test_3982.txt"
    targets = target_path.read_text().split()
    if len(targets) != 3982 or len(set(targets)) != 3982:
        raise RuntimeError("The frozen target list must contain 3,982 unique gauges")

    stations = pd.read_csv(PROCESSED / "station_index.csv", dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    row_by_gauge = dict(zip(stations.GAGE_ID, stations.row.astype(int), strict=True))
    if not set(targets).issubset(row_by_gauge):
        raise RuntimeError("At least one frozen target is absent from station_index.csv")

    dates = np.load(PROCESSED / "dates.npy", mmap_mode="r")
    forcing = np.load(PROCESSED / "forcing.npy", mmap_mode="r")
    start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    stop = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    test_dates = np.asarray(dates[start:stop], dtype="datetime64[D]")
    if len(test_dates) != 2192:
        raise RuntimeError(f"Expected 2,192 target dates, found {len(test_dates)}")

    months = pd.DatetimeIndex(test_dates.astype("datetime64[ns]")).month.to_numpy()
    warm = (months >= 6) & (months <= 9)
    selected = np.full((len(targets), len(QUANTILES)), -1, dtype=np.int16)
    fallback = np.zeros(len(targets), dtype=np.int8)

    for basin_index, gauge in enumerate(targets):
        row = row_by_gauge[gauge]
        x = np.asarray(forcing[row, start:stop, :3], dtype=np.float64)
        daily_balance = x[:, 0] - x[:, 1]
        antecedent_balance = np.convolve(daily_balance, np.ones(30), mode="full")[: len(daily_balance)]
        antecedent_balance[:29] = np.nan
        candidates = np.flatnonzero(warm & (x[:, 2] > 5.0) & np.isfinite(antecedent_balance))
        if len(candidates) < len(QUANTILES):
            fallback[basin_index] = 1
            candidates = np.flatnonzero(warm & np.isfinite(antecedent_balance))
        if len(candidates) < len(QUANTILES):
            raise RuntimeError(f"Too few forcing-only candidates for {gauge}")

        targets_at_quantiles = np.quantile(antecedent_balance[candidates], QUANTILES)
        used: set[int] = set()
        chosen: list[int] = []
        for value in targets_at_quantiles:
            order = candidates[
                np.argsort(np.abs(antecedent_balance[candidates] - value), kind="stable")
            ]
            index = next((int(item) for item in order if int(item) not in used), None)
            if index is None:
                raise RuntimeError(f"Could not select 12 unique anchors for {gauge}")
            used.add(index)
            chosen.append(index)
        selected[basin_index] = np.asarray(chosen, dtype=np.int16)

    if np.any(selected < 0) or np.any(selected >= len(test_dates)):
        raise RuntimeError("An anchor index falls outside the NeuralHydrology test period")
    if any(len(np.unique(row)) != 12 for row in selected):
        raise RuntimeError("Every gauge must have 12 unique forcing-only anchors")

    anchor_dates = test_dates[selected]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        gage_ids=np.asarray(targets),
        quantiles=QUANTILES.astype(np.float32),
        eval_time_index=selected,
        anchor_dates=anchor_dates,
        temperature_filter_fallback=fallback,
        definition="warm-season quantiles of antecedent 30-day precipitation minus PET",
        target_discharge_accessed=np.asarray(False),
    )

    receipt = {
        "archive": str(OUTPUT.relative_to(ROOT)),
        "archive_sha256": sha256(OUTPUT),
        "target_gauges": len(targets),
        "anchors_per_gauge": int(selected.shape[1]),
        "gauges_with_complete_anchors": int(np.all(selected >= 0, axis=1).sum()),
        "temperature_filter_fallback_gauges": int(fallback.sum()),
        "selection_period": [str(test_dates[0]), str(test_dates[-1])],
        "selection_rule": "12 quantiles (0.05--0.95) of antecedent 30-day P-PET during June--September; T > 5 C when at least 12 candidates exist",
        "uses_target_discharge": False,
        "uses_model_predictions_or_states": False,
        "target_list_sha256": sha256(target_path),
        "forcing_sha256": sha256(PROCESSED / "forcing.npy"),
        "dates_sha256": sha256(PROCESSED / "dates.npy"),
        "station_index_sha256": sha256(PROCESSED / "station_index.csv"),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
