#!/usr/bin/env python3
"""Audit the complete dummy-target dataset before any national predictions are frozen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
TARGET_START = "2019-01-01"
TARGET_END = "2024-12-31"
TARGET_GAUGES = 3982
TARGET_DAYS = 2192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    gauges_path = ROOT / "data/basin_lists/test_3982.txt"
    gauges = gauges_path.read_text().split()
    if len(gauges) != TARGET_GAUGES or len(set(gauges)) != TARGET_GAUGES:
        raise RuntimeError("Target basin list is not 3,982 unique gauges")

    records = []
    for index, gauge in enumerate(gauges, start=1):
        path = ROOT / "data/generic_blind/time_series" / f"{gauge}.nc"
        with xr.open_dataset(path) as dataset:
            target = dataset["q_mm_day"].sel(date=slice(TARGET_START, TARGET_END))
            values = np.asarray(target.values)
            if values.shape != (TARGET_DAYS,):
                raise RuntimeError(f"Unexpected target-period length for {gauge}: {values.shape}")
            if not np.all(values == 0):
                raise RuntimeError(f"Non-dummy target discharge found in {path}")
            if int(dataset.attrs.get("target_discharge_blinded", 0)) != 1:
                raise RuntimeError(f"Missing blinded-target marker in {path}")
            if str(dataset.attrs.get("role", "")) != "target":
                raise RuntimeError(f"Incorrect role marker in {path}")
        records.append({
            "gauge_id": gauge,
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        if index % 500 == 0:
            print(f"Audited {index}/{TARGET_GAUGES} blinded target files", flush=True)

    inventory_path = ROOT / "provenance/blind_target_file_inventory.json"
    inventory_path.write_text(json.dumps(records, indent=2) + "\n")
    receipt = {
        "target_gauges": TARGET_GAUGES,
        "target_days": TARGET_DAYS,
        "target_period": [TARGET_START, TARGET_END],
        "all_target_values_are_dummy_zero": True,
        "all_files_marked_blinded_target": True,
        "target_discharge_accessed": False,
        "target_basin_file": str(gauges_path.relative_to(ROOT)),
        "target_basin_file_sha256": sha256(gauges_path),
        "inventory": str(inventory_path.relative_to(ROOT)),
        "inventory_sha256": sha256(inventory_path),
    }
    receipt_path = ROOT / "provenance/blind_target_verification.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
