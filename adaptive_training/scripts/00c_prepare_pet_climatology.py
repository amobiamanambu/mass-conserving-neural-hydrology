#!/usr/bin/env python3
"""Create one audited 2003-2018 calendar-day PET archive for target gauges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from _pet_climatology import (
    CALENDAR_KEYS,
    CLIMATOLOGY_END,
    CLIMATOLOGY_START,
    PET_ARCHIVE,
    _calendar_day_means,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    gauges = [
        gauge.zfill(8)
        for gauge in (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    ]
    values = np.full((len(gauges), len(CALENDAR_KEYS)), np.nan, dtype=np.float32)
    for index, gauge in enumerate(gauges):
        means = _calendar_day_means(gauge)
        values[index] = [means[key] for key in CALENDAR_KEYS]
        if (index + 1) % 250 == 0 or index + 1 == len(gauges):
            print(f"PET climatology {index + 1}/{len(gauges)}", flush=True)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError("PET climatology archive contains invalid values")
    PET_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PET_ARCHIVE,
        gauge_ids=np.asarray(gauges, dtype="U16"),
        calendar_keys=np.asarray(CALENDAR_KEYS, dtype="U5"),
        pet_mm_day=values,
    )
    receipt = {
        "archive": str(PET_ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256(PET_ARCHIVE),
        "gauges": len(gauges),
        "calendar_days": len(CALENDAR_KEYS),
        "period": [CLIMATOLOGY_START, CLIMATOLOGY_END],
        "target_discharge_accessed": False,
        "minimum_pet_mm_day": float(values.min()),
        "maximum_pet_mm_day": float(values.max()),
    }
    receipt_path = PET_ARCHIVE.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
