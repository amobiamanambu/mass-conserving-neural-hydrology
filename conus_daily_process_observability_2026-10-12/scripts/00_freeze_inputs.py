#!/usr/bin/env python3
"""Freeze input identities and verify the inherited spatial split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
PROVENANCE = PROJECT / "provenance"
FROZEN = PROJECT / "data" / "frozen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    files = [
        SOURCE / "data/processed/station_index.csv",
        SOURCE / "data/processed/dates.npy",
        SOURCE / "data/processed/forcing.npy",
        SOURCE / "data/processed/forcing_originally_valid.npy",
        SOURCE / "data/processed/q_donor_development.npy",
        SOURCE / "data/processed/static.npy",
        SOURCE / "data/processed/static_normalization.csv",
        SOURCE / "data/frozen/station_split.csv",
        SOURCE / "scripts/hydro_models.py",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    stations = pd.read_csv(files[0], dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    expected = {"donor": 500, "development": 200, "target": 3982}
    observed = stations.role.value_counts().to_dict()
    if observed != expected:
        raise RuntimeError(f"Spatial split changed: expected {expected}, found {observed}")
    if stations.GAGE_ID.duplicated().any():
        raise RuntimeError("Gauge IDs are not unique")
    dates = np.load(files[1], mmap_mode="r")
    if str(dates[0]) != "2003-10-05" or str(dates[-1]) != "2024-12-31":
        raise RuntimeError(f"Unexpected date support: {dates[[0, -1]]}")

    manifest = []
    for path in files:
        manifest.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest).to_csv(PROVENANCE / "frozen_input_checksums.csv", index=False)
    stations.to_csv(FROZEN / "station_index.csv", index=False)
    summary = {
        "source_project": str(SOURCE),
        "roles": observed,
        "gauges": int(len(stations)),
        "dates": [str(dates[0]), str(dates[-1])],
        "days": int(len(dates)),
        "files_hashed": len(files),
        "target_outcomes_used": False,
    }
    (PROVENANCE / "frozen_input_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
