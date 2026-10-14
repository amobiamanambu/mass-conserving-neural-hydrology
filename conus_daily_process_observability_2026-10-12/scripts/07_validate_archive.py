#!/usr/bin/env python3
"""Validate the compact experiment archive and write a checksum manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_annual(product: str, start: str, end: str, frequency: str = "D") -> dict:
    dates = pd.date_range(start, end, freq=frequency)
    directory = PROJECT / "data/derived" / product
    found = []
    for year in sorted(dates.year.unique()):
        path = directory / f"{product}_{year}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        archive = np.load(path)
        expected = dates[dates.year == year].to_numpy(dtype="datetime64[D]")
        if not np.array_equal(archive["dates"].astype("datetime64[D]"), expected):
            raise RuntimeError(f"Date mismatch in {path}")
        found.append(path)
    return {
        "files": len(found), "days": len(dates),
        "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()),
        "bytes": int(sum(path.stat().st_size for path in found)),
    }


def validate_manifests(directory: Path) -> dict:
    paths = sorted(directory.glob("*_manifest_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No manifests in {directory}")
    rows = 0
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.status.eq("ok").all():
            raise RuntimeError(f"Non-OK record in {path}")
        if frame.sha256.isna().any() or frame.bytes.le(0).any():
            raise RuntimeError(f"Incomplete provenance in {path}")
        rows += len(frame)
    return {"files": len(paths), "records": rows}


def main() -> None:
    nldas = validate_annual("nldas_daily", "2003-10-05", "2024-12-31")
    ssebop = validate_annual("ssebop_monthly", "2015-01-01", "2018-12-31", "MS")
    nldas_manifest = validate_manifests(PROJECT / "provenance/nldas")
    ssebop_manifest = validate_manifests(PROJECT / "provenance/ssebop")
    ready = [
        PROJECT / "data/model_ready/nldas_rootzone_saturation.npy",
        PROJECT / "data/model_ready/nldas_rootzone_coverage.npy",
        PROJECT / "data/model_ready/ssebop_monthly.npz",
    ]
    for path in ready:
        if not path.exists():
            raise FileNotFoundError(path)
    expected_runs = 2 * 4 * 3
    receipts = sorted((PROJECT / "outputs").glob("*/receipt.json"))
    if len(receipts) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} evaluated runs, found {len(receipts)}")
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt["maximum_float64_daily_mass_residual_mm"] > 1e-8:
            raise RuntimeError(f"Daily mass-closure tolerance failed: {path}")
        if receipt["maximum_float64_cumulative_mass_residual_mm"] > 1e-8:
            raise RuntimeError(f"Cumulative mass-closure tolerance failed: {path}")
    if not (PROJECT / "results/decision.json").exists():
        raise FileNotFoundError(PROJECT / "results/decision.json")

    packages = {}
    for name in ["numpy", "pandas", "scipy", "torch", "geopandas", "rasterio", "shapely", "xarray", "netCDF4"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages,
    }
    environment_path = PROJECT / "provenance/software_environment.json"
    environment_path.write_text(json.dumps(environment, indent=2) + "\n")

    include = [
        "config", "protocols", "scripts", "provenance", "data/derived",
        "data/model_ready", "data/weights", "data/retired_daily_probe",
        "runs", "outputs", "results",
    ]
    rows = []
    for item in include:
        root = PROJECT / item
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or "/cache/" in str(path):
                continue
            rows.append({
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = pd.DataFrame(rows)
    manifest_path = PROJECT / "provenance/archive_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "status": "valid", "nldas_daily": nldas, "ssebop_monthly": ssebop,
        "nldas_source_manifest": nldas_manifest, "ssebop_source_manifest": ssebop_manifest,
        "evaluated_runs": len(receipts), "mass_closure_tolerance_mm": 1e-8,
        "archived_files": len(manifest),
        "archived_bytes_excluding_raw_granule_cache": int(manifest.bytes.sum()),
        "raw_granules_retained": False,
        "software_environment": str(environment_path.relative_to(PROJECT)),
        "archive_manifest": str(manifest_path.relative_to(PROJECT)),
    }
    (PROJECT / "provenance/archive_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
