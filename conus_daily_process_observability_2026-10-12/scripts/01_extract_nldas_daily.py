#!/usr/bin/env python3
"""Stream daily 12 UTC NLDAS soil storage and archive basin-day arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
import shapely
import xarray as xr


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
CACHE = PROJECT / "data" / "cache" / "nldas"
OUT = PROJECT / "data" / "derived" / "nldas_daily"
PROVENANCE = PROJECT / "provenance" / "nldas"
WEIGHTS = PROJECT / "data" / "weights" / "nldas_basin_grid_weights.npz"
GAUGES = PROJECT / "data" / "weights" / "nldas_gauge_order.csv"
WEIGHT_META = PROJECT / "data" / "weights" / "nldas_weight_metadata.json"
BASE = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_NOAH0125_H.2.0"
VARIABLES = ["SoilM_0_10cm", "SoilM_0_100cm"]
THREAD_LOCAL = threading.local()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2003-10-05")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-days", type=int, default=24)
    parser.add_argument("--years", nargs="*", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_for(date: pd.Timestamp) -> tuple[str, str]:
    name = f"NLDAS_NOAH0125_H.A{date:%Y%m%d}.1200.020.nc"
    return name, f"{BASE}/{date.year}/{date.dayofyear:03d}/{name}"


def download(date: pd.Timestamp) -> dict:
    name, url = source_for(date)
    target = CACHE / name
    part = target.with_suffix(target.suffix + ".part")
    cookie = getattr(THREAD_LOCAL, "cookie", None)
    if cookie is None:
        cookie = Path("/tmp") / f"nldas_urs_cookie_{threading.get_ident()}"
        THREAD_LOCAL.cookie = cookie
    started = time.time()
    command = [
        "curl", "-L", "--fail", "--silent", "--show-error", "--netrc",
        "--cookie", str(cookie), "--cookie-jar", str(cookie),
        "--max-redirs", "10", "--retry", "4", "--retry-all-errors",
        "--connect-timeout", "20", "--max-time", "300", url, "-o", str(part),
    ]
    try:
        subprocess.run(command, check=True)
        part.replace(target)
        return {
            "date": date, "name": name, "url": url, "path": target,
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "download_seconds": time.time() - started, "status": "downloaded",
        }
    except Exception as exc:
        part.unlink(missing_ok=True)
        return {
            "date": date, "name": name, "url": url,
            "download_seconds": time.time() - started,
            "status": "failed", "error": repr(exc),
        }


def load_basins() -> gpd.GeoDataFrame:
    stations = pd.read_csv(SOURCE / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    ids = stations.GAGE_ID.str.zfill(8).tolist()
    basins = gpd.read_file(ROOT / "gridmet_data/GAGE_II_Basins.shp")[["GAGE_ID", "geometry"]].to_crs(4326)
    basins["GAGE_ID"] = basins.GAGE_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    basins = basins.drop_duplicates("GAGE_ID").set_index("GAGE_ID").reindex(ids).reset_index()
    if basins.geometry.isna().any():
        raise RuntimeError(f"Missing {int(basins.geometry.isna().sum())} basin polygons")
    basins["geometry"] = shapely.make_valid(basins.geometry.to_numpy())
    return basins


def build_weights(basins: gpd.GeoDataFrame, latitude: np.ndarray, longitude: np.ndarray) -> sparse.csr_matrix:
    dy = float(np.median(np.diff(latitude)))
    dx = float(np.median(np.diff(longitude)))
    lat_edges = np.r_[latitude - dy / 2, latitude[-1] + dy / 2]
    lon_edges = np.r_[longitude - dx / 2, longitude[-1] + dx / 2]
    rows, cols, vals = [], [], []
    nlat, nlon = len(latitude), len(longitude)
    for row, geometry in enumerate(basins.geometry):
        minx, miny, maxx, maxy = geometry.bounds
        x0 = max(0, int(np.searchsorted(lon_edges, minx, side="right") - 1))
        x1 = min(nlon, int(np.searchsorted(lon_edges, maxx, side="left") + 1))
        y0 = max(0, int(np.searchsorted(lat_edges, miny, side="right") - 1))
        y1 = min(nlat, int(np.searchsorted(lat_edges, maxy, side="left") + 1))
        iy, ix = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        iy, ix = iy.ravel(), ix.ravel()
        cells = shapely.box(lon_edges[ix], lat_edges[iy], lon_edges[ix + 1], lat_edges[iy + 1])
        overlap = shapely.area(shapely.intersection(cells, geometry))
        keep = np.isfinite(overlap) & (overlap > 0)
        if not keep.any():
            raise RuntimeError(f"No NLDAS overlap for {basins.iloc[row].GAGE_ID}")
        iy, ix, overlap = iy[keep], ix[keep], overlap[keep]
        area = overlap * np.cos(np.deg2rad(latitude[iy]))
        rows.append(np.full(len(ix), row, np.int32))
        cols.append((iy * nlon + ix).astype(np.int32))
        vals.append(area.astype(np.float64))
        if (row + 1) % 500 == 0 or row + 1 == len(basins):
            print(f"NLDAS weights {row + 1:,}/{len(basins):,}", flush=True)
    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(basins), nlat * nlon),
    )
    matrix = sparse.diags(1.0 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(WEIGHTS, matrix, compressed=True)
    pd.DataFrame({"row": np.arange(len(basins)), "GAGE_ID": basins.GAGE_ID}).to_csv(GAUGES, index=False)
    WEIGHT_META.write_text(json.dumps({
        "method": "exact polygon-cell overlap in lon-lat with cosine-latitude area correction",
        "shape": list(matrix.shape), "nonzero_weights": int(matrix.nnz),
        "latitude": [float(latitude[0]), float(latitude[-1])],
        "longitude": [float(longitude[0]), float(longitude[-1])],
        "dx_degrees": dx, "dy_degrees": dy,
    }, indent=2) + "\n")
    return matrix.tocsr()


def load_or_build_weights(basins: gpd.GeoDataFrame, sample: Path) -> sparse.csr_matrix:
    with xr.open_dataset(sample) as ds:
        latitude = ds.lat.values.astype(float)
        longitude = ds.lon.values.astype(float)
    if WEIGHTS.exists() and GAUGES.exists():
        order = pd.read_csv(GAUGES, dtype={"GAGE_ID": str}).GAGE_ID.str.zfill(8).tolist()
        if order != basins.GAGE_ID.tolist():
            raise RuntimeError("Saved NLDAS weights use another gauge order")
        matrix = sparse.load_npz(WEIGHTS).tocsr()
        if matrix.shape != (len(basins), len(latitude) * len(longitude)):
            raise RuntimeError("Saved NLDAS weights use another grid")
        return matrix
    return build_weights(basins, latitude, longitude)


def basin_mean(matrix: sparse.csr_matrix, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, np.float64).ravel()
    valid = np.isfinite(flat) & (flat >= 0)
    denominator = np.asarray(matrix @ valid.astype(np.float64)).ravel()
    numerator = np.asarray(matrix @ np.where(valid, flat, 0.0)).ravel()
    mean = np.divide(numerator, denominator, out=np.full(matrix.shape[0], np.nan), where=denominator > 0)
    return mean.astype(np.float32), denominator.astype(np.float32)


def extract(path: Path, matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    values, coverage = [], []
    with xr.open_dataset(path) as ds:
        for variable in VARIABLES:
            mean, valid = basin_mean(matrix, ds[variable].isel(time=0).values)
            values.append(mean); coverage.append(valid)
    return np.stack(values, axis=1), np.stack(coverage, axis=1)


def process_year(year: int, dates: pd.DatetimeIndex, basins: gpd.GeoDataFrame, args: argparse.Namespace) -> None:
    output = OUT / f"nldas_daily_{year}.npz"
    manifest_path = PROVENANCE / f"nldas_manifest_{year}.csv"
    if output.exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if len(manifest) == len(dates) and manifest.status.eq("ok").all():
            print(f"{year}: already complete", flush=True)
            return
    values = np.full((len(basins), len(dates), len(VARIABLES)), np.nan, np.float32)
    coverage = np.full_like(values, np.nan)
    records: list[dict] = []
    matrix: sparse.csr_matrix | None = None
    for first in range(0, len(dates), args.batch_days):
        batch = dates[first:first + args.batch_days]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, pd.Timestamp(date)): pd.Timestamp(date) for date in batch}
            downloaded = [future.result() for future in as_completed(futures)]
        downloaded.sort(key=lambda row: row["date"])
        for record in downloaded:
            date = pd.Timestamp(record["date"])
            index = int(np.searchsorted(dates.values, date.to_datetime64()))
            if record["status"] != "downloaded":
                records.append({**record, "date": date.date(), "path": ""})
                continue
            path = Path(record["path"])
            started = time.time()
            try:
                if matrix is None:
                    matrix = load_or_build_weights(basins, path)
                values[:, index], coverage[:, index] = extract(path, matrix)
                records.append({
                    **record, "date": date.date(), "path": path.name, "status": "ok",
                    "extraction_seconds": time.time() - started,
                    "valid_rootzone_gauges": int(np.isfinite(values[:, index, 1]).sum()),
                })
            except Exception as exc:
                records.append({
                    **record, "date": date.date(), "path": path.name,
                    "status": "extract_failed", "error": repr(exc),
                    "extraction_seconds": time.time() - started,
                })
            finally:
                path.unlink(missing_ok=True)
        print(f"{year}: {min(first + len(batch), len(dates))}/{len(dates)} days", flush=True)
    manifest = pd.DataFrame(records).sort_values("date")
    manifest.to_csv(manifest_path, index=False)
    if len(manifest) != len(dates) or not manifest.status.eq("ok").all():
        raise RuntimeError(f"{year}: incomplete extraction; inspect {manifest_path}")
    np.savez_compressed(
        output, dates=dates.values.astype("datetime64[D]"),
        gage_ids=basins.GAGE_ID.to_numpy(str), variables=np.asarray(VARIABLES),
        values=values, valid_fraction=coverage,
    )
    print(f"{year}: wrote {output}", flush=True)


def main() -> None:
    args = arguments()
    for directory in [CACHE, OUT, PROVENANCE, WEIGHTS.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    basins = load_basins()
    all_dates = pd.date_range(args.start, args.end, freq="D")
    years = args.years or sorted(all_dates.year.unique())
    for year in years:
        dates = all_dates[all_dates.year == year]
        if len(dates):
            process_year(year, dates, basins, args)
    files = sorted(OUT.glob("nldas_daily_*.npz"))
    summary = {
        "product": "NLDAS_NOAH0125_H 2.0", "doi": "10.5067/T4OW83T8EXDO",
        "snapshot_utc": "12:00", "variables": VARIABLES, "gauges": len(basins),
        "annual_files": len(files), "raw_files_retained": False,
    }
    (PROVENANCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
