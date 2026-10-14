#!/usr/bin/env python3
"""Stream monthly SSEBop ET and archive donor/development basin means."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy import sparse
import shapely


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
CACHE = PROJECT / "data" / "cache" / "ssebop"
OUT = PROJECT / "data" / "derived" / "ssebop_monthly"
PROVENANCE = PROJECT / "provenance" / "ssebop"
WEIGHT_DIR = PROJECT / "data" / "weights"
ANALYSIS_GRID = WEIGHT_DIR / "ssebop_monthly_analysis_grid.json"
BASE = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads"
SCALE = 1.0
DOWNSAMPLE = 4


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-months", type=int, default=12)
    parser.add_argument("--years", nargs="*", type=int)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_for(date: pd.Timestamp) -> tuple[str, str]:
    stem = f"m{date.year}{date.month:02d}"
    return f"{stem}.zip", f"{BASE}/{stem}.zip"


def download(date: pd.Timestamp) -> dict:
    name, url = source_for(date)
    started = time.time()
    part = CACHE / f"{name}.part"
    part.unlink(missing_ok=True)
    command = [
        "curl", "-L", "--fail", "--silent", "--show-error",
        "--retry", "6", "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "20", "--max-time", "300",
        "--speed-time", "45", "--speed-limit", "1024",
        "--user-agent", "CONUS-hydrology-research/1.0",
        url, "-o", str(part),
    ]
    try:
        subprocess.run(command, check=True)
        content = part.read_bytes()
        return {
            "date": date, "name": name, "url": url, "content": content,
            "bytes": len(content), "sha256": sha256_bytes(content),
            "download_seconds": time.time() - started, "status": "downloaded",
        }
    except Exception as exc:
        error = exc
    finally:
        part.unlink(missing_ok=True)
    return {
        "date": date, "name": name, "url": url,
        "download_seconds": time.time() - started,
        "status": "failed", "error": repr(error),
    }


def load_basins() -> gpd.GeoDataFrame:
    stations = pd.read_csv(SOURCE / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    stations = stations[stations.role.isin(["donor", "development"])].copy()
    ids = stations.GAGE_ID.str.zfill(8).tolist()
    basins = gpd.read_file(ROOT / "gridmet_data/GAGE_II_Basins.shp")[["GAGE_ID", "geometry"]].to_crs(4326)
    basins["GAGE_ID"] = basins.GAGE_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    basins = basins.drop_duplicates("GAGE_ID").set_index("GAGE_ID").reindex(ids).reset_index()
    if basins.geometry.isna().any():
        raise RuntimeError(f"Missing {int(basins.geometry.isna().sum())} basin polygons")
    basins["geometry"] = shapely.make_valid(basins.geometry.to_numpy())
    return basins


def analysis_grid(dataset: rasterio.io.DatasetReader) -> tuple[int, int, Affine, dict]:
    """Use one fixed CONUS grid so historical source-grid changes cannot move basins."""
    if ANALYSIS_GRID.exists():
        specification = json.loads(ANALYSIS_GRID.read_text())
        transform = Affine(*specification["transform"])
        return specification["height"], specification["width"], transform, specification
    height = int(np.ceil(dataset.height / DOWNSAMPLE))
    width = int(np.ceil(dataset.width / DOWNSAMPLE))
    transform = dataset.transform * dataset.transform.scale(
        dataset.width / width, dataset.height / height
    )
    specification = {
        "height": height, "width": width,
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "crs": "EPSG:4326", "source_native_shape": [dataset.height, dataset.width],
        "downsample_factor": DOWNSAMPLE,
        "method": "fixed geographic grid; rasterio average reprojection from each native monthly grid",
    }
    ANALYSIS_GRID.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_GRID.write_text(json.dumps(specification, indent=2) + "\n")
    return height, width, transform, specification


def read_zip(content: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if len(names) != 1:
            raise RuntimeError(f"Expected one TIFF, found {names}")
        payload = archive.read(names[0])
    with rasterio.MemoryFile(payload) as memory:
        with memory.open() as dataset:
            height, width, transform, specification = analysis_grid(dataset)
            raw = np.full((height, width), np.nan, np.float32)
            reproject(
                source=dataset.read(1), destination=raw,
                src_transform=dataset.transform, src_crs=dataset.crs,
                src_nodata=dataset.nodata, dst_transform=transform,
                dst_crs=specification["crs"], dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            lon = transform.c + (np.arange(width) + 0.5) * transform.a
            lat_descending = transform.f + (np.arange(height) + 0.5) * transform.e
            values = np.asarray(raw, np.float32)[::-1]
            latitude = lat_descending[::-1]
            metadata = {
                "native_shape": [dataset.height, dataset.width],
                "extraction_shape": [height, width],
                "native_crs": str(dataset.crs),
                "native_nodata": dataset.nodata,
                "analysis_grid": specification,
                "resampling": "average reprojection to fixed analysis grid",
                "raster_member": names[0],
            }
    values = np.where(np.isfinite(values) & (values != 9999), values * SCALE, np.nan).astype(np.float32)
    return values, latitude.astype(float), lon.astype(float), metadata


def build_weights(basins: gpd.GeoDataFrame, latitude: np.ndarray, longitude: np.ndarray, metadata: dict) -> sparse.csr_matrix:
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
            # For very small basins, use the nearest extraction-grid center.
            centroid = geometry.centroid
            ix = np.asarray([int(np.argmin(np.abs(longitude - centroid.x)))])
            iy = np.asarray([int(np.argmin(np.abs(latitude - centroid.y)))])
            overlap = np.ones(1)
        else:
            iy, ix, overlap = iy[keep], ix[keep], overlap[keep]
        area = overlap * np.cos(np.deg2rad(latitude[iy]))
        rows.append(np.full(len(ix), row, np.int32))
        cols.append((iy * nlon + ix).astype(np.int32))
        vals.append(area.astype(np.float64))
        if (row + 1) % 100 == 0 or row + 1 == len(basins):
            print(f"SSEBop weights {row + 1:,}/{len(basins):,}", flush=True)
    matrix = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(basins), nlat * nlon),
    )
    matrix = sparse.diags(1.0 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix
    weight_file, gauge_file, metadata_file = weight_paths(latitude, longitude)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(weight_file, matrix, compressed=True)
    pd.DataFrame({"row": np.arange(len(basins)), "GAGE_ID": basins.GAGE_ID}).to_csv(gauge_file, index=False)
    metadata.update({
        "method": "polygon-cell overlap on fixed monthly analysis grid after average reprojection",
        "shape": list(matrix.shape), "nonzero_weights": int(matrix.nnz),
        "latitude": [float(latitude[0]), float(latitude[-1])],
        "longitude": [float(longitude[0]), float(longitude[-1])],
        "dx_degrees": dx, "dy_degrees": dy,
    })
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return matrix.tocsr()


def grid_signature(latitude: np.ndarray, longitude: np.ndarray) -> str:
    """Identify a raster grid by dimensions, bounds, and spacing."""
    description = {
        "nlat": len(latitude), "nlon": len(longitude),
        "lat_first": round(float(latitude[0]), 9),
        "lat_last": round(float(latitude[-1]), 9),
        "lon_first": round(float(longitude[0]), 9),
        "lon_last": round(float(longitude[-1]), 9),
        "dlat": round(float(np.median(np.diff(latitude))), 9),
        "dlon": round(float(np.median(np.diff(longitude))), 9),
    }
    suffix = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:12]
    return f"{len(latitude)}x{len(longitude)}_{suffix}"


def weight_paths(latitude: np.ndarray, longitude: np.ndarray) -> tuple[Path, Path, Path]:
    """Return exact-grid paths because the historical archive uses several grids."""
    key = grid_signature(latitude, longitude)
    return (
        WEIGHT_DIR / f"ssebop_monthly_basin_grid_weights_{key}.npz",
        WEIGHT_DIR / f"ssebop_monthly_gauge_order_{key}.csv",
        WEIGHT_DIR / f"ssebop_monthly_weight_metadata_{key}.json",
    )


def load_or_build_weights(
    basins: gpd.GeoDataFrame, latitude: np.ndarray, longitude: np.ndarray, metadata: dict,
) -> sparse.csr_matrix:
    weight_file, gauge_file, _ = weight_paths(latitude, longitude)
    if weight_file.exists() and gauge_file.exists():
        order = pd.read_csv(gauge_file, dtype={"GAGE_ID": str}).GAGE_ID.str.zfill(8).tolist()
        if order != basins.GAGE_ID.tolist():
            raise RuntimeError("Saved SSEBop weights use another gauge order")
        matrix = sparse.load_npz(weight_file).tocsr()
        if matrix.shape != (len(basins), len(latitude) * len(longitude)):
            raise RuntimeError("Saved SSEBop weights use another extraction grid")
        return matrix
    return build_weights(basins, latitude, longitude, metadata)


def basin_mean(matrix: sparse.csr_matrix, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, np.float64).ravel()
    valid = np.isfinite(flat)
    denominator = np.asarray(matrix @ valid.astype(np.float64)).ravel()
    numerator = np.asarray(matrix @ np.where(valid, flat, 0.0)).ravel()
    mean = np.divide(numerator, denominator, out=np.full(matrix.shape[0], np.nan), where=denominator > 0)
    return mean.astype(np.float32), denominator.astype(np.float32)


def process_year(year: int, dates: pd.DatetimeIndex, basins: gpd.GeoDataFrame, args: argparse.Namespace) -> None:
    output = OUT / f"ssebop_monthly_{year}.npz"
    manifest_path = PROVENANCE / f"ssebop_monthly_manifest_{year}.csv"
    if output.exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if len(manifest) == len(dates) and manifest.status.eq("ok").all():
            print(f"{year}: already complete", flush=True)
            return
    eta = np.full((len(basins), len(dates)), np.nan, np.float32)
    coverage = np.full_like(eta, np.nan)
    records: list[dict] = []
    matrices: dict[str, sparse.csr_matrix] = {}
    for first in range(0, len(dates), args.batch_months):
        batch = dates[first:first + args.batch_months]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, pd.Timestamp(date)): pd.Timestamp(date) for date in batch}
            downloaded = [future.result() for future in as_completed(futures)]
        downloaded.sort(key=lambda row: row["date"])
        for record in downloaded:
            date = pd.Timestamp(record["date"])
            index = int(np.searchsorted(dates.values, date.to_datetime64()))
            if record["status"] != "downloaded":
                records.append({k: v for k, v in record.items() if k != "content"} | {"date": date.date()})
                continue
            started = time.time()
            try:
                values, latitude, longitude, metadata = read_zip(record["content"])
                signature = grid_signature(latitude, longitude)
                if signature not in matrices:
                    matrices[signature] = load_or_build_weights(basins, latitude, longitude, metadata)
                matrix = matrices[signature]
                eta[:, index], coverage[:, index] = basin_mean(matrix, values)
                clean = {k: v for k, v in record.items() if k != "content"}
                records.append({
                    **clean, "date": date.date(), "status": "ok",
                    "grid_signature": signature,
                    "extraction_seconds": time.time() - started,
                    "valid_gauges": int(np.isfinite(eta[:, index]).sum()),
                })
            except Exception as exc:
                clean = {k: v for k, v in record.items() if k != "content"}
                records.append({
                    **clean, "date": date.date(), "status": "extract_failed",
                    "error": repr(exc), "extraction_seconds": time.time() - started,
                })
        print(f"{year}: {min(first + len(batch), len(dates))}/{len(dates)} months", flush=True)
    manifest = pd.DataFrame(records).sort_values("date")
    manifest.to_csv(manifest_path, index=False)
    if len(manifest) != len(dates) or not manifest.status.eq("ok").all():
        raise RuntimeError(f"{year}: incomplete extraction; inspect {manifest_path}")
    np.savez_compressed(
        output, dates=dates.values.astype("datetime64[D]"),
        gage_ids=basins.GAGE_ID.to_numpy(str), eta_mm_month=eta,
        valid_fraction=coverage,
    )
    print(f"{year}: wrote {output}", flush=True)


def main() -> None:
    args = arguments()
    for directory in [CACHE, OUT, PROVENANCE, WEIGHT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    basins = load_basins()
    all_dates = pd.date_range(args.start, args.end, freq="MS")
    years = args.years or sorted(all_dates.year.unique())
    for year in years:
        dates = all_dates[all_dates.year == year]
        if len(dates):
            process_year(year, dates, basins, args)
    files = sorted(OUT.glob("ssebop_monthly_*.npz"))
    summary = {
        "product": "USGS monthly SSEBop actual ET version 2", "doi": "10.5066/P9L2YMV",
        "native_units": "mm month-1", "scale_to_mm_month": SCALE,
        "gauges": len(basins), "roles": ["donor", "development"],
        "annual_files": len(files), "raw_files_retained": False,
    }
    (PROVENANCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
