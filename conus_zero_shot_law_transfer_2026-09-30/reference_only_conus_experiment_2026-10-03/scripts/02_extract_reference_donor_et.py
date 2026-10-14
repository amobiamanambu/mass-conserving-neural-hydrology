#!/usr/bin/env python3
"""Extract April-October SSEBop ET for reference-only donor basins.

The extractor computes fractional pixel coverage directly from the GeoTIFF
grid and basin polygons. Its output is checked against the archived
exactextract product wherever both products are available.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
ARCHIVE_DIR = ROOT / "conus_hydrologic_law_identifiability_2026-09-30/data/external/ssebop"
POLYGON_PATH = ROOT / "gridmet_data/GAGE_II_Basins.shp"
ARCHIVED_EXACT = ROOT / "conus_hydrologic_law_identifiability_2026-09-30/data/interim/ssebop_apr_oct_et_2003_2020.parquet"


package_dir = PROJECT / ".python_packages"
if package_dir.exists():
    sys.path.insert(0, str(package_dir))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from PIL import Image  # noqa: E402


Image.MAX_IMAGE_PIXELS = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def grid_definition(image: Image.Image) -> tuple[int, int, float, float, float, float]:
    width, height = image.size
    scale = image.tag_v2.get(33550)
    tie = image.tag_v2.get(33922)
    if scale is None or tie is None:
        raise RuntimeError("SSEBop GeoTIFF lacks pixel scale or tie point")
    dx, dy = float(scale[0]), float(scale[1])
    x0, y0 = float(tie[3]), float(tie[4])
    return width, height, x0, y0, dx, dy


def polygon_weights(geometry, grid) -> tuple[np.ndarray, np.ndarray]:
    width, height, x0, y0, dx, dy = grid
    xmin, ymin, xmax, ymax = geometry.bounds
    c0 = max(0, int(np.floor((xmin - x0) / dx)))
    c1 = min(width, int(np.ceil((xmax - x0) / dx)))
    r0 = max(0, int(np.floor((y0 - ymax) / dy)))
    r1 = min(height, int(np.ceil((y0 - ymin) / dy)))
    if c1 <= c0 or r1 <= r0:
        raise RuntimeError("Basin lies outside the SSEBop raster")
    cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    cols = cols.ravel()
    rows = rows.ravel()
    left = x0 + cols * dx
    top = y0 - rows * dy
    pixels = shapely.box(left, top - dy, left + dx, top)
    overlap = shapely.area(shapely.intersection(pixels, geometry))
    keep = overlap > 0.0
    if not np.any(keep):
        point = geometry.representative_point()
        col = int(np.clip(np.floor((point.x - x0) / dx), 0, width - 1))
        row = int(np.clip(np.floor((y0 - point.y) / dy), 0, height - 1))
        return np.asarray([row * width + col], dtype=np.int64), np.asarray([1.0])
    return (rows[keep] * width + cols[keep]).astype(np.int64), overlap[keep].astype(np.float64)


def main() -> None:
    data_dir = EXPERIMENT / "data"
    provenance = EXPERIMENT / "provenance"
    stations = pd.read_csv(data_dir / "station_index.csv", dtype={"GAGE_ID": str})
    donors = stations.loc[stations.role.eq("donor"), ["GAGE_ID", "tier_short"]].copy()
    donor_ids = set(donors.GAGE_ID)
    polygons = gpd.read_file(POLYGON_PATH)[["GAGE_ID", "geometry"]].to_crs(4326)
    polygons["GAGE_ID"] = normalize_id(polygons.GAGE_ID)
    polygons = polygons.loc[polygons.GAGE_ID.isin(donor_ids)].drop_duplicates("GAGE_ID")
    polygons = polygons.set_index("GAGE_ID").reindex(donors.GAGE_ID).reset_index()
    if polygons.geometry.isna().any():
        raise RuntimeError("At least one reference donor polygon is missing")
    polygons.geometry = polygons.geometry.simplify(0.001, preserve_topology=True)

    rows: list[dict] = []
    manifest: list[dict] = []
    weight_cache: dict[
        tuple[int, int, float, float, float, float],
        tuple[list[np.ndarray], list[np.ndarray]],
    ] = {}
    with tempfile.TemporaryDirectory(prefix="reference_ssebop_") as temporary_name:
        temporary = Path(temporary_name)
        for year in range(2003, 2019):
            archive = ARCHIVE_DIR / f"d{year}091304.zip"
            started = time.time()
            with zipfile.ZipFile(archive) as source:
                members = [name for name in source.namelist() if name.lower().endswith((".tif", ".tiff"))]
                if len(members) != 1:
                    raise RuntimeError(f"Expected one TIFF in {archive}")
                source.extract(members[0], temporary)
                raster_path = temporary / members[0]
            with Image.open(raster_path) as image:
                grid = grid_definition(image)
                if grid not in weight_cache:
                    pixel_indices: list[np.ndarray] = []
                    pixel_weights: list[np.ndarray] = []
                    for count, geometry in enumerate(polygons.geometry, start=1):
                        index, weight = polygon_weights(geometry, grid)
                        pixel_indices.append(index)
                        pixel_weights.append(weight)
                        if count % 100 == 0 or count == len(polygons):
                            print(f"fractional pixel weights {count}/{len(polygons)}", flush=True)
                    weight_cache[grid] = (pixel_indices, pixel_weights)
                values = np.asarray(image, dtype=np.float64).reshape(-1)
            pixel_indices, pixel_weights = weight_cache[grid]
            for gage_id, index, weight in zip(donors.GAGE_ID, pixel_indices, pixel_weights):
                local = values[index]
                valid = np.isfinite(local) & (local != 9999) & (local >= 0)
                estimate = float(np.average(local[valid], weights=weight[valid])) if valid.any() else np.nan
                rows.append(
                    {
                        "year": year,
                        "GAGE_ID": gage_id,
                        "ssebop_apr_oct_et_mm": estimate,
                        "valid_pixel_count": int(valid.sum()),
                        "fractional_pixel_count": int(len(index)),
                    }
                )
            manifest.append(
                {
                    "year": year,
                    "archive": str(archive),
                    "archive_sha256": sha256(archive),
                    "seconds": time.time() - started,
                }
            )
            raster_path.unlink(missing_ok=True)
            print(f"SSEBop {year}: {len(donors)} reference donors", flush=True)

    result = pd.DataFrame(rows)
    if len(result) != 6400 or result.GAGE_ID.nunique() != 400:
        raise RuntimeError(f"Unexpected SSEBop output shape {result.shape}")
    result["et_imputed_within_basin"] = result.ssebop_apr_oct_et_mm.isna()
    missing_before = int(result.et_imputed_within_basin.sum())
    if missing_before:
        for _, index in result.groupby("GAGE_ID").groups.items():
            ordered = result.loc[index].sort_values("year")
            filled = ordered.ssebop_apr_oct_et_mm.interpolate(
                method="linear", limit_direction="both"
            )
            result.loc[ordered.index, "ssebop_apr_oct_et_mm"] = filled.to_numpy()
    if result.ssebop_apr_oct_et_mm.isna().any():
        raise RuntimeError("A donor has no valid SSEBop year from which to interpolate")
    output = data_dir / "donor_ssebop_et_2003_2018.csv.gz"
    result.to_csv(output, index=False, compression="gzip")
    pd.DataFrame(manifest).to_csv(provenance / "ssebop_manifest.csv", index=False)

    exact = pd.read_parquet(ARCHIVED_EXACT)
    exact.GAGE_ID = normalize_id(exact.GAGE_ID)
    comparison = result.merge(
        exact[["year", "GAGE_ID", "ssebop_apr_oct_et_mm"]],
        on=["year", "GAGE_ID"],
        suffixes=("_fractional", "_exactextract"),
    )
    difference = comparison.ssebop_apr_oct_et_mm_fractional - comparison.ssebop_apr_oct_et_mm_exactextract
    validation = {
        "overlapping_basin_years": int(len(comparison)),
        "overlapping_basins": int(comparison.GAGE_ID.nunique()),
        "pearson_r": float(comparison.ssebop_apr_oct_et_mm_fractional.corr(comparison.ssebop_apr_oct_et_mm_exactextract)),
        "median_absolute_error_mm": float(np.median(np.abs(difference))),
        "p95_absolute_error_mm": float(np.quantile(np.abs(difference), 0.95)),
        "median_signed_error_mm": float(np.median(difference)),
    }
    comparison.to_csv(provenance / "ssebop_exactextract_validation.csv.gz", index=False, compression="gzip")
    summary = {
        "dataset": "USGS SSEBop MODIS CONUS April-October cumulative actual ET",
        "years": [2003, 2018],
        "reference_donors": 400,
        "tier_counts": donors.tier_short.value_counts().to_dict(),
        "nonreference_basins_extracted": 0,
        "target_et_accessed": False,
        "donor_years_imputed_within_basin": missing_before,
        "extraction_validation": validation,
        "output_sha256": sha256(output),
    }
    (provenance / "ssebop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
