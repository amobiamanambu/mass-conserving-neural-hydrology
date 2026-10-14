#!/usr/bin/env python3
"""Extract April-October SSEBop ET for the 500 donor basins only."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from exactextract import exact_extract


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
ARCHIVE_DIR = ROOT / "conus_hydrologic_law_identifiability_2026-09-30/data/external/ssebop"
POLYGON_PATH = ROOT / "gridmet_data/GAGE_II_Basins.shp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    processed = PROJECT / "data/processed"
    provenance = PROJECT / "provenance"
    processed.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(processed / "station_index.csv", dtype={"GAGE_ID": str})
    donor_ids = set(stations.loc[stations.role.eq("donor"), "GAGE_ID"])
    polygons = gpd.read_file(POLYGON_PATH)[["GAGE_ID", "geometry"]].to_crs(4326)
    polygons["GAGE_ID"] = polygons.GAGE_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    polygons = polygons[polygons.GAGE_ID.isin(donor_ids)].drop_duplicates("GAGE_ID").copy()
    if set(polygons.GAGE_ID) != donor_ids:
        missing = sorted(donor_ids - set(polygons.GAGE_ID))
        raise RuntimeError(f"Donor polygons missing: {missing[:10]}")
    polygons.geometry = polygons.geometry.simplify(0.001, preserve_topology=True)

    rows, manifest = [], []
    with tempfile.TemporaryDirectory(prefix="ssebop_donor_") as temporary:
        temporary = Path(temporary)
        for year in range(2003, 2019):
            archive = ARCHIVE_DIR / f"d{year}091304.zip"
            if not archive.exists():
                raise FileNotFoundError(archive)
            started = time.time()
            with zipfile.ZipFile(archive) as source:
                members = [name for name in source.namelist() if name.lower().endswith((".tif", ".tiff"))]
                if len(members) != 1:
                    raise RuntimeError(f"Expected one raster in {archive}, found {members}")
                source.extract(members[0], temporary)
                raster = temporary / members[0]
            extracted = exact_extract(
                str(raster), polygons, ["mean", "count"], include_cols=["GAGE_ID"],
                output="pandas", strategy="raster-sequential",
            ).rename(columns={"mean": "ssebop_apr_oct_et_mm", "count": "valid_pixel_count"})
            extracted.insert(0, "year", year)
            rows.append(extracted)
            manifest.append({
                "year": year,
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "archive_bytes": archive.stat().st_size,
                "raster_member": members[0],
                "seconds": time.time() - started,
            })
            raster.unlink(missing_ok=True)
            print(f"SSEBop {year}: {len(extracted):,} donor basins", flush=True)

    result = pd.concat(rows, ignore_index=True)
    if result.GAGE_ID.nunique() != 500 or len(result) != 8000:
        raise RuntimeError(f"Unexpected SSEBop result shape: {result.shape}, basins={result.GAGE_ID.nunique()}")
    result.to_parquet(processed / "donor_ssebop_et_2003_2018.parquet", index=False)
    pd.DataFrame(manifest).to_csv(provenance / "donor_ssebop_manifest.csv", index=False)
    summary = {
        "dataset": "USGS SSEBop MODIS CONUS seasonal cumulative actual ET",
        "season": "April-October",
        "years": [2003, 2018],
        "donor_basins": int(result.GAGE_ID.nunique()),
        "rows": int(len(result)),
        "nonmissing": int(result.ssebop_apr_oct_et_mm.notna().sum()),
        "target_et_read": False,
        "output_sha256": sha256(processed / "donor_ssebop_et_2003_2018.parquet"),
    }
    (provenance / "donor_ssebop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
