#!/usr/bin/env python3
"""Rebuild GAGES-II attributes and freeze the dense zero-shot station split.

Run this script with the bundled document/spreadsheet Python because that
runtime contains openpyxl. The script reads the precise source XLSX rather than
the rounded CSV distribution. It never inspects discharge values; only the
previously audited coverage fractions are used to define eligibility.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE_XLSX = ROOT / "basinchar_and_report_sept_2011/gagesII_sept30_2011_conterm.xlsx"
SOURCE_POPULATION = ROOT / "conus_ai_law_falsification_2026-09-30/data/interim/population_audit.csv"
DESIGN_PATH = PROJECT / "config/study_design.json"

# These variables are static, available outside the target streamflow record,
# and interpretable. Streamflow-derived BFI and all FlowRec/Hydro signatures
# are deliberately excluded from the model matrix.
CONTINUOUS_SOURCES = [
    ("BasinID", "DRAIN_SQKM"),
    ("Bas_Morph", "BAS_COMPACTNESS"),
    ("Climate", "PPTAVG_BASIN"),
    ("Climate", "T_AVG_BASIN"),
    ("Climate", "T_MAXSTD_BASIN"),
    ("Climate", "T_MINSTD_BASIN"),
    ("Climate", "RH_BASIN"),
    ("Climate", "WD_BASIN"),
    ("Climate", "WDMAX_BASIN"),
    ("Climate", "WDMIN_BASIN"),
    ("Climate", "PET"),
    ("Climate", "SNOW_PCT_PRECIP"),
    ("Climate", "PRECIP_SEAS_IND"),
    ("Soils", "AWCAVE"),
    ("Soils", "PERMAVE"),
    ("Soils", "BDAVE"),
    ("Soils", "OMAVE"),
    ("Soils", "WTDEPAVE"),
    ("Soils", "ROCKDEPAVE"),
    ("Soils", "CLAYAVE"),
    ("Soils", "SILTAVE"),
    ("Soils", "SANDAVE"),
    ("Soils", "KFACT_UP"),
    ("Topo", "ELEV_MEAN_M_BASIN"),
    ("Topo", "ELEV_STD_M_BASIN"),
    ("Topo", "RRMEAN"),
    ("Topo", "SLOPE_PCT"),
    ("Topo", "ASPECT_NORTHNESS"),
    ("Topo", "ASPECT_EASTNESS"),
    ("Hydro", "STREAMS_KM_SQ_KM"),
    ("Hydro", "STRAHLER_MAX"),
    ("Hydro", "MAINSTEM_SINUOUSITY"),
    ("Hydro", "HIRES_LENTIC_PCT"),
    ("Hydro", "TOPWET"),
    ("LC06_Basin", "DEVNLCD06"),
    ("LC06_Basin", "FORESTNLCD06"),
    ("LC06_Basin", "WATERNLCD06"),
    ("LC06_Basin", "SNOWICENLCD06"),
    ("LC06_Basin", "BARRENNLCD06"),
    ("LC06_Basin", "SHRUBNLCD06"),
    ("LC06_Basin", "GRASSNLCD06"),
    ("LC06_Basin", "PASTURENLCD06"),
    ("LC06_Basin", "CROPSNLCD06"),
    ("LC06_Basin", "WOODYWETNLCD06"),
    ("LC06_Basin", "EMERGWETNLCD06"),
    ("HydroMod_Dams", "DDENS_2009"),
    ("HydroMod_Dams", "STOR_NID_2009"),
    ("HydroMod_Other", "CANALS_PCT"),
    ("HydroMod_Other", "NPDES_MAJ_DENS"),
    ("HydroMod_Other", "FRESHW_WITHDRAWAL"),
    ("HydroMod_Other", "MINING92_PCT"),
    ("HydroMod_Other", "PCT_IRRIG_AG"),
    ("Pop_Infrastr", "PDEN_2000_BLOCK"),
    ("Pop_Infrastr", "ROADS_KM_SQ_KM"),
    ("Pop_Infrastr", "IMPNLCD06"),
    ("Prot_Areas", "PADCAT1_PCT_BASIN"),
]

LOG10_SOURCES = {("BasinID", "DRAIN_SQKM")}
LOG1P_SOURCES = {
    ("Climate", "WD_BASIN"),
    ("Climate", "WDMAX_BASIN"),
    ("Climate", "WDMIN_BASIN"),
    ("Soils", "PERMAVE"),
    ("Soils", "WTDEPAVE"),
    ("Soils", "ROCKDEPAVE"),
    ("Hydro", "STREAMS_KM_SQ_KM"),
    ("HydroMod_Dams", "DDENS_2009"),
    ("HydroMod_Dams", "STOR_NID_2009"),
    ("HydroMod_Other", "NPDES_MAJ_DENS"),
    ("HydroMod_Other", "FRESHW_WITHDRAWAL"),
    ("Pop_Infrastr", "PDEN_2000_BLOCK"),
    ("Pop_Infrastr", "ROADS_KM_SQ_KM"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    valid = text.str.fullmatch(r"\d{1,15}")
    return text.str.zfill(8).where(valid)


def read_sheet(workbook: openpyxl.Workbook, sheet_name: str) -> tuple[pd.DataFrame, list[str]]:
    sheet = workbook[sheet_name]
    rows = sheet.iter_rows(values_only=True)
    raw_header = next(rows)
    keep = [i for i, value in enumerate(raw_header) if value is not None and str(value).strip()]
    header = [str(raw_header[i]).strip() for i in keep]
    if "STAID" not in header:
        return pd.DataFrame(), header
    values = ([row[i] if i < len(row) else None for i in keep] for row in rows)
    frame = pd.DataFrame(values, columns=header)
    frame["STAID"] = normalize_id(frame["STAID"])
    frame = frame[frame.STAID.notna() & frame.STAID.str.fullmatch(r"\d{8,15}", na=False)].copy()
    if frame.STAID.duplicated().any():
        # Duplicate station rows in the source are acceptable only when the
        # first nonmissing value is unambiguous for every variable.
        conflicts = []
        for station, group in frame[frame.STAID.duplicated(False)].groupby("STAID"):
            for column in frame.columns.drop("STAID"):
                unique = pd.Series(group[column]).dropna().astype(str).unique()
                if len(unique) > 1:
                    conflicts.append((station, column, unique[:3].tolist()))
        if conflicts:
            raise RuntimeError(f"Conflicting duplicate rows in {sheet_name}: {conflicts[:5]}")
        frame = frame.groupby("STAID", as_index=False).first()
    return frame, header


def allocate_quota(frame: pd.DataFrame, total: int) -> dict[tuple[str, str], int]:
    counts = frame.groupby(["tier_short", "reference_status"], observed=True).size().sort_index()
    raw = counts / counts.sum() * total
    quota = np.floor(raw).astype(int)
    remainder = total - int(quota.sum())
    order = (raw - quota).sort_values(ascending=False, kind="mergesort").index.tolist()
    for key in order[:remainder]:
        quota.loc[key] += 1
    if (quota > counts).any():
        raise RuntimeError("A sampling quota exceeds its stratum population")
    return {tuple(key): int(value) for key, value in quota.items()}


def farthest_points(indices: np.ndarray, coordinates: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=int)
    if count >= len(indices):
        return indices.copy()
    local = coordinates[indices]
    center = np.nanmedian(local, axis=0)
    selected = [int(np.argmin(np.sum((local - center) ** 2, axis=1)))]
    nearest = np.sum((local - local[selected[0]]) ** 2, axis=1)
    nearest[selected[0]] = -np.inf
    for _ in range(1, count):
        choice = int(np.argmax(nearest))
        selected.append(choice)
        distance = np.sum((local - local[choice]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[selected] = -np.inf
    return indices[np.asarray(selected, dtype=int)]


def nearest_haversine(
    target_latlon: np.ndarray, donor_latlon: np.ndarray, chunk_size: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest donor distance and index without a geospatial dependency."""
    donor_lat = np.deg2rad(donor_latlon[:, 0])[None, :]
    donor_lon = np.deg2rad(donor_latlon[:, 1])[None, :]
    distances, indices = [], []
    for first in range(0, len(target_latlon), chunk_size):
        block = np.deg2rad(target_latlon[first:first + chunk_size])
        lat = block[:, 0][:, None]
        lon = block[:, 1][:, None]
        a = (
            np.sin((lat - donor_lat) / 2.0) ** 2
            + np.cos(lat) * np.cos(donor_lat) * np.sin((lon - donor_lon) / 2.0) ** 2
        )
        angular = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        choice = np.argmin(angular, axis=1)
        distances.append(angular[np.arange(len(block)), choice] * 6371.0088)
        indices.append(choice)
    return np.concatenate(distances), np.concatenate(indices)


def build_feature_matrix(attributes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = pd.DataFrame({"GAGE_ID": attributes.STAID})
    catalog = []
    for sheet, column in CONTINUOUS_SOURCES:
        source_name = f"{sheet}__{column}"
        values = pd.to_numeric(attributes[source_name], errors="coerce")
        values = values.mask(values <= -99.0)
        if (sheet, column) in LOG10_SOURCES:
            name = f"log10_{column}"
            values = np.log10(values.where(values > 0))
            transform = "log10"
        elif (sheet, column) in LOG1P_SOURCES:
            name = f"log1p_{column}"
            values = np.log1p(values.clip(lower=0))
            transform = "log1p"
        else:
            name = column
            transform = "identity"
        if name in out.columns:
            name = f"{sheet}_{name}"
        out[name] = values
        catalog.append({"model_feature": name, "source_sheet": sheet, "source_column": column, "transform": transform})

    geology = attributes["Geology__GEOL_HUNT_DOM_CODE"].fillna("UNKNOWN").astype(str).str.strip()
    geology = geology.mask(geology.isin(["", "-999", "nan", "None"]), "UNKNOWN")
    for level in sorted(geology.unique()):
        safe = "".join(character if character.isalnum() else "_" for character in level).strip("_") or "UNKNOWN"
        name = f"geology__{safe}"
        out[name] = geology.eq(level).astype(float)
        catalog.append({"model_feature": name, "source_sheet": "Geology", "source_column": "GEOL_HUNT_DOM_CODE", "transform": f"one_hot:{level}"})
    return out, catalog


def main() -> None:
    design = json.loads(DESIGN_PATH.read_text())
    interim = PROJECT / "data/interim"
    frozen = PROJECT / "data/frozen"
    provenance = PROJECT / "provenance"
    for directory in [interim, frozen, provenance]:
        directory.mkdir(parents=True, exist_ok=True)

    workbook = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    sheet_frames: dict[str, pd.DataFrame] = {}
    sheet_audit = []
    all_catalog = []
    full = None
    for sheet_name in workbook.sheetnames:
        frame, header = read_sheet(workbook, sheet_name)
        if frame.empty:
            sheet_audit.append({"sheet": sheet_name, "rows": 0, "columns": len(header), "joined": False})
            continue
        sheet_frames[sheet_name] = frame
        renamed = frame.rename(columns={column: f"{sheet_name}__{column}" for column in frame.columns if column != "STAID"})
        for column in frame.columns.drop("STAID"):
            all_catalog.append({"sheet": sheet_name, "column": column, "joined_name": f"{sheet_name}__{column}"})
        full = renamed if full is None else full.merge(renamed, on="STAID", how="outer", validate="one_to_one")
        sheet_audit.append({"sheet": sheet_name, "rows": len(frame), "columns": len(frame.columns), "joined": True})
        print(f"read {sheet_name}: {len(frame):,} stations x {len(frame.columns):,} columns", flush=True)
    workbook.close()
    if full is None:
        raise RuntimeError("No GAGES-II sheets were read")

    master_path = interim / "gagesii_all_attributes.csv.gz"
    with gzip.open(master_path, "wt", compresslevel=6, encoding="utf-8", newline="") as stream:
        full.to_csv(stream, index=False)
    pd.DataFrame(all_catalog).to_csv(interim / "gagesii_attribute_catalog.csv", index=False)
    pd.DataFrame(sheet_audit).to_csv(provenance / "gagesii_sheet_audit.csv", index=False)

    features, model_catalog = build_feature_matrix(full)
    id_fields = full[[
        "STAID", "BasinID__STANAME", "BasinID__LAT_GAGE", "BasinID__LNG_GAGE",
        "BasinID__STATE", "BasinID__HUC02", "Bas_Classif__CLASS",
        "Bas_Classif__AGGECOREGION", "Geology__GEOL_HUNT_DOM_CODE",
    ]].rename(columns={
        "STAID": "GAGE_ID",
        "BasinID__STANAME": "station_name",
        "BasinID__LAT_GAGE": "latitude",
        "BasinID__LNG_GAGE": "longitude",
        "BasinID__STATE": "state",
        "BasinID__HUC02": "HUC02",
        "Bas_Classif__CLASS": "reference_class",
        "Bas_Classif__AGGECOREGION": "ecoregion",
        "Geology__GEOL_HUNT_DOM_CODE": "geology_code",
    })
    attributes = id_fields.merge(features, on="GAGE_ID", how="left", validate="one_to_one")

    source = pd.read_csv(SOURCE_POPULATION, dtype={"GAGE_ID": str}, low_memory=False)
    source["GAGE_ID"] = normalize_id(source.GAGE_ID)
    source = source.drop_duplicates("GAGE_ID")
    base_columns = [
        "GAGE_ID", "basin_id", "quality_tier", "snow_fraction", "area_km2",
        "daily_file_exists", "daily_rows_in_window", "q_valid_days",
        "forcing_valid_days", "q_coverage", "forcing_coverage",
    ]
    population = source[base_columns].merge(attributes, on="GAGE_ID", how="left", validate="one_to_one", indicator=True)
    if not population._merge.eq("both").all():
        missing = population.loc[population._merge.ne("both"), "GAGE_ID"].tolist()
        raise RuntimeError(f"Accepted stations missing from precise GAGES-II workbook: {missing[:10]}")
    population = population.drop(columns="_merge")
    population["tier_short"] = np.where(population.quality_tier.str.contains("TIER1"), "Tier 1", "Tier 2")
    population["reference_status"] = np.where(population.reference_class.astype(str).str.lower().eq("ref"), "Reference", "Non-reference")
    population["eligible"] = (
        population.daily_file_exists.fillna(False).astype(bool)
        & population.q_coverage.ge(float(design["q_coverage_min"]))
        & population.forcing_coverage.ge(float(design["forcing_coverage_min"]))
        & population.ecoregion.notna()
        & population.latitude.notna()
        & population.longitude.notna()
    )
    eligible = population[population.eligible].sort_values("GAGE_ID").reset_index(drop=True)
    expected_total = sum(item["eligible"] for item in design["ecoregion_allocations"].values())
    if len(eligible) != expected_total:
        raise RuntimeError(f"Expected {expected_total} eligible gauges, found {len(eligible)}")

    feature_names = [item["model_feature"] for item in model_catalog]
    feature_coverage = eligible[feature_names].notna().mean()
    active_features = [
        name for name in feature_names
        if feature_coverage[name] >= 0.95 and eligible[name].nunique(dropna=True) > 1
    ]
    excluded_features = sorted(set(feature_names) - set(active_features))
    if not active_features:
        raise RuntimeError("No model features survived the coverage and variance audit")
    feature_audit = pd.DataFrame(model_catalog)
    feature_audit["eligible_nonmissing_fraction"] = feature_audit.model_feature.map(feature_coverage)
    feature_audit["active"] = feature_audit.model_feature.isin(active_features)
    feature_audit["exclusion_reason"] = np.where(
        feature_audit.active,
        "",
        np.where(feature_audit.eligible_nonmissing_fraction.lt(0.95), "coverage_below_0.95", "zero_variance"),
    )
    feature_audit.to_csv(frozen / "model_feature_catalog.csv", index=False)

    matrix = eligible[active_features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    medians = np.nanmedian(matrix, axis=0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    lower, upper = np.nanpercentile(matrix, [1, 99], axis=0)
    matrix = np.clip(matrix, lower, upper)
    q25, q75 = np.nanpercentile(matrix, [25, 75], axis=0)
    scales = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    scaled = np.clip((matrix - medians) / scales, -8, 8)
    centered = scaled - scaled.mean(axis=0, keepdims=True)
    _u, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    maximum_components = min(15, scaled.shape[1])
    coordinates_full = centered @ right_vectors[:maximum_components].T
    explained_ratio = singular_values[:maximum_components] ** 2 / np.sum(singular_values ** 2)
    cumulative = np.cumsum(explained_ratio)
    dimensions = min(len(cumulative), int(np.searchsorted(cumulative, 0.90) + 1))
    coordinates = coordinates_full[:, :dimensions]

    roles = np.full(len(eligible), "target", dtype=object)
    selection_rows = []
    for ecoregion, allocation in design["ecoregion_allocations"].items():
        eco_positions = np.flatnonzero(eligible.ecoregion.eq(ecoregion).to_numpy())
        if len(eco_positions) != allocation["eligible"]:
            raise RuntimeError(f"{ecoregion}: expected {allocation['eligible']}, found {len(eco_positions)}")
        eco = eligible.iloc[eco_positions]
        donor_quota = allocate_quota(eco, allocation["donor"])
        development_quota = allocate_quota(eco, allocation["development"])
        for key in sorted(donor_quota):
            tier, reference = key
            stratum = eco_positions[
                eligible.iloc[eco_positions].tier_short.eq(tier).to_numpy()
                & eligible.iloc[eco_positions].reference_status.eq(reference).to_numpy()
            ]
            donor = farthest_points(stratum, coordinates, donor_quota[key])
            roles[donor] = "donor"
            remaining = stratum[~np.isin(stratum, donor)]
            development = farthest_points(remaining, coordinates, development_quota.get(key, 0))
            roles[development] = "development"
            selection_rows.append({
                "ecoregion": ecoregion,
                "tier": tier,
                "reference_status": reference,
                "eligible": len(stratum),
                "donor": len(donor),
                "development": len(development),
                "target": len(stratum) - len(donor) - len(development),
            })

    eligible["role"] = roles
    counts = eligible.role.value_counts().to_dict()
    expected_counts = {"donor": 500, "development": 200, "target": 3982}
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {counts}, expected {expected_counts}")

    # A transparent spatial sensitivity flag; it does not remove any target.
    donor = eligible[eligible.role.eq("donor")]
    target = eligible[eligible.role.eq("target")]
    nearest_km, nearest_index = nearest_haversine(
        target[["latitude", "longitude"]].to_numpy(float),
        donor[["latitude", "longitude"]].to_numpy(float),
    )
    nearest_id = donor.GAGE_ID.to_numpy()[nearest_index]
    eligible["nearest_donor_km"] = np.nan
    eligible["nearest_donor_id"] = ""
    target_positions = eligible.index[eligible.role.eq("target")]
    eligible.loc[target_positions, "nearest_donor_km"] = nearest_km
    eligible.loc[target_positions, "nearest_donor_id"] = nearest_id
    eligible["distance_independent_50km"] = eligible.nearest_donor_km.gt(50.0)

    split_columns = [
        "GAGE_ID", "station_name", "state", "latitude", "longitude", "HUC02",
        "ecoregion", "tier_short", "quality_tier", "reference_status",
        "reference_class", "geology_code", "basin_id", "snow_fraction",
        "area_km2", "q_coverage", "forcing_coverage", "role",
        "nearest_donor_id", "nearest_donor_km", "distance_independent_50km",
    ] + active_features
    split_path = frozen / "station_split.csv"
    eligible[split_columns].to_csv(split_path, index=False)
    pd.DataFrame(selection_rows).to_csv(frozen / "split_stratum_counts.csv", index=False)

    allocation_audit = (
        eligible.groupby(["ecoregion", "tier_short", "reference_status", "role"], observed=True)
        .size().unstack(fill_value=0).reset_index()
    )
    allocation_audit.to_csv(provenance / "split_allocation_audit.csv", index=False)
    summary = {
        "title": design["title"],
        "source_xlsx": str(SOURCE_XLSX),
        "source_xlsx_sha256": sha256(SOURCE_XLSX),
        "source_population": str(SOURCE_POPULATION),
        "source_population_sha256": sha256(SOURCE_POPULATION),
        "accepted_population": int(len(population)),
        "eligible_population": int(len(eligible)),
        "role_counts": {key: int(value) for key, value in counts.items()},
        "tier_counts": {key: int(value) for key, value in eligible.tier_short.value_counts().items()},
        "ecoregion_counts": {key: int(value) for key, value in eligible.ecoregion.value_counts().sort_index().items()},
        "reference_counts": {key: int(value) for key, value in eligible.reference_status.value_counts().items()},
        "candidate_model_features": len(feature_names),
        "active_model_features": len(active_features),
        "active_feature_names": active_features,
        "excluded_features": excluded_features,
        "selection_pca_dimensions": dimensions,
        "selection_pca_variance_fraction": float(cumulative[dimensions - 1]),
        "target_distance_independent_50km": int(eligible.distance_independent_50km.sum()),
        "station_split_sha256": sha256(split_path),
        "all_attribute_file": str(master_path),
        "all_attribute_file_sha256": sha256(master_path),
        "discharge_values_read": False,
        "bfi_used_as_predictor": False,
    }
    (provenance / "population_and_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
