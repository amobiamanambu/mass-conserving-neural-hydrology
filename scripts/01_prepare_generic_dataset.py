#!/usr/bin/env python3
"""Build the blind national NeuralHydrology GenericDataset from frozen arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = ROOT / "data" / "source_inputs"
PROCESSED = SOURCE_INPUTS / "processed"
FROZEN = SOURCE_INPUTS / "frozen_split"
DATA = ROOT / "data" / "generic_blind"
LISTS = ROOT / "data" / "basin_lists"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n")


def main() -> None:
    manifest_path = ROOT / "provenance" / "generic_dataset_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    dates = np.load(PROCESSED / "dates.npy", mmap_mode="r")
    forcing = np.load(PROCESSED / "forcing.npy", mmap_mode="r")
    discharge = np.load(PROCESSED / "q_donor_development.npy", mmap_mode="r")
    stations = pd.read_csv(PROCESSED / "station_index.csv", dtype={"GAGE_ID": str})
    split = pd.read_csv(FROZEN / "station_split.csv", dtype={"GAGE_ID": str}, low_memory=False)
    role_order = pd.Categorical(split.role, ["donor", "development", "target"], ordered=True)
    split = (
        split.assign(_role_order=role_order)
        .sort_values(["_role_order", "GAGE_ID"])
        .drop(columns="_role_order")
        .reset_index(drop=True)
    )
    feature_names = pd.read_csv(PROCESSED / "static_normalization.csv")["feature"].tolist()

    expected = {"donor": 500, "development": 200, "target": 3982}
    counts = stations.role.value_counts().to_dict()
    if counts != expected:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    if forcing.shape != (4682, len(dates), 5):
        raise RuntimeError(f"Unexpected forcing shape: {forcing.shape}")
    if discharge.shape != (700, len(dates)):
        raise RuntimeError(f"Unexpected donor/development discharge shape: {discharge.shape}")
    if stations.GAGE_ID.tolist() != split.GAGE_ID.tolist():
        raise RuntimeError("Frozen split and processed station order differ")

    attributes = split.set_index("GAGE_ID")[feature_names].astype(float).copy()
    donor_ids = stations.loc[stations.role.eq("donor"), "GAGE_ID"]
    donor_medians = attributes.loc[donor_ids].replace([np.inf, -np.inf], np.nan).median()
    attributes = attributes.replace([np.inf, -np.inf], np.nan).fillna(donor_medians)
    if not np.isfinite(attributes.to_numpy()).all():
        raise RuntimeError("Static attributes remain non-finite after donor-median filling")
    (DATA / "attributes").mkdir(parents=True, exist_ok=True)
    (DATA / "time_series").mkdir(parents=True, exist_ok=True)
    LISTS.mkdir(parents=True, exist_ok=True)
    attributes.to_csv(DATA / "attributes" / "attributes.csv", index_label="basin")

    roles = {
        role: stations.loc[stations.role.eq(role), "GAGE_ID"].tolist()
        for role in expected
    }
    write_list(LISTS / "train_500.txt", roles["donor"])
    write_list(LISTS / "validation_200.txt", roles["development"])
    write_list(LISTS / "test_3982.txt", roles["target"])
    write_list(LISTS / "smoke_validation_4.txt", roles["development"][:4])
    write_list(LISTS / "smoke_test_4.txt", roles["target"][:4])

    target_mask = (
        (dates >= np.datetime64("2019-01-01"))
        & (dates <= np.datetime64("2024-12-31"))
    )
    variable_names = ["pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy"]
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in variable_names + ["q_mm_day"]
    }
    for count, station in enumerate(stations.itertuples(index=False), start=1):
        row = int(station.row)
        q = np.full(len(dates), np.nan, dtype=np.float32)
        if station.role in {"donor", "development"}:
            q[:] = np.asarray(discharge[int(station.q_row)], dtype=np.float32)
        else:
            q[target_mask] = 0.0
        dataset = xr.Dataset(
            data_vars={
                **{
                    name: ("date", np.asarray(forcing[row, :, index], dtype=np.float32))
                    for index, name in enumerate(variable_names)
                },
                "q_mm_day": ("date", q),
            },
            coords={"date": np.asarray(dates, dtype="datetime64[ns]")},
            attrs={
                "basin_id": station.GAGE_ID,
                "role": station.role,
                "target_discharge_blinded": int(station.role == "target"),
            },
        )
        destination = DATA / "time_series" / f"{station.GAGE_ID}.nc"
        temporary = destination.with_suffix(".nc.tmp")
        dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        temporary.replace(destination)
        if count % 250 == 0 or count == len(stations):
            print(f"generic files {count:,}/{len(stations):,}", flush=True)

    manifest = {
        "created_by": str(Path(__file__).resolve()),
        "source_directory": str(PROCESSED.resolve()),
        "split_counts": counts,
        "dates": [str(dates[0]), str(dates[-1])],
        "forcing_shape": list(forcing.shape),
        "target_discharge_loaded": False,
        "target_framework_value_2019_2024": "dummy zero used only for sample construction",
        "static_feature_count": len(feature_names),
        "source_sha256": {
            name: sha256(PROCESSED / name)
            for name in [
                "dates.npy", "forcing.npy", "q_donor_development.npy",
                "station_index.csv", "static_normalization.csv",
            ]
        },
        "station_split_sha256": sha256(FROZEN / "station_split.csv"),
        "explicitly_not_read": "data/evaluation/q_target_2019_2024.npy",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
