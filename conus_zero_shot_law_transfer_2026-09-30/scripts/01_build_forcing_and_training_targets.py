#!/usr/bin/env python3
"""Build memory-mapped daily forcing and donor/development discharge arrays.

Target CSV files are opened with forcing-only columns. No target discharge is
parsed or stored by this script, which makes the zero-shot firewall auditable.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DAILY_DIR = ROOT / "continental_run/07_matched_daily/daily"
START = pd.Timestamp("2003-10-05")
END = pd.Timestamp("2024-12-31")
DATES = pd.date_range(START, END, freq="D")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_daily(job: tuple[int, int, str, str]) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray | None]:
    row, q_row, gage_id, role = job
    include_q = role in {"donor", "development"}
    columns = ["date", "pr_mm", "pet_mm", "tmean_c"] + (["q_mm_day"] if include_q else [])
    frame = pd.read_csv(
        DAILY_DIR / f"{gage_id}.csv.gz",
        usecols=columns,
        parse_dates=["date"],
    ).drop_duplicates("date").set_index("date").sort_index().reindex(DATES)
    forcing = frame[["pr_mm", "pet_mm", "tmean_c"]].apply(pd.to_numeric, errors="coerce")
    originally_valid = forcing.notna().all(axis=1).to_numpy(bool)
    forcing["tmean_c"] = forcing.tmean_c.interpolate(limit=14, limit_direction="both")
    forcing["pet_mm"] = forcing.pet_mm.interpolate(limit=7, limit_direction="both")
    forcing["pr_mm"] = forcing.pr_mm.fillna(0.0)
    forcing = forcing.fillna({"pet_mm": 0.0, "tmean_c": 0.0})
    values = forcing.to_numpy(np.float32)
    values[:, :2] = np.maximum(values[:, :2], 0.0)
    q = None
    if include_q:
        q = pd.to_numeric(frame.q_mm_day, errors="coerce").to_numpy(np.float32)
    return row, q_row, values, originally_valid, q


def main() -> None:
    frozen = PROJECT / "data/frozen"
    processed = PROJECT / "data/processed"
    provenance = PROJECT / "provenance"
    processed.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(frozen / "station_split.csv", dtype={"GAGE_ID": str}, low_memory=False)
    role_order = pd.Categorical(stations.role, ["donor", "development", "target"], ordered=True)
    stations = stations.assign(_role_order=role_order).sort_values(["_role_order", "GAGE_ID"]).drop(columns="_role_order").reset_index(drop=True)
    stations.insert(0, "row", np.arange(len(stations), dtype=int))
    stations["q_row"] = -1
    observed = stations.role.isin(["donor", "development"])
    stations.loc[observed, "q_row"] = np.arange(observed.sum(), dtype=int)
    stations[[
        "row", "q_row", "GAGE_ID", "role", "ecoregion", "tier_short",
        "reference_status", "latitude", "longitude", "area_km2",
        "nearest_donor_id", "nearest_donor_km", "distance_independent_50km",
    ]].to_csv(processed / "station_index.csv", index=False)

    n, t = len(stations), len(DATES)
    forcing = np.lib.format.open_memmap(processed / "forcing.npy", mode="w+", dtype=np.float32, shape=(n, t, 5))
    forcing_valid = np.lib.format.open_memmap(processed / "forcing_originally_valid.npy", mode="w+", dtype=np.bool_, shape=(n, t))
    q_train_dev = np.lib.format.open_memmap(processed / "q_donor_development.npy", mode="w+", dtype=np.float32, shape=(int(observed.sum()), t))
    q_train_dev[:] = np.nan

    jobs = [
        (int(row.row), int(row.q_row), str(row.GAGE_ID), str(row.role))
        for row in stations.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(read_daily, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            row, q_row, values, valid, q = future.result()
            forcing[row, :, :3] = values
            forcing_valid[row] = valid
            if q_row >= 0:
                if q is None:
                    raise RuntimeError("Donor/development job returned no discharge")
                q_train_dev[q_row] = q
            elif q is not None:
                raise RuntimeError("Target discharge entered the pre-prediction data product")
            if count % 250 == 0 or count == len(futures):
                print(f"daily files {count:,}/{len(futures):,}", flush=True)

    day = DATES.dayofyear.to_numpy(float)
    forcing[:, :, 3] = np.sin(2 * np.pi * day / 365.25)[None, :]
    forcing[:, :, 4] = np.cos(2 * np.pi * day / 365.25)[None, :]
    forcing.flush(); forcing_valid.flush(); q_train_dev.flush()
    np.save(processed / "dates.npy", DATES.to_numpy(dtype="datetime64[D]"))

    feature_catalog = pd.read_csv(frozen / "model_feature_catalog.csv")
    feature_names = feature_catalog.loc[feature_catalog.active.astype(bool), "model_feature"].tolist()
    raw_static = stations[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    donor_mask = stations.role.eq("donor").to_numpy()
    medians = np.nanmedian(raw_static[donor_mask], axis=0)
    if not np.isfinite(medians).all():
        bad = np.asarray(feature_names)[~np.isfinite(medians)].tolist()
        raise RuntimeError(f"Donors have all-missing static features: {bad}")
    filled = np.where(np.isfinite(raw_static), raw_static, medians[None, :])
    q25, q75 = np.nanpercentile(filled[donor_mask], [25, 75], axis=0)
    scales = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    one_hot = np.asarray([name.startswith("geology__") for name in feature_names])
    static = np.clip((filled - medians[None, :]) / scales[None, :], -8, 8).astype(np.float32)
    static[:, one_hot] = filled[:, one_hot]
    np.save(processed / "static.npy", static)
    pd.DataFrame({
        "feature": feature_names,
        "donor_median": medians,
        "donor_iqr": scales,
        "one_hot_unscaled": one_hot,
        "donor_missing_fraction": np.mean(~np.isfinite(raw_static[donor_mask]), axis=0),
        "development_missing_fraction": np.mean(~np.isfinite(raw_static[stations.role.eq('development')]), axis=0),
        "target_missing_fraction": np.mean(~np.isfinite(raw_static[stations.role.eq('target')]), axis=0),
    }).to_csv(processed / "static_normalization.csv", index=False)

    dates = DATES.to_numpy(dtype="datetime64[D]")
    periods = {
        "donor_fit": (dates >= np.datetime64("2003-10-05")) & (dates <= np.datetime64("2014-12-31")),
        "development": (dates >= np.datetime64("2015-01-01")) & (dates <= np.datetime64("2018-12-31")),
        "refit": (dates >= np.datetime64("2003-10-05")) & (dates <= np.datetime64("2018-12-31")),
        "target": (dates >= np.datetime64("2019-01-01")) & (dates <= np.datetime64("2024-12-31")),
    }
    coverage_rows = []
    for role in ["donor", "development", "target"]:
        rows = stations.role.eq(role).to_numpy()
        for period, time in periods.items():
            record = {
                "role": role,
                "period": period,
                "gauges": int(rows.sum()),
                "days": int(time.sum()),
                "forcing_valid_fraction": float(np.asarray(forcing_valid)[np.ix_(rows, time)].mean()),
                "q_finite_fraction": np.nan,
            }
            if role != "target":
                q_rows = stations.loc[rows, "q_row"].to_numpy(int)
                record["q_finite_fraction"] = float(np.isfinite(np.asarray(q_train_dev)[np.ix_(q_rows, time)]).mean())
            coverage_rows.append(record)
    pd.DataFrame(coverage_rows).to_csv(provenance / "daily_data_coverage.csv", index=False)

    summary = {
        "gauges": n,
        "donors": int(stations.role.eq("donor").sum()),
        "development": int(stations.role.eq("development").sum()),
        "targets": int(stations.role.eq("target").sum()),
        "days": t,
        "start": str(START.date()),
        "end": str(END.date()),
        "forcing_shape": list(forcing.shape),
        "q_donor_development_shape": list(q_train_dev.shape),
        "target_discharge_parsed": False,
        "static_features": len(feature_names),
        "forcing_features": ["pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy"],
        "station_split_sha256": sha256(frozen / "station_split.csv"),
        "station_index_sha256": sha256(processed / "station_index.csv"),
        "forcing_sha256": sha256(processed / "forcing.npy"),
        "q_donor_development_sha256": sha256(processed / "q_donor_development.npy"),
        "static_sha256": sha256(processed / "static.npy"),
    }
    (provenance / "model_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
