#!/usr/bin/env python3
"""Create an outcome-blind, reference-gauge-only CONUS experiment.

All 913 eligible GAGES-II reference gauges in Tier 1 or Tier 2 are resplit.
Only donor and development discharge is opened in this stage. Target discharge
is deliberately left unopened until all model predictions and counterfactual
fingerprints have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DAILY = ROOT / "continental_run/07_matched_daily/daily"
SEED = 314159
N_DONOR = 400
N_DEVELOPMENT = 100
N_TARGET = 413


def add_local_packages() -> None:
    package_dir = PROJECT / ".python_packages"
    if package_dir.exists():
        sys.path.insert(0, str(package_dir))


add_local_packages()
import pandas as pd  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def largest_remainder(counts: pd.Series, total: int) -> dict[str, int]:
    """Allocate a fixed total proportionally, retaining every stratum."""
    labels = counts.index.astype(str).tolist()
    values = counts.to_numpy(int)
    if total < len(values):
        raise ValueError("Requested total is smaller than the number of strata")
    if np.any(values < 1):
        raise ValueError("Empty stratum")
    allocation = np.ones(len(values), dtype=int)
    capacity = values - allocation
    remaining = total - int(allocation.sum())
    ideal = remaining * values / values.sum()
    add = np.minimum(np.floor(ideal).astype(int), capacity)
    allocation += add
    remaining = total - int(allocation.sum())
    remainder = ideal - np.floor(ideal)
    while remaining:
        eligible = np.flatnonzero(allocation < values)
        if not len(eligible):
            raise RuntimeError("Allocation exhausted available gauges")
        order = eligible[np.lexsort((np.asarray(labels)[eligible], -remainder[eligible]))]
        for index in order:
            if remaining == 0:
                break
            if allocation[index] < values[index]:
                allocation[index] += 1
                remaining -= 1
    return dict(zip(labels, allocation.tolist()))


def robust_attribute_space(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    median = np.nanmedian(raw, axis=0)
    raw = np.where(np.isfinite(raw), raw, median[None, :])
    q25, q75 = np.nanpercentile(raw, [25, 75], axis=0)
    scale = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
    standardized = np.clip((raw - median) / scale, -8.0, 8.0)
    standardized -= standardized.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    variance = singular**2
    keep = int(np.searchsorted(np.cumsum(variance) / variance.sum(), 0.90) + 1)
    keep = max(2, min(15, keep))
    return (standardized @ vt[:keep].T).astype(np.float64)


def farthest_first(candidates: np.ndarray, coordinates: np.ndarray, n: int, seed: int) -> list[int]:
    """Deterministic maximin selection within one ecoregion-tier stratum."""
    if n <= 0:
        return []
    if n >= len(candidates):
        return candidates.tolist()
    rng = np.random.default_rng(seed)
    local = coordinates[candidates]
    center = np.median(local, axis=0)
    distance_center = np.sum((local - center) ** 2, axis=1)
    first_pool = np.flatnonzero(distance_center == distance_center.max())
    first = int(rng.choice(first_pool))
    selected_local = [first]
    min_distance = np.sum((local - local[first]) ** 2, axis=1)
    min_distance[first] = -np.inf
    while len(selected_local) < n:
        best_pool = np.flatnonzero(min_distance == np.nanmax(min_distance))
        chosen = int(rng.choice(best_pool))
        selected_local.append(chosen)
        new_distance = np.sum((local - local[chosen]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, new_distance)
        min_distance[selected_local] = -np.inf
    return candidates[np.asarray(selected_local, dtype=int)].tolist()


def make_split(reference: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    attribute = robust_attribute_space(reference, feature_names)
    strata = reference["ecoregion"].astype(str) + "__" + reference["tier_short"].astype(str)
    counts = strata.value_counts().sort_index()
    donor_quota = largest_remainder(counts, N_DONOR)
    donor_index: list[int] = []
    for number, label in enumerate(counts.index):
        candidates = np.flatnonzero(strata.to_numpy() == label)
        donor_index.extend(
            farthest_first(candidates, attribute, donor_quota[label], SEED + number)
        )

    donor_mask = np.zeros(len(reference), dtype=bool)
    donor_mask[donor_index] = True
    remaining_counts = strata[~donor_mask].value_counts().sort_index()
    development_quota = largest_remainder(remaining_counts, N_DEVELOPMENT)
    development_index: list[int] = []
    for number, label in enumerate(remaining_counts.index):
        candidates = np.flatnonzero((strata.to_numpy() == label) & ~donor_mask)
        development_index.extend(
            farthest_first(
                candidates,
                attribute,
                development_quota[label],
                SEED + 1000 + number,
            )
        )

    role = np.full(len(reference), "target", dtype=object)
    role[np.asarray(donor_index)] = "donor"
    role[np.asarray(development_index)] = "development"
    result = reference.copy()
    result["reference_only_role"] = role
    expected = {"donor": N_DONOR, "development": N_DEVELOPMENT, "target": N_TARGET}
    if result.reference_only_role.value_counts().to_dict() != expected:
        raise RuntimeError(result.reference_only_role.value_counts().to_dict())
    return result


def read_observed(job: tuple[int, str]) -> tuple[int, np.ndarray]:
    q_row, gage_id = job
    frame = pd.read_csv(
        DAILY / f"{gage_id}.csv.gz",
        usecols=["date", "q_mm_day"],
        parse_dates=["date"],
    ).drop_duplicates("date").set_index("date").sort_index()
    dates = pd.date_range("2003-10-05", "2024-12-31", freq="D")
    q = pd.to_numeric(frame.reindex(dates).q_mm_day, errors="coerce").to_numpy(np.float32)
    return q_row, q


def build_common_anchors(forcing: np.ndarray, dates: np.ndarray, target_rows: np.ndarray) -> np.ndarray:
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    local_dates = dates[eval_start:eval_end]
    month = pd.DatetimeIndex(local_dates.astype("datetime64[ns]")).month.to_numpy()
    warm_season = (month >= 6) & (month <= 9)
    quantiles = np.linspace(0.05, 0.95, 12)
    result = np.full((len(target_rows), len(quantiles)), -1, dtype=np.int16)
    for basin, row in enumerate(target_rows):
        x = np.asarray(forcing[row, eval_start:eval_end, :3], dtype=float)
        balance = x[:, 0] - x[:, 1]
        rolling = np.convolve(balance, np.ones(30), mode="full")[: len(balance)]
        rolling[:29] = np.nan
        candidates = np.flatnonzero(warm_season & (x[:, 2] > 5.0) & np.isfinite(rolling))
        if len(candidates) < len(quantiles):
            candidates = np.flatnonzero(warm_season & np.isfinite(rolling))
        targets = np.quantile(rolling[candidates], quantiles)
        used: set[int] = set()
        selected: list[int] = []
        for value in targets:
            order = candidates[np.argsort(np.abs(rolling[candidates] - value), kind="stable")]
            chosen = next((int(item) for item in order if int(item) not in used), int(order[0]))
            used.add(chosen)
            selected.append(chosen)
        result[basin] = np.asarray(selected, dtype=np.int16)
    return result


def main() -> None:
    frozen = PROJECT / "data/frozen"
    parent_processed = PROJECT / "data/processed"
    data_dir = EXPERIMENT / "data"
    provenance = EXPERIMENT / "provenance"
    data_dir.mkdir(parents=True, exist_ok=True)
    provenance.mkdir(parents=True, exist_ok=True)

    all_gauges = pd.read_csv(frozen / "station_split.csv", dtype={"GAGE_ID": str}, low_memory=False)
    all_gauges["GAGE_ID"] = normalize_id(all_gauges.GAGE_ID)
    reference = all_gauges.loc[
        all_gauges.reference_status.eq("Reference")
        & all_gauges.tier_short.isin(["Tier 1", "Tier 2"])
    ].copy().sort_values("GAGE_ID").reset_index(drop=True)
    if len(reference) != 913:
        raise RuntimeError(f"Expected 913 eligible reference gauges, found {len(reference)}")

    catalog = pd.read_csv(frozen / "model_feature_catalog.csv")
    feature_names = catalog.loc[catalog.active.astype(bool), "model_feature"].tolist()
    split = make_split(reference, feature_names)
    role_order = pd.Categorical(
        split.reference_only_role, ["donor", "development", "target"], ordered=True
    )
    split = split.assign(_order=role_order).sort_values(["_order", "GAGE_ID"]).drop(columns="_order").reset_index(drop=True)
    split.insert(0, "row", np.arange(len(split), dtype=int))
    split["q_row"] = -1
    observed = split.reference_only_role.isin(["donor", "development"])
    split.loc[observed, "q_row"] = np.arange(int(observed.sum()), dtype=int)

    parent_index = pd.read_csv(parent_processed / "station_index.csv", dtype={"GAGE_ID": str})
    parent_index.GAGE_ID = normalize_id(parent_index.GAGE_ID)
    parent_row = dict(zip(parent_index.GAGE_ID, parent_index.row.astype(int)))
    if not set(split.GAGE_ID).issubset(parent_row):
        raise RuntimeError("Reference split includes gauges absent from the forcing archive")
    split["parent_forcing_row"] = split.GAGE_ID.map(parent_row).astype(int)
    split["role"] = split.reference_only_role
    split.to_csv(data_dir / "station_index.csv", index=False)

    parent_dates = np.load(parent_processed / "dates.npy", mmap_mode="r")
    parent_forcing = np.load(parent_processed / "forcing.npy", mmap_mode="r")
    n, t = len(split), len(parent_dates)
    forcing = np.lib.format.open_memmap(
        data_dir / "forcing.npy", mode="w+", dtype=np.float32, shape=(n, t, 5)
    )
    for first in range(0, n, 64):
        stop = min(first + 64, n)
        rows = split.parent_forcing_row.iloc[first:stop].to_numpy(int)
        forcing[first:stop] = np.asarray(parent_forcing[rows])
    forcing.flush()
    np.save(data_dir / "dates.npy", np.asarray(parent_dates))

    raw_static = split[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    donor_mask = split.role.eq("donor").to_numpy()
    medians = np.nanmedian(raw_static[donor_mask], axis=0)
    if not np.isfinite(medians).all():
        raise RuntimeError("All-missing donor static attribute")
    filled = np.where(np.isfinite(raw_static), raw_static, medians[None])
    q25, q75 = np.nanpercentile(filled[donor_mask], [25, 75], axis=0)
    scales = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    one_hot = np.asarray([name.startswith("geology__") for name in feature_names])
    static = np.clip((filled - medians[None]) / scales[None], -8.0, 8.0).astype(np.float32)
    static[:, one_hot] = filled[:, one_hot]
    np.save(data_dir / "static.npy", static)
    pd.DataFrame(
        {
            "feature": feature_names,
            "donor_median": medians,
            "donor_iqr": scales,
            "one_hot_unscaled": one_hot,
        }
    ).to_csv(data_dir / "static_normalization.csv", index=False)

    q = np.lib.format.open_memmap(
        data_dir / "q_donor_development.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N_DONOR + N_DEVELOPMENT, t),
    )
    q[:] = np.nan
    jobs = [(int(row.q_row), str(row.GAGE_ID)) for row in split.loc[observed].itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(read_observed, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            q_row, values = future.result()
            q[q_row] = values
            if count % 100 == 0 or count == len(futures):
                print(f"donor/development discharge {count}/{len(futures)}", flush=True)
    q.flush()

    target_rows = split.loc[split.role.eq("target"), "row"].to_numpy(int)
    common_time = build_common_anchors(forcing, np.asarray(parent_dates), target_rows)
    np.savez_compressed(
        data_dir / "common_forcing_anchors.npz",
        gage_ids=split.loc[split.role.eq("target"), "GAGE_ID"].to_numpy(str),
        quantiles=np.linspace(0.05, 0.95, 12).astype(np.float32),
        eval_time_index=common_time,
        definition="warm-season quantiles of antecedent 30-day P-minus-PET",
    )

    census = (
        split.groupby(["ecoregion", "tier_short", "role"], observed=True)
        .size().rename("gauges").reset_index()
    )
    census.to_csv(provenance / "split_census.csv", index=False)
    summary = {
        "population": "GAGES-II reference gauges meeting Tier 1 or Tier 2 data criteria",
        "eligible_reference_gauges": len(split),
        "donors": N_DONOR,
        "development": N_DEVELOPMENT,
        "targets": N_TARGET,
        "split_seed": SEED,
        "split_method": "ecoregion-by-tier proportional allocation with static-attribute maximin selection",
        "nonreference_gauges_used": 0,
        "target_discharge_accessed": False,
        "static_normalization_population": "400 reference donors only",
        "forcing_period": [str(parent_dates[0]), str(parent_dates[-1])],
        "station_index_sha256": sha256(data_dir / "station_index.csv"),
        "forcing_sha256": sha256(data_dir / "forcing.npy"),
        "static_sha256": sha256(data_dir / "static.npy"),
        "q_train_development_sha256": sha256(data_dir / "q_donor_development.npy"),
    }
    (provenance / "preprediction_data_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
