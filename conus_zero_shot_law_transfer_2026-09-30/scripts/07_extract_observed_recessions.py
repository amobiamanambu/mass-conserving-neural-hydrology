#!/usr/bin/env python3
"""Extract independent warm-season rain-free recession fingerprints."""

from __future__ import annotations

import hashlib
import json
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


SCENARIOS = {
    "primary": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.001},
    "strict_rain": {"future_p": 0.1, "prior_p": 0.5, "declines": 6, "q0": 0.001},
    "relaxed_rain": {"future_p": 2.0, "prior_p": 4.0, "declines": 0, "q0": 0.001},
    "low_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.0001},
    "high_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.01},
}


def events_for_basin(
    q: np.ndarray, forcing: np.ndarray, dates: pd.DatetimeIndex, scenario: dict[str, float | int]
) -> list[dict[str, float | int | str]]:
    n = len(q)
    q_window = np.lib.stride_tricks.sliding_window_view(q, 8)
    p_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 0], 8)
    t_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 2], 8)
    starts = np.arange(2, n - 7)
    local_q = q_window[starts]
    decline_count = np.sum(
        np.diff(local_q, axis=1) <= np.maximum(1e-8, 0.01 * local_q[:, :-1]), axis=1
    )
    prior_p = forcing[starts - 2, 0] + forcing[starts - 1, 0]
    months = dates.month.to_numpy()[starts]
    mask = (
        np.isfinite(local_q).all(axis=1)
        & (local_q[:, 0] >= scenario["q0"])
        & (months >= 6) & (months <= 9)
        & (np.nanmin(t_window[starts], axis=1) > 5.0)
        & (prior_p <= scenario["prior_p"])
        & (np.nansum(p_window[starts], axis=1) <= scenario["future_p"])
        & (decline_count >= scenario["declines"])
        & (local_q[:, -1] < local_q[:, 0])
    )
    candidate_starts = starts[mask]
    candidate_q = local_q[mask]
    candidate_declines = decline_count[mask]

    selected = []
    next_allowed = 0
    for start, segment_q, declines in zip(candidate_starts, candidate_q, candidate_declines):
        if start < next_allowed:
            continue
        selected.append({
            "start_index": start,
            "start_date": str(dates[start].date()),
            "q0_mm_day": float(segment_q[0]),
            "q7_mm_day": float(segment_q[-1]),
            "r7": float(segment_q[-1] / segment_q[0]),
            "daily_declines": int(declines),
            "prior_2day_p_mm": float(np.nansum(forcing[start - 2:start, 0])),
            "event_8day_p_mm": float(np.nansum(forcing[start:start + 8, 0])),
            "event_min_tmean_c": float(np.nanmin(forcing[start:start + 8, 2])),
        })
        next_allowed = start + 8
    return selected


def main() -> None:
    evaluation_receipt = json.loads((PROJECT / "provenance/target_evaluation_receipt.json").read_text())
    q_path = PROJECT / evaluation_receipt["target_discharge"]
    if sha256(q_path) != evaluation_receipt["target_discharge_sha256"]:
        raise RuntimeError("Target-discharge checksum mismatch")
    q = np.load(q_path, mmap_mode="r")
    forcing = np.load(PROJECT / "data/processed/forcing.npy", mmap_mode="r")
    dates_all = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    start = int(np.searchsorted(dates_all, np.datetime64("2019-01-01")))
    end = int(np.searchsorted(dates_all, np.datetime64("2024-12-31"), side="right"))
    dates = pd.DatetimeIndex(np.asarray(dates_all[start:end]).astype("datetime64[ns]"))
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations[stations.role.eq("target")].reset_index(drop=True)

    event_records, summary_records = [], []
    for basin, station in enumerate(target.itertuples(index=False)):
        local_forcing = np.asarray(forcing[int(station.row), start:end])
        for name, scenario in SCENARIOS.items():
            events = events_for_basin(np.asarray(q[basin]), local_forcing, dates, scenario)
            for event in events:
                event_records.append({"GAGE_ID": station.GAGE_ID, "scenario": name, **event})
            r7 = np.asarray([event["r7"] for event in events], dtype=float)
            summary_records.append({
                "GAGE_ID": station.GAGE_ID,
                "scenario": name,
                "events": len(events),
                "observed_median_r7": float(np.median(r7)) if len(r7) else np.nan,
                "observed_iqr_r7": float(np.quantile(r7, 0.75) - np.quantile(r7, 0.25)) if len(r7) >= 2 else np.nan,
                "observed_mad_r7": float(np.median(np.abs(r7 - np.median(r7)))) if len(r7) else np.nan,
                "usable": len(events) >= 3,
            })
        if (basin + 1) % 250 == 0 or basin + 1 == len(target):
            print(f"observed recessions {basin + 1:,}/{len(target):,}", flush=True)

    output_dir = PROJECT / "outputs/observed_recessions"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "recession_events.parquet"
    summary_path = output_dir / "recession_summary.parquet"
    pd.DataFrame(event_records).to_parquet(events_path, index=False)
    summary = pd.DataFrame(summary_records).merge(
        target[["GAGE_ID", "ecoregion", "tier_short", "reference_status", "distance_independent_50km"]],
        on="GAGE_ID", how="left", validate="many_to_one",
    )
    summary.to_parquet(summary_path, index=False)
    scenario_counts = (
        summary.groupby("scenario", as_index=False)
        .agg(gauges=("GAGE_ID", "size"), usable_gauges=("usable", "sum"))
    )
    usable_statistics = (
        summary.loc[summary["usable"]]
        .groupby("scenario", as_index=False)
        .agg(median_events=("events", "median"), median_observed_r7=("observed_median_r7", "median"))
    )
    scenario_summary = scenario_counts.merge(usable_statistics, on="scenario", how="left", validate="one_to_one")
    scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    receipt = {
        "target_discharge_sha256": evaluation_receipt["target_discharge_sha256"],
        "events": len(event_records),
        "events_file": str(events_path.relative_to(PROJECT)),
        "events_sha256": sha256(events_path),
        "summary_file": str(summary_path.relative_to(PROJECT)),
        "summary_sha256": sha256(summary_path),
        "scenarios": SCENARIOS,
    }
    (PROJECT / "provenance/observed_recession_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(scenario_summary.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
