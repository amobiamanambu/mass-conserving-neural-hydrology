#!/usr/bin/env python3
"""Independent, figure-free audit of the catchment-law identification test.

Purpose
-------
This single file reproduces the inferential analysis used to test whether two
mass-conserving hydrologic architectures learned the same transferable
seven-day recession law (R7). It covers the complete national target
population, the skill-qualified primary population, prespecified subgroups and
sensitivities, and the expanded GAGES-II reference-basin analysis with matched
non-reference controls.

The script is deliberately independent of every other project Python module.
It does not train neural networks and it does not make figures. Its scientific
starting point is the frozen, outcome-blind prediction/intervention archive.
When the three ``--recompute-*`` switches are supplied, it reconstructs:

1. target-gauge prediction metrics from frozen predictions and USGS discharge;
2. observed recession benchmarks from USGS discharge and GridMET forcing; and
3. model R7 fingerprints from the archived 90-day intervention curves.

Thus, with the data archive present, an auditor can recompute the evidence
without trusting the project's previously derived analysis tables.

Required software
-----------------
Python 3.10+ with numpy, pandas, scipy, and a Parquet engine (pyarrow).

Quick audit using archived derived datasets
-------------------------------------------
python independent_identification_audit.py --project-root /path/to/project

Deep audit rebuilding all derived inputs
----------------------------------------
python independent_identification_audit.py \
  --project-root /path/to/project \
  --recompute-skill \
  --recompute-observed-recessions \
  --recompute-fingerprints

All outputs are written to ``audit/identification_reanalysis_outputs`` by
default. No archived result or model output is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")
RUNGS = ("q", "qet")
PRIMARY_SKILL_THRESHOLD = 0.30
GLOBAL_MEDIAN_KGE_THRESHOLD = 0.40
GLOBAL_FRACTION_KGE_THRESHOLD = 0.60
MASS_BALANCE_THRESHOLD_MM = 0.005
CLASS_BREAKS = (0.5, float(np.sqrt(0.5)))
SKILL_THRESHOLDS = (0.0, 0.30, 0.40, 0.50, 0.60)
MODEL_Q0_THRESHOLDS = (0.0001, 0.001, 0.01)
RECESSION_SCENARIOS = {
    "primary": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.001},
    "strict_rain": {"future_p": 0.1, "prior_p": 0.5, "declines": 6, "q0": 0.001},
    "relaxed_rain": {"future_p": 2.0, "prior_p": 4.0, "declines": 0, "q0": 0.001},
    "low_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.0001},
    "high_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.01},
}
METRIC_COLUMNS = (
    "q_cross_architecture_rho",
    "q_cross_architecture_mad",
    "qet_cross_architecture_rho",
    "qet_cross_architecture_mad",
    "disagreement_reduction_fraction",
    "dhbv_observed_rho",
    "dhbv_observed_mae",
    "fluxgate_observed_rho",
    "fluxgate_observed_mae",
    "all_model_class_agreement_fraction",
)
ATTRIBUTE_COLUMNS = {
    "Climate__PPTAVG_BASIN": "pptavg_basin",
    "Climate__T_AVG_BASIN": "tavg_basin",
    "Climate__PET": "pet",
    "Climate__SNOW_PCT_PRECIP": "snow_pct_precip",
    "Climate__PRECIP_SEAS_IND": "precip_seasonality",
}
HYDROCLIMATE_MATCH_WEIGHTS = {
    # Fixed from outcome-blind covariate-balance diagnostics; no R7 or observed
    # recession outcome was used to choose these weights.
    "log10_area_km2": 4.0,
    "pptavg_basin": 2.0,
}


def find_project_root() -> Path:
    """Find a nearby project root when --project-root is omitted."""
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "provenance/frozen_prediction_registry.json").exists():
            return candidate
    return Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the national and reference-basin law-identification analysis."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=find_project_root(),
        help="Directory containing data/, outputs/, and provenance/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Audit output directory; defaults inside project/audit/.",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=1000,
        help="Ecoregion-stratified bootstrap draws (paper uses 1000).",
    )
    parser.add_argument(
        "--recompute-skill",
        action="store_true",
        help="Rebuild target model metrics from frozen predictions and target discharge.",
    )
    parser.add_argument(
        "--recompute-observed-recessions",
        action="store_true",
        help="Rebuild observed R7 events from target discharge and GridMET forcing.",
    )
    parser.add_argument(
        "--recompute-fingerprints",
        action="store_true",
        help="Rebuild median model R7 from archived intervention curves.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip asset checksum verification (not recommended for an audit).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def normalize_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_registry(project: Path, verify_checksums: bool) -> tuple[dict, list[dict]]:
    """Verify the outcome-blind registry itself and return an audit log."""
    registry_path = require_file(project / "provenance/frozen_prediction_registry.json")
    checksum_path = require_file(project / "provenance/frozen_prediction_registry.sha256")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    observed = sha256(registry_path)
    if verify_checksums and observed != expected:
        raise RuntimeError("Frozen prediction registry checksum mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("target_discharge_accessed") is not False:
        raise RuntimeError("Registry does not document outcome-blind freezing")
    return registry, [{
        "asset": str(registry_path.relative_to(project)),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "matches": observed == expected,
    }]


def verify_registered_asset(
    project: Path,
    run: dict,
    path_key: str,
    hash_key: str,
    verify_checksums: bool,
) -> dict:
    path = require_file(project / run[path_key])
    expected = run[hash_key]
    observed = sha256(path) if verify_checksums else "not_computed"
    if verify_checksums and observed != expected:
        raise RuntimeError(f"Checksum mismatch for {path}")
    return {
        "asset": str(path.relative_to(project)),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "matches": None if not verify_checksums else observed == expected,
    }


def load_target_stations(project: Path) -> pd.DataFrame:
    path = require_file(project / "data/processed/station_index.csv")
    stations = pd.read_csv(path, dtype={"GAGE_ID": str})
    require_columns(
        stations,
        (
            "GAGE_ID", "row", "role", "ecoregion", "tier_short",
            "reference_status", "area_km2", "distance_independent_50km",
        ),
        "station_index.csv",
    )
    stations["GAGE_ID"] = normalize_ids(stations.GAGE_ID)
    target = stations.loc[stations.role.eq("target")].copy().reset_index(drop=True)
    if target.GAGE_ID.duplicated().any():
        raise ValueError("Target GAGE_ID values are not unique")
    return target


# ---------------------------------------------------------------------------
# Optional reconstruction of prediction skill from frozen arrays
# ---------------------------------------------------------------------------


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    names = (
        "kge", "nse", "log_nse", "correlation", "variability_ratio", "mean_ratio",
        "percent_bias", "high_flow_bias_fraction", "low_flow_bias_fraction",
        "observed_zero_fraction", "predicted_zero_fraction", "zero_flow_brier",
    )
    if valid.sum() < 365:
        return {name: np.nan for name in names} | {"valid_days": int(valid.sum())}

    o = observed[valid].astype(np.float64)
    p = predicted[valid].astype(np.float64)
    o_mean, p_mean = o.mean(), p.mean()
    o_std, p_std = o.std(ddof=1), p.std(ddof=1)
    correlation = float(np.corrcoef(o, p)[0, 1]) if o_std > 0 and p_std > 0 else np.nan
    alpha = float(p_std / o_std) if o_std > 0 else np.nan
    beta = float(p_mean / o_mean) if o_mean > 0 else np.nan
    if np.isfinite([correlation, alpha, beta]).all():
        kge = float(1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = np.nan
    high = o >= np.quantile(o, 0.95)
    low = o <= np.quantile(o, 0.20)

    def volume_bias(mask: np.ndarray) -> float:
        denominator = o[mask].sum()
        return float((p[mask].sum() - denominator) / denominator) if denominator > 1e-8 else np.nan

    observed_zero = o < 0.001
    predicted_zero = p < 0.001
    # log-NSE is NSE on ln(1 + Q) with Q in mm d-1, not ln(Q + epsilon).
    return {
        "valid_days": int(valid.sum()),
        "kge": kge,
        "nse": nse(o, p),
        "log_nse": nse(np.log1p(o), np.log1p(p)),
        "correlation": correlation,
        "variability_ratio": alpha,
        "mean_ratio": beta,
        "percent_bias": float(100.0 * (p.sum() - o.sum()) / o.sum()) if o.sum() > 0 else np.nan,
        "high_flow_bias_fraction": volume_bias(high),
        "low_flow_bias_fraction": volume_bias(low),
        "observed_zero_fraction": float(observed_zero.mean()),
        "predicted_zero_fraction": float(predicted_zero.mean()),
        "zero_flow_brier": float(
            np.mean((predicted_zero.astype(float) - observed_zero.astype(float)) ** 2)
        ),
    }


def load_target_discharge(
    project: Path, target: pd.DataFrame, verify_checksums: bool
) -> tuple[np.ndarray, dict]:
    receipt_path = require_file(project / "provenance/target_evaluation_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    q_path = require_file(project / receipt["target_discharge"])
    observed_hash = sha256(q_path) if verify_checksums else "not_computed"
    if verify_checksums and observed_hash != receipt["target_discharge_sha256"]:
        raise RuntimeError("Target-discharge checksum mismatch")
    q = np.load(q_path, mmap_mode="r")
    if q.shape[0] != len(target):
        raise ValueError(f"Target discharge has {q.shape[0]} rows; expected {len(target)}")
    audit = {
        "asset": str(q_path.relative_to(project)),
        "expected_sha256": receipt["target_discharge_sha256"],
        "observed_sha256": observed_hash,
        "matches": None if not verify_checksums else observed_hash == receipt["target_discharge_sha256"],
    }
    return q, audit


def rebuild_skill_metrics(
    project: Path,
    target: pd.DataFrame,
    registry: dict,
    q: np.ndarray,
    integrity_log: list[dict],
    verify_checksums: bool,
) -> pd.DataFrame:
    predictions: dict[str, np.ndarray] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    metadata: dict[str, dict] = {}
    for run in registry["runs"]:
        integrity_log.append(
            verify_registered_asset(
                project, run, "prediction", "prediction_sha256", verify_checksums
            )
        )
        prediction = np.load(project / run["prediction"], mmap_mode="r")
        if prediction.shape != q.shape:
            raise ValueError(f"Prediction shape mismatch for {run['run']}: {prediction.shape}")
        predictions[run["run"]] = prediction
        metadata[run["run"]] = run
        groups.setdefault((run["architecture"], run["rung"]), []).append(run["run"])

    records: list[dict] = []
    for run_name, prediction in predictions.items():
        run = metadata[run_name]
        for basin, station in enumerate(target.itertuples(index=False)):
            records.append({
                "GAGE_ID": station.GAGE_ID,
                "model": run_name,
                "architecture": run["architecture"],
                "rung": run["rung"],
                "seed": int(run["seed"]),
                "ensemble": False,
                **prediction_metrics(np.asarray(q[basin]), np.asarray(prediction[basin])),
            })
        print(f"Recomputed prediction metrics: {run_name}", flush=True)

    for (architecture, rung), run_names in sorted(groups.items()):
        ensemble_name = f"{architecture}_{rung}_ensemble"
        for basin, station in enumerate(target.itertuples(index=False)):
            # Match the frozen evaluation pipeline: average the three seed
            # hydrographs before calculating gauge-level prediction metrics.
            prediction = np.mean(
                np.stack([np.asarray(predictions[name][basin]) for name in run_names]), axis=0
            )
            records.append({
                "GAGE_ID": station.GAGE_ID,
                "model": ensemble_name,
                "architecture": architecture,
                "rung": rung,
                "seed": -1,
                "ensemble": True,
                **prediction_metrics(np.asarray(q[basin]), prediction),
            })
        print(f"Recomputed prediction metrics: {ensemble_name}", flush=True)

    result = pd.DataFrame(records).merge(
        target[[
            "GAGE_ID", "ecoregion", "tier_short", "reference_status",
            "latitude", "longitude", "area_km2", "distance_independent_50km",
        ]],
        on="GAGE_ID",
        how="left",
        validate="many_to_one",
    )
    return result


# ---------------------------------------------------------------------------
# Optional reconstruction of observed recession events
# ---------------------------------------------------------------------------


def events_for_basin(
    q: np.ndarray,
    forcing: np.ndarray,
    dates: pd.DatetimeIndex,
    scenario: dict[str, float | int],
) -> list[dict[str, float | int | str]]:
    n = len(q)
    q_window = np.lib.stride_tricks.sliding_window_view(q, 8)
    p_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 0], 8)
    t_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 2], 8)
    starts = np.arange(2, n - 7)
    local_q = q_window[starts]
    decline_count = np.sum(
        np.diff(local_q, axis=1) <= np.maximum(1e-8, 0.01 * local_q[:, :-1]),
        axis=1,
    )
    prior_p = forcing[starts - 2, 0] + forcing[starts - 1, 0]
    months = dates.month.to_numpy()[starts]
    mask = (
        np.isfinite(local_q).all(axis=1)
        & (local_q[:, 0] >= scenario["q0"])
        & (months >= 6)
        & (months <= 9)
        & (np.nanmin(t_window[starts], axis=1) > 5.0)
        & (prior_p <= scenario["prior_p"])
        & (np.nansum(p_window[starts], axis=1) <= scenario["future_p"])
        & (decline_count >= scenario["declines"])
        & (local_q[:, -1] < local_q[:, 0])
    )
    candidate_starts = starts[mask]
    candidate_q = local_q[mask]
    candidate_declines = decline_count[mask]

    selected: list[dict[str, float | int | str]] = []
    next_allowed = 0
    for start, segment_q, declines in zip(
        candidate_starts, candidate_q, candidate_declines, strict=True
    ):
        if start < next_allowed:
            continue
        selected.append({
            "start_index": int(start),
            "start_date": str(dates[start].date()),
            "q0_mm_day": float(segment_q[0]),
            "q7_mm_day": float(segment_q[-1]),
            "r7": float(segment_q[-1] / segment_q[0]),
            "daily_declines": int(declines),
            "prior_2day_p_mm": float(np.nansum(forcing[start - 2:start, 0])),
            "event_8day_p_mm": float(np.nansum(forcing[start:start + 8, 0])),
            "event_min_tmean_c": float(np.nanmin(forcing[start:start + 8, 2])),
        })
        next_allowed = int(start) + 8
    return selected


def rebuild_observed_recessions(
    project: Path,
    target: pd.DataFrame,
    q: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forcing = np.load(require_file(project / "data/processed/forcing.npy"), mmap_mode="r")
    dates_all = np.load(require_file(project / "data/processed/dates.npy"), mmap_mode="r")
    start = int(np.searchsorted(dates_all, np.datetime64("2019-01-01")))
    end = int(np.searchsorted(dates_all, np.datetime64("2024-12-31"), side="right"))
    dates = pd.DatetimeIndex(np.asarray(dates_all[start:end]).astype("datetime64[ns]"))
    if q.shape[1] != len(dates):
        raise ValueError("Target discharge and evaluation-date lengths differ")

    event_records: list[dict] = []
    summary_records: list[dict] = []
    for basin, station in enumerate(target.itertuples(index=False)):
        local_forcing = np.asarray(forcing[int(station.row), start:end])
        for name, scenario in RECESSION_SCENARIOS.items():
            events = events_for_basin(np.asarray(q[basin]), local_forcing, dates, scenario)
            event_records.extend(
                {"GAGE_ID": station.GAGE_ID, "scenario": name, **event}
                for event in events
            )
            r7 = np.asarray([event["r7"] for event in events], dtype=float)
            summary_records.append({
                "GAGE_ID": station.GAGE_ID,
                "scenario": name,
                "events": len(events),
                "observed_median_r7": float(np.median(r7)) if len(r7) else np.nan,
                "observed_iqr_r7": (
                    float(np.quantile(r7, 0.75) - np.quantile(r7, 0.25))
                    if len(r7) >= 2 else np.nan
                ),
                "observed_mad_r7": (
                    float(np.median(np.abs(r7 - np.median(r7)))) if len(r7) else np.nan
                ),
                "usable": len(events) >= 3,
            })
        if (basin + 1) % 250 == 0 or basin + 1 == len(target):
            print(f"Recomputed observed recessions: {basin + 1:,}/{len(target):,}", flush=True)

    summary = pd.DataFrame(summary_records).merge(
        target[[
            "GAGE_ID", "ecoregion", "tier_short", "reference_status",
            "distance_independent_50km",
        ]],
        on="GAGE_ID",
        how="left",
        validate="many_to_one",
    )
    return pd.DataFrame(event_records), summary


# ---------------------------------------------------------------------------
# Optional reconstruction of R7 fingerprints from intervention curves
# ---------------------------------------------------------------------------


def rebuild_fingerprint_from_curves(curve_path: Path, q0_threshold: float = 0.001) -> pd.DataFrame:
    archive = np.load(curve_path)
    gage_ids = pd.Series(archive["gage_ids"].astype(str))
    # Preserve the archived float32 division before assigning into float64.
    # This exactly mirrors the original fingerprint calculation and avoids
    # harmless 1e-8 differences caused by promoting both operands first.
    q = archive["q_mm_day"]
    q0 = q[:, :, 0]
    valid = np.isfinite(q0) & (q0 >= q0_threshold)
    r7 = np.full(q0.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r7[valid] = q[:, :, 7][valid] / q0[valid]
    finite = np.isfinite(r7)
    valid_anchors = finite.sum(axis=1)
    median_r7 = np.full(len(gage_ids), np.nan, dtype=float)
    usable = valid_anchors > 0
    median_r7[usable] = np.nanmedian(r7[usable], axis=1)
    return pd.DataFrame({
        "GAGE_ID": normalize_ids(gage_ids),
        "valid_anchors": valid_anchors,
        "median_r7": median_r7,
    })


def load_fingerprints(
    project: Path,
    registry: dict,
    recompute: bool,
    integrity_log: list[dict],
    verify_checksums: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    tables: dict[str, pd.DataFrame] = {}
    reconstruction_checks: list[dict] = []
    constrained_runs = [run for run in registry["runs"] if run["architecture"] in ARCHITECTURES]
    for run in constrained_runs:
        if recompute:
            integrity_log.append(
                verify_registered_asset(
                    project, run, "law_curves", "law_curves_sha256", verify_checksums
                )
            )
            table = rebuild_fingerprint_from_curves(project / run["law_curves"])
            archived_path = require_file(project / run["law_fingerprints"])
            archived = pd.read_parquet(archived_path)
            archived["GAGE_ID"] = normalize_ids(archived.GAGE_ID)
            comparison = table.merge(
                archived[["GAGE_ID", "valid_anchors", "median_r7"]],
                on="GAGE_ID",
                suffixes=("_rebuilt", "_archived"),
                validate="one_to_one",
            )
            max_difference = float(
                np.nanmax(np.abs(comparison.median_r7_rebuilt - comparison.median_r7_archived))
            )
            reconstruction_checks.append({
                "run": run["run"],
                "gauges": len(table),
                "max_absolute_median_r7_difference": max_difference,
                "valid_anchor_counts_identical": bool(
                    np.array_equal(
                        comparison.valid_anchors_rebuilt,
                        comparison.valid_anchors_archived,
                    )
                ),
            })
        else:
            integrity_log.append(
                verify_registered_asset(
                    project, run, "law_fingerprints", "law_fingerprints_sha256", verify_checksums
                )
            )
            table = pd.read_parquet(project / run["law_fingerprints"])
            require_columns(table, ("GAGE_ID", "median_r7"), run["run"])
            table["GAGE_ID"] = normalize_ids(table.GAGE_ID)
        if table.GAGE_ID.duplicated().any():
            raise ValueError(f"Duplicate fingerprint GAGE_ID values in {run['run']}")
        tables[run["run"]] = table
    return tables, reconstruction_checks


# ---------------------------------------------------------------------------
# Gauge-level assembly and identification metrics
# ---------------------------------------------------------------------------


def rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def median_absolute_difference(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    difference = np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    return float(np.nanmedian(difference))


def retention_class(values: np.ndarray) -> np.ndarray:
    """Assign persistence classes while preserving missing fingerprints."""
    values = np.asarray(values, dtype=float)
    classes = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    classes[valid] = np.digitize(values[valid], CLASS_BREAKS)
    return classes


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    q_mad = median_absolute_difference(frame.dhbv_q_r7, frame.fluxgate_q_r7)
    qet_mad = median_absolute_difference(frame.dhbv_qet_r7, frame.fluxgate_qet_r7)
    return {
        "gauges": int(len(frame)),
        "q_cross_architecture_rho": rho(frame.dhbv_q_r7, frame.fluxgate_q_r7),
        "q_cross_architecture_mad": q_mad,
        "qet_cross_architecture_rho": rho(frame.dhbv_qet_r7, frame.fluxgate_qet_r7),
        "qet_cross_architecture_mad": qet_mad,
        "disagreement_reduction_fraction": (
            float(1.0 - qet_mad / q_mad) if q_mad > 0 else np.nan
        ),
        "dhbv_observed_rho": rho(frame.dhbv_qet_r7, frame.observed_median_r7),
        "dhbv_observed_mae": median_absolute_difference(
            frame.dhbv_qet_r7, frame.observed_median_r7
        ),
        "fluxgate_observed_rho": rho(frame.fluxgate_qet_r7, frame.observed_median_r7),
        "fluxgate_observed_mae": median_absolute_difference(
            frame.fluxgate_qet_r7, frame.observed_median_r7
        ),
        "all_model_class_agreement_fraction": float(frame.class_agreement.mean(skipna=True)),
    }


def assemble_gauge_level(
    target: pd.DataFrame,
    fingerprints: dict[str, pd.DataFrame],
    skill: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    frame = target.set_index("GAGE_ID").copy()
    seed_columns: list[str] = []
    for architecture in ARCHITECTURES:
        for rung in RUNGS:
            local_columns = []
            for seed in SEEDS:
                run = f"{architecture}_{rung}_seed{seed}"
                table = fingerprints[run].set_index("GAGE_ID")
                missing = frame.index.difference(table.index)
                if len(missing):
                    raise ValueError(f"{run} is missing {len(missing)} target gauges")
                column = f"{run}_r7"
                frame[column] = table.reindex(frame.index).median_r7.to_numpy(float)
                seed_columns.append(column)
                local_columns.append(column)
            frame[f"{architecture}_{rung}_r7"] = frame[local_columns].median(axis=1)

    qet_seed_columns = [column for column in seed_columns if "_qet_" in column]
    classes = np.column_stack(
        [retention_class(frame[column].to_numpy(float)) for column in qet_seed_columns]
    )
    complete = np.isfinite(classes).all(axis=1)
    class_agreement = np.full(len(frame), np.nan, dtype=float)
    class_agreement[complete] = np.all(
        classes[complete] == classes[complete][:, [0]], axis=1
    )
    frame["class_agreement"] = class_agreement

    skill = skill.copy()
    skill["GAGE_ID"] = normalize_ids(skill.GAGE_ID)
    ensemble = skill.loc[skill.ensemble.astype(bool)].copy()
    if ensemble.duplicated(["GAGE_ID", "architecture", "rung"]).any():
        raise ValueError("Ensemble skill table has duplicate gauge/model rows")
    ensemble = ensemble.set_index(["GAGE_ID", "architecture", "rung"])
    for architecture in ("ealstm", *ARCHITECTURES):
        available_rungs = ("q",) if architecture == "ealstm" else RUNGS
        for rung in available_rungs:
            local = ensemble.xs((architecture, rung), level=("architecture", "rung"))
            frame[f"{architecture}_{rung}_kge"] = local.reindex(frame.index).kge.to_numpy(float)

    observed = observed.copy()
    observed["GAGE_ID"] = normalize_ids(observed.GAGE_ID)
    primary_observed = observed.loc[observed.scenario.eq("primary")].set_index("GAGE_ID")
    frame["observed_events"] = primary_observed.reindex(frame.index).events.to_numpy(float)
    frame["observed_median_r7"] = (
        primary_observed.reindex(frame.index).observed_median_r7.to_numpy(float)
    )
    frame["observed_usable"] = (
        primary_observed.reindex(frame.index).usable.fillna(False).to_numpy(bool)
    )
    return frame.reset_index()


def primary_population(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (
        frame.observed_usable
        & frame.dhbv_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
        & frame.fluxgate_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
    )
    return frame.loc[mask].copy().reset_index(drop=True)


def stratified_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = [
        group.index.to_numpy()
        for _, group in frame.groupby("ecoregion", sort=True)
    ]
    return np.concatenate([
        rng.choice(index, size=len(index), replace=True) for index in groups
    ])


def bootstrap_metrics(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(draws):
        chosen = stratified_indices(frame, rng)
        records.append({"draw": draw, **calculate_metrics(frame.loc[chosen])})
    return pd.DataFrame(records)


def metrics_with_ci(
    label: str, frame: pd.DataFrame, draws: int, seed: int
) -> tuple[dict, pd.DataFrame]:
    point = calculate_metrics(frame)
    bootstrap = bootstrap_metrics(frame, draws, seed)
    record = {"population": label, **point}
    for metric in METRIC_COLUMNS:
        lo, hi = bootstrap[metric].quantile([0.025, 0.975])
        record[f"{metric}_ci_low"] = float(lo)
        record[f"{metric}_ci_high"] = float(hi)
    bootstrap.insert(1, "population", label)
    return record, bootstrap


def supports_nonidentification(record: pd.Series | dict) -> bool:
    """Apply the frozen support rule for subgroup and sensitivity analyses.

    This diagnostic requires failure of the magnitude gate, failure of the
    expected ET-direction criterion, and failure of at least one observational
    criterion for each architecture. Rank and class stability are reported for
    each population but are not components of this support flag.
    """
    return bool(
        record["qet_cross_architecture_mad"] > 0.05
        and record["disagreement_reduction_fraction"] < 0.20
        and (
            record["dhbv_observed_rho"] < 0.60
            or record["dhbv_observed_mae"] > 0.10
        )
        and (
            record["fluxgate_observed_rho"] < 0.60
            or record["fluxgate_observed_mae"] > 0.10
        )
    )


def subgroup_analysis(primary: pd.DataFrame) -> pd.DataFrame:
    records = []
    specs = {
        "ecoregion": sorted(primary.ecoregion.dropna().unique()),
        "tier_short": sorted(primary.tier_short.dropna().unique()),
        "reference_status": sorted(primary.reference_status.dropna().unique()),
        "distance_independent_50km": [True],
    }
    for variable, values in specs.items():
        for value in values:
            subset = primary.loc[primary[variable].eq(value)]
            if len(subset) >= 30:
                record = {
                    "group_variable": variable,
                    "group_value": str(value),
                    **calculate_metrics(subset),
                }
                record["supports_nonidentification"] = supports_nonidentification(record)
                records.append(record)
    for omitted in sorted(primary.ecoregion.dropna().unique()):
        subset = primary.loc[~primary.ecoregion.eq(omitted)]
        record = {
            "group_variable": "leave_one_ecoregion_out",
            "group_value": str(omitted),
            **calculate_metrics(subset),
        }
        record["supports_nonidentification"] = supports_nonidentification(record)
        records.append(record)
    return pd.DataFrame(records)


def sensitivity_analysis(frame: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    observed = observed.copy()
    observed["GAGE_ID"] = normalize_ids(observed.GAGE_ID)
    observed_wide = observed.pivot(
        index="GAGE_ID", columns="scenario", values=["observed_median_r7", "usable"]
    )
    records = []
    for threshold in SKILL_THRESHOLDS:
        for scenario in RECESSION_SCENARIOS:
            usable = observed_wide.reindex(frame.GAGE_ID)[("usable", scenario)].to_numpy(bool)
            subset = frame.loc[
                frame.dhbv_qet_kge.ge(threshold)
                & frame.fluxgate_qet_kge.ge(threshold)
                & usable
            ].copy()
            subset["observed_median_r7"] = observed_wide.reindex(subset.GAGE_ID)[
                ("observed_median_r7", scenario)
            ].to_numpy(float)
            record = {
                "skill_threshold": threshold,
                "recession_scenario": scenario,
                **calculate_metrics(subset),
            }
            record["supports_nonidentification"] = supports_nonidentification(record)
            records.append(record)
    return pd.DataFrame(records)


def model_q0_sensitivity(
    project: Path,
    registry: dict,
    target: pd.DataFrame,
    skill: pd.DataFrame,
    observed: pd.DataFrame,
    integrity_log: list[dict],
    verify_checksums: bool,
) -> pd.DataFrame:
    """Rebuild fingerprints under each prespecified modeled-Q0 threshold.

    Observed recession R7 remains at its primary definition. This isolates the
    model-side admissibility threshold and uses only the frozen intervention
    curves; no model is retrained or rerun.
    """
    constrained_runs = [
        run for run in registry["runs"] if run["architecture"] in ARCHITECTURES
    ]
    verified: set[str] = set()
    records = []
    for threshold in MODEL_Q0_THRESHOLDS:
        tables: dict[str, pd.DataFrame] = {}
        for run in constrained_runs:
            if run["run"] not in verified:
                integrity_log.append(
                    verify_registered_asset(
                        project, run, "law_curves", "law_curves_sha256", verify_checksums
                    )
                )
                verified.add(run["run"])
            tables[run["run"]] = rebuild_fingerprint_from_curves(
                project / run["law_curves"], q0_threshold=threshold
            )

        full = assemble_gauge_level(target, tables, skill, observed)
        primary = primary_population(full)
        seed_columns = [
            f"{architecture}_qet_seed{seed}_r7"
            for architecture in ARCHITECTURES
            for seed in SEEDS
        ]
        ensemble_columns = [
            f"{architecture}_{rung}_r7"
            for architecture in ARCHITECTURES
            for rung in RUNGS
        ]
        complete = np.isfinite(primary[[*seed_columns, *ensemble_columns]]).all(axis=1)
        analysis = primary.loc[complete].copy()
        metrics = calculate_metrics(analysis)
        record = {
            "model_q0_threshold_mm_day": threshold,
            "primary_gauges": int(len(primary)),
            "gauges_excluded_for_q0": int((~complete).sum()),
            **metrics,
        }
        record["supports_nonidentification"] = supports_nonidentification(record)
        records.append(record)
    return pd.DataFrame(records)


def global_skill(frame: pd.DataFrame) -> dict:
    result = {}
    for architecture in ARCHITECTURES:
        values = frame[f"{architecture}_qet_kge"].to_numpy(float)
        median_kge = float(np.nanmedian(values))
        fraction = float(np.mean(values >= PRIMARY_SKILL_THRESHOLD))
        result[architecture] = {
            "median_kge": median_kge,
            "fraction_kge_ge_030": fraction,
            "passes": bool(
                median_kge >= GLOBAL_MEDIAN_KGE_THRESHOLD
                and fraction >= GLOBAL_FRACTION_KGE_THRESHOLD
            ),
        }
    return result


def mass_balance_audit(project: Path) -> dict:
    records = []
    for path in sorted((project / "outputs/law_interventions").glob("*_receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if "maximum_daily_mass_residual_mm" not in receipt:
            continue
        records.append({
            "run": receipt["run"],
            "maximum_daily_mass_residual_mm": float(receipt["maximum_daily_mass_residual_mm"]),
            "target_discharge_accessed": receipt.get("target_discharge_accessed"),
        })
    if not records:
        raise FileNotFoundError("No intervention mass-balance receipts were found")
    maximum = max(record["maximum_daily_mass_residual_mm"] for record in records)
    return {
        "maximum_daily_residual_mm": maximum,
        "threshold_mm": MASS_BALANCE_THRESHOLD_MM,
        "passes": maximum <= MASS_BALANCE_THRESHOLD_MM,
        "note": (
            "The intervention receipts archive maximum daily residuals. "
            "Cumulative training-run residuals are outside this inferential audit."
        ),
        "runs": records,
    }


def national_gates(
    frame: pd.DataFrame,
    primary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    subgroups: pd.DataFrame,
    sensitivities: pd.DataFrame,
    model_q0: pd.DataFrame,
    mass_balance: dict,
) -> dict:
    metrics = calculate_metrics(primary)

    def interval(metric: str) -> list[float]:
        return [float(value) for value in bootstrap[metric].quantile([0.025, 0.975])]

    competence = global_skill(frame)
    ecoregions = subgroups.loc[subgroups.group_variable.eq("ecoregion")]
    gates = {
        "numerical_mass_balance": mass_balance,
        "global_predictive_competence": {
            "models": competence,
            "passes": all(item["passes"] for item in competence.values()),
        },
        "gate_1_cross_architecture_rank": {
            "value": metrics["qet_cross_architecture_rho"],
            "threshold": 0.75,
            "ci95": interval("qet_cross_architecture_rho"),
            "passes": metrics["qet_cross_architecture_rho"] >= 0.75,
        },
        "gate_2_cross_architecture_magnitude": {
            "value": metrics["qet_cross_architecture_mad"],
            "threshold": 0.05,
            "ci95": interval("qet_cross_architecture_mad"),
            "passes": metrics["qet_cross_architecture_mad"] <= 0.05,
        },
        "gate_3_et_reduces_disagreement": {
            "value": metrics["disagreement_reduction_fraction"],
            "threshold": 0.20,
            "ci95": interval("disagreement_reduction_fraction"),
            "passes": bool(
                metrics["disagreement_reduction_fraction"] >= 0.20
                and interval("disagreement_reduction_fraction")[0] > 0
            ),
        },
        "gate_4_observed_recession": {
            "dhbv": {
                "rho": metrics["dhbv_observed_rho"],
                "mae": metrics["dhbv_observed_mae"],
                "rho_ci95": interval("dhbv_observed_rho"),
                "mae_ci95": interval("dhbv_observed_mae"),
            },
            "fluxgate": {
                "rho": metrics["fluxgate_observed_rho"],
                "mae": metrics["fluxgate_observed_mae"],
                "rho_ci95": interval("fluxgate_observed_rho"),
                "mae_ci95": interval("fluxgate_observed_mae"),
            },
            "thresholds": {"rho_min": 0.60, "mae_max": 0.10},
            "passes": bool(
                metrics["dhbv_observed_rho"] >= 0.60
                and metrics["dhbv_observed_mae"] <= 0.10
                and metrics["fluxgate_observed_rho"] >= 0.60
                and metrics["fluxgate_observed_mae"] <= 0.10
            ),
        },
        "gate_5_seed_architecture_class_stability": {
            "value": metrics["all_model_class_agreement_fraction"],
            "threshold": 0.75,
            "ci95": interval("all_model_class_agreement_fraction"),
            "class_breaks": list(CLASS_BREAKS),
            "passes": metrics["all_model_class_agreement_fraction"] >= 0.75,
        },
        "gate_6_robustness": {
            "ecoregions_showing_nonidentification": int(
                ecoregions.supports_nonidentification.sum()
            ),
            "ecoregions_required": 7,
            "both_tiers_reported": True,
            "reference_status_reported": True,
            "leave_one_ecoregion_out_reported": True,
            "strict_spatial_subset_reported": True,
            "supports_nonidentification": bool(
                ecoregions.supports_nonidentification.sum() >= 7
            ),
        },
        "gate_7_sensitivity": {
            "observed_recession_skill_grid_tests": int(len(sensitivities)),
            "alternative_model_q0_tests": int(
                model_q0.model_q0_threshold_mm_day.ne(0.001).sum()
            ),
            "tests": int(
                len(sensitivities)
                + model_q0.model_q0_threshold_mm_day.ne(0.001).sum()
            ),
            "tests_supporting_nonidentification": int(
                sensitivities.supports_nonidentification.sum()
                + model_q0.loc[
                    model_q0.model_q0_threshold_mm_day.ne(0.001),
                    "supports_nonidentification",
                ].sum()
            ),
            "supports_nonidentification": bool(
                sensitivities.supports_nonidentification.all()
                and model_q0.supports_nonidentification.all()
            ),
        },
    }
    identification_gate_names = (
        "global_predictive_competence",
        "gate_1_cross_architecture_rank",
        "gate_2_cross_architecture_magnitude",
        "gate_3_et_reduces_disagreement",
        "gate_4_observed_recession",
        "gate_5_seed_architecture_class_stability",
    )
    all_pass = all(gates[name]["passes"] for name in identification_gate_names)
    if all_pass and mass_balance["passes"]:
        formal_decision = "transferable_cross_structural_law_identified"
    elif not gates["global_predictive_competence"]["passes"] or not mass_balance["passes"]:
        formal_decision = "formally_inconclusive_global_test_due_to_predictive_competence_gate"
    else:
        formal_decision = "prediction_without_constitutive_law_identification"
    return {
        "primary_skill_threshold": PRIMARY_SKILL_THRESHOLD,
        "primary_identification_gauges": len(primary),
        "primary_metrics": metrics,
        "gates": gates,
        "formal_decision": formal_decision,
        "conditional_interpretation": (
            "On the skill-qualified population, cross-architecture magnitude, ET, "
            "observed-recession, class-stability, subgroup, and sensitivity tests "
            "support non-identification. The prespecified global verdict remains "
            "inconclusive if either constrained architecture fails the global "
            "prediction-competence threshold."
        ),
    }


# ---------------------------------------------------------------------------
# Expanded reference-basin analysis
# ---------------------------------------------------------------------------


def load_hydroclimate(project: Path, primary: pd.DataFrame) -> pd.DataFrame:
    path = require_file(project / "data/interim/gagesii_all_attributes.csv.gz")
    attributes = pd.read_csv(
        path,
        dtype={"STAID": str},
        usecols=["STAID", *ATTRIBUTE_COLUMNS],
    ).rename(columns={"STAID": "GAGE_ID", **ATTRIBUTE_COLUMNS})
    attributes["GAGE_ID"] = normalize_ids(attributes.GAGE_ID)
    frame = primary.merge(attributes, on="GAGE_ID", how="left", validate="one_to_one")
    frame["log10_area_km2"] = np.log10(frame.area_km2.clip(lower=0.01))
    columns = [*ATTRIBUTE_COLUMNS.values(), "log10_area_km2"]
    if frame[columns].isna().any().any():
        count = int(frame[columns].isna().any(axis=1).sum())
        raise RuntimeError(f"Hydroclimate attributes are missing for {count} primary gauges")
    return frame


def robust_scaled(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = frame[columns].astype(float).copy()
    median = scaled.median()
    spread = scaled.quantile(0.75) - scaled.quantile(0.25)
    spread = spread.mask(spread <= 0, scaled.std()).replace(0, 1.0).fillna(1.0)
    return (scaled - median) / spread


def optimal_match(
    frame: pd.DataFrame,
    columns: list[str],
    design: str,
    feature_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Match every reference gauge to one unique control within ecoregion/tier."""
    scaled = robust_scaled(frame, columns)
    feature_weights = feature_weights or {}
    weights = np.asarray([feature_weights.get(column, 1.0) for column in columns])
    records = []
    for (ecoregion, tier), stratum in frame.groupby(
        ["ecoregion", "tier_short"], sort=True
    ):
        reference = stratum.loc[stratum.reference_status.eq("Reference")]
        control = stratum.loc[stratum.reference_status.eq("Non-reference")]
        if reference.empty:
            continue
        if len(control) < len(reference):
            raise RuntimeError(f"Insufficient controls in {ecoregion}, {tier}")
        x = scaled.loc[reference.index, columns].to_numpy(float)
        y = scaled.loc[control.index, columns].to_numpy(float)
        cost = np.sqrt((((x[:, None, :] - y[None, :, :]) * weights) ** 2).sum(axis=2))
        rows, columns_selected = linear_sum_assignment(cost)
        for row, column in zip(rows, columns_selected, strict=True):
            records.append({
                "design": design,
                "ecoregion": ecoregion,
                "tier_short": tier,
                "reference_id": reference.iloc[row].GAGE_ID,
                "control_id": control.iloc[column].GAGE_ID,
                "distance": float(cost[row, column]),
            })
    matched = pd.DataFrame(records)
    expected = int(frame.reference_status.eq("Reference").sum())
    if len(matched) != expected or matched.control_id.nunique() != expected:
        raise RuntimeError("Matching did not produce one unique control per reference gauge")
    return matched


def standardized_difference(reference: pd.Series, control: pd.Series) -> float:
    reference = reference.astype(float)
    control = control.astype(float)
    pooled = np.sqrt((reference.var(ddof=1) + control.var(ddof=1)) / 2.0)
    return float((reference.mean() - control.mean()) / pooled) if pooled > 0 else 0.0


def balance_table(
    frame: pd.DataFrame, pairs: pd.DataFrame, columns: list[str], design: str
) -> list[dict]:
    indexed = frame.set_index("GAGE_ID", drop=False)
    reference_all = frame.loc[frame.reference_status.eq("Reference")]
    control_all = frame.loc[frame.reference_status.eq("Non-reference")]
    reference_matched = indexed.loc[pairs.reference_id].reset_index(drop=True)
    control_matched = indexed.loc[pairs.control_id].reset_index(drop=True)
    return [{
        "design": design,
        "feature": column,
        "smd_before_matching": standardized_difference(
            reference_all[column], control_all[column]
        ),
        "smd_after_matching": standardized_difference(
            reference_matched[column], control_matched[column]
        ),
        "reference_mean": float(reference_matched[column].mean()),
        "control_mean": float(control_matched[column].mean()),
    } for column in columns]


def bootstrap_unmatched_contrast(
    reference: pd.DataFrame,
    control: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    reference = reference.reset_index(drop=True)
    control = control.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(draws):
        reference_metrics = calculate_metrics(
            reference.loc[stratified_indices(reference, rng)]
        )
        control_metrics = calculate_metrics(control.loc[stratified_indices(control, rng)])
        record = {"draw": draw, "comparison": "unmatched_reference_minus_non_reference"}
        for metric in METRIC_COLUMNS:
            record[metric] = reference_metrics[metric] - control_metrics[metric]
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_matched_contrast(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    indexed = frame.set_index("GAGE_ID", drop=False)
    groups = [group.index.to_numpy() for _, group in pairs.groupby("ecoregion", sort=True)]
    records = []
    for draw in range(draws):
        chosen = np.concatenate([
            rng.choice(index, size=len(index), replace=True) for index in groups
        ])
        sample = pairs.loc[chosen]
        reference = indexed.loc[sample.reference_id].reset_index(drop=True)
        control = indexed.loc[sample.control_id].reset_index(drop=True)
        reference_metrics = calculate_metrics(reference)
        control_metrics = calculate_metrics(control)
        record = {
            "draw": draw,
            "comparison": f"{pairs.design.iloc[0]}_reference_minus_control",
        }
        for metric in METRIC_COLUMNS:
            record[metric] = reference_metrics[metric] - control_metrics[metric]
        records.append(record)
    return pd.DataFrame(records)


def contrast_summary(
    name: str,
    reference_metrics: dict,
    control_metrics: dict,
    bootstrap: pd.DataFrame,
) -> list[dict]:
    records = []
    for metric in METRIC_COLUMNS:
        lo, hi = bootstrap[metric].quantile([0.025, 0.975])
        records.append({
            "comparison": name,
            "metric": metric,
            "reference_value": reference_metrics[metric],
            "control_value": control_metrics[metric],
            "reference_minus_control": (
                reference_metrics[metric] - control_metrics[metric]
            ),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "interval_excludes_zero": bool((lo > 0) or (hi < 0)),
        })
    return records


def matched_metrics(frame: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict, dict]:
    indexed = frame.set_index("GAGE_ID", drop=False)
    reference = indexed.loc[pairs.reference_id].reset_index(drop=True)
    control = indexed.loc[pairs.control_id].reset_index(drop=True)
    return calculate_metrics(reference), calculate_metrics(control)


def reference_skill_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for threshold in SKILL_THRESHOLDS:
        eligible = (
            frame.observed_usable
            & frame.dhbv_qet_kge.ge(threshold)
            & frame.fluxgate_qet_kge.ge(threshold)
        )
        for population in ("Reference", "Non-reference"):
            subset = frame.loc[eligible & frame.reference_status.eq(population)]
            records.append({
                "skill_threshold": threshold,
                "population": population,
                **calculate_metrics(subset),
            })
    return pd.DataFrame(records)


def predictive_population_summary(frame: pd.DataFrame) -> pd.DataFrame:
    populations = {
        "All targets": np.ones(len(frame), dtype=bool),
        "Reference": frame.reference_status.eq("Reference"),
        "Non-reference": frame.reference_status.eq("Non-reference"),
        "Tier 1": frame.tier_short.eq("Tier 1"),
        "Tier 2": frame.tier_short.eq("Tier 2"),
        "Reference Tier 1": (
            frame.reference_status.eq("Reference") & frame.tier_short.eq("Tier 1")
        ),
        "Reference Tier 2": (
            frame.reference_status.eq("Reference") & frame.tier_short.eq("Tier 2")
        ),
    }
    records = []
    for population, mask in populations.items():
        subset = frame.loc[mask]
        joint = (
            subset.observed_usable
            & subset.dhbv_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
            & subset.fluxgate_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
        )
        records.append({
            "population": population,
            "target_gauges": len(subset),
            "observed_usable_gauges": int(subset.observed_usable.sum()),
            "primary_gauges": int(joint.sum()),
            "dhbv_median_kge": float(subset.dhbv_qet_kge.median()),
            "dhbv_fraction_kge_ge_030": float(
                subset.dhbv_qet_kge.ge(PRIMARY_SKILL_THRESHOLD).mean()
            ),
            "fluxgate_median_kge": float(subset.fluxgate_qet_kge.median()),
            "fluxgate_fraction_kge_ge_030": float(
                subset.fluxgate_qet_kge.ge(PRIMARY_SKILL_THRESHOLD).mean()
            ),
            "joint_fraction_kge_ge_030": float(
                (
                    subset.dhbv_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
                    & subset.fluxgate_qet_kge.ge(PRIMARY_SKILL_THRESHOLD)
                ).mean()
            ),
        })
    return pd.DataFrame(records)


def reference_analysis(
    project: Path,
    full: pd.DataFrame,
    primary: pd.DataFrame,
    draws: int,
) -> dict[str, pd.DataFrame]:
    primary = load_hydroclimate(project, primary)
    groups = {
        "All primary": primary,
        "Reference": primary.loc[primary.reference_status.eq("Reference")],
        "Non-reference": primary.loc[primary.reference_status.eq("Non-reference")],
        "Tier 1": primary.loc[primary.tier_short.eq("Tier 1")],
        "Tier 2": primary.loc[primary.tier_short.eq("Tier 2")],
        "Reference Tier 1": primary.loc[
            primary.reference_status.eq("Reference") & primary.tier_short.eq("Tier 1")
        ],
        "Reference Tier 2": primary.loc[
            primary.reference_status.eq("Reference") & primary.tier_short.eq("Tier 2")
        ],
    }
    group_records, group_bootstraps = [], []
    for label, group in groups.items():
        record, bootstrap = metrics_with_ci(label, group, draws, seed=20261001)
        group_records.append(record)
        group_bootstraps.append(bootstrap)

    reference = groups["Reference"].copy().reset_index(drop=True)
    non_reference = groups["Non-reference"].copy().reset_index(drop=True)
    unmatched_bootstrap = bootstrap_unmatched_contrast(
        reference, non_reference, draws, seed=20261002
    )
    contrast_records = contrast_summary(
        "Unmatched reference minus non-reference",
        calculate_metrics(reference),
        calculate_metrics(non_reference),
        unmatched_bootstrap,
    )

    skill_columns = ["dhbv_qet_kge", "fluxgate_qet_kge"]
    hydroclimate_columns = [
        "log10_area_km2", "pptavg_basin", "tavg_basin", "pet",
        "snow_pct_precip", "precip_seasonality", *skill_columns,
    ]
    skill_pairs = optimal_match(primary, skill_columns, "skill_matched")
    hydro_pairs = optimal_match(
        primary,
        hydroclimate_columns,
        "hydroclimate_skill_matched",
        feature_weights=HYDROCLIMATE_MATCH_WEIGHTS,
    )

    contrast_bootstraps = [unmatched_bootstrap]
    balance_records = []
    for name, pairs, columns in (
        ("Skill-matched reference minus control", skill_pairs, skill_columns),
        (
            "Hydroclimate-and-skill-matched reference minus control",
            hydro_pairs,
            hydroclimate_columns,
        ),
    ):
        reference_metrics, control_metrics = matched_metrics(primary, pairs)
        bootstrap = bootstrap_matched_contrast(primary, pairs, draws, seed=20261003)
        contrast_records.extend(
            contrast_summary(name, reference_metrics, control_metrics, bootstrap)
        )
        contrast_bootstraps.append(bootstrap)
        balance_records.extend(balance_table(primary, pairs, columns, pairs.design.iloc[0]))

    return {
        "group_metrics": pd.DataFrame(group_records),
        "group_bootstrap": pd.concat(group_bootstraps, ignore_index=True),
        "contrasts": pd.DataFrame(contrast_records),
        "contrast_bootstrap": pd.concat(contrast_bootstraps, ignore_index=True),
        "matched_pairs": pd.concat([skill_pairs, hydro_pairs], ignore_index=True),
        "match_balance": pd.DataFrame(balance_records),
        "skill_sensitivity": reference_skill_sensitivity(full),
        "predictive_population": predictive_population_summary(full),
    }


def compare_with_archived_results(
    project: Path,
    national: dict,
    reference: dict[str, pd.DataFrame],
) -> dict:
    """Compare point estimates only; archived results are never analysis inputs."""
    report: dict[str, dict | bool] = {}
    national_path = project / "outputs/identification/identification_gates.json"
    if national_path.exists():
        archived = json.loads(national_path.read_text(encoding="utf-8"))
        differences = {
            metric: abs(
                float(national["primary_metrics"][metric])
                - float(archived["primary_metrics"][metric])
            )
            for metric in METRIC_COLUMNS
        }
        report["national_point_estimates"] = {
            "available": True,
            "maximum_absolute_difference": max(differences.values()),
            "differences": differences,
        }
    else:
        report["national_point_estimates"] = {"available": False}

    reference_path = (
        project
        / "outputs/identification/reference_expansion/reference_group_metrics_with_ci.csv"
    )
    if reference_path.exists():
        archived = pd.read_csv(reference_path).set_index("population")
        rebuilt = reference["group_metrics"].set_index("population")
        common = archived.index.intersection(rebuilt.index)
        differences = np.abs(
            archived.loc[common, list(METRIC_COLUMNS)].to_numpy(float)
            - rebuilt.loc[common, list(METRIC_COLUMNS)].to_numpy(float)
        )
        report["reference_group_point_estimates"] = {
            "available": True,
            "populations_compared": len(common),
            "maximum_absolute_difference": float(np.nanmax(differences)),
        }
    else:
        report["reference_group_point_estimates"] = {"available": False}
    report["point_estimates_reproduced"] = bool(
        all(
            item.get("maximum_absolute_difference", 0.0) <= 1e-12
            for item in report.values()
            if isinstance(item, dict) and item.get("available")
        )
    )
    return report


def output_checksums(output_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_dir)): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "output_checksums.json"
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 10:
        raise ValueError("Use at least 10 bootstrap draws; the paper uses 1000")
    project = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project / "audit/identification_reanalysis_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = output_dir / "reconstructed_inputs"
    derived_dir.mkdir(parents=True, exist_ok=True)
    verify_checksums = not args.skip_checksums

    registry, integrity_log = load_registry(project, verify_checksums)
    target = load_target_stations(project)
    if len(target) != int(registry["target_gauges"]):
        raise ValueError("Target count differs from the frozen registry")

    q: np.ndarray | None = None
    if args.recompute_skill or args.recompute_observed_recessions:
        q, q_audit = load_target_discharge(project, target, verify_checksums)
        integrity_log.append(q_audit)

    if args.recompute_skill:
        assert q is not None
        skill = rebuild_skill_metrics(
            project, target, registry, q, integrity_log, verify_checksums
        )
        skill.to_parquet(derived_dir / "target_model_metrics_rebuilt.parquet", index=False)
    else:
        skill_path = require_file(project / "outputs/evaluation/target_model_metrics.parquet")
        skill = pd.read_parquet(skill_path)
        integrity_log.append({
            "asset": str(skill_path.relative_to(project)),
            "expected_sha256": None,
            "observed_sha256": sha256(skill_path) if verify_checksums else "not_computed",
            "matches": None,
        })

    if args.recompute_observed_recessions:
        assert q is not None
        events, observed = rebuild_observed_recessions(project, target, q)
        events.to_parquet(derived_dir / "recession_events_rebuilt.parquet", index=False)
        observed.to_parquet(derived_dir / "recession_summary_rebuilt.parquet", index=False)
    else:
        observed_path = require_file(
            project / "outputs/observed_recessions/recession_summary.parquet"
        )
        observed = pd.read_parquet(observed_path)
        integrity_log.append({
            "asset": str(observed_path.relative_to(project)),
            "expected_sha256": None,
            "observed_sha256": sha256(observed_path) if verify_checksums else "not_computed",
            "matches": None,
        })

    fingerprints, fingerprint_checks = load_fingerprints(
        project,
        registry,
        args.recompute_fingerprints,
        integrity_log,
        verify_checksums,
    )
    if fingerprint_checks:
        pd.DataFrame(fingerprint_checks).to_csv(
            derived_dir / "fingerprint_reconstruction_checks.csv", index=False
        )

    full = assemble_gauge_level(target, fingerprints, skill, observed)
    primary = primary_population(full)
    if len(full) != 3982:
        raise RuntimeError(f"Expected 3,982 target gauges, obtained {len(full)}")
    full.to_parquet(output_dir / "gauge_level_identification.parquet", index=False)
    primary.to_parquet(output_dir / "primary_identification_population.parquet", index=False)

    national_record, national_bootstrap = metrics_with_ci(
        "All primary", primary, args.bootstrap_draws, seed=20260930
    )
    pd.DataFrame([national_record]).to_csv(
        output_dir / "national_metrics_with_ci.csv", index=False
    )
    national_bootstrap.to_csv(
        output_dir / "national_bootstrap.csv.gz", index=False, compression="gzip"
    )
    subgroups = subgroup_analysis(primary)
    sensitivities = sensitivity_analysis(full, observed)
    model_q0 = model_q0_sensitivity(
        project,
        registry,
        target,
        skill,
        observed,
        integrity_log,
        verify_checksums,
    )
    subgroups.to_csv(output_dir / "national_subgroup_metrics.csv", index=False)
    sensitivities.to_csv(output_dir / "national_sensitivity_grid.csv", index=False)
    model_q0.to_csv(output_dir / "national_model_q0_sensitivity.csv", index=False)
    mass_balance = mass_balance_audit(project)
    national = national_gates(
        full,
        primary,
        national_bootstrap,
        subgroups,
        sensitivities,
        model_q0,
        mass_balance,
    )
    write_json(output_dir / "identification_gates.json", national)

    reference = reference_analysis(project, full, primary, args.bootstrap_draws)
    reference["group_metrics"].to_csv(
        output_dir / "reference_group_metrics_with_ci.csv", index=False
    )
    reference["group_bootstrap"].to_csv(
        output_dir / "reference_group_bootstrap.csv.gz", index=False, compression="gzip"
    )
    reference["contrasts"].to_csv(
        output_dir / "reference_contrasts_with_ci.csv", index=False
    )
    reference["contrast_bootstrap"].to_csv(
        output_dir / "reference_contrast_bootstrap.csv.gz", index=False, compression="gzip"
    )
    reference["matched_pairs"].to_csv(
        output_dir / "reference_matched_pairs.csv", index=False
    )
    reference["match_balance"].to_csv(
        output_dir / "reference_match_balance.csv", index=False
    )
    reference["skill_sensitivity"].to_csv(
        output_dir / "reference_skill_sensitivity.csv", index=False
    )
    reference["predictive_population"].to_csv(
        output_dir / "reference_predictive_population.csv", index=False
    )

    comparison = compare_with_archived_results(project, national, reference)
    write_json(output_dir / "comparison_with_archived_results.json", comparison)
    pd.DataFrame(integrity_log).to_csv(output_dir / "input_integrity_log.csv", index=False)

    reference_metrics = reference["group_metrics"].set_index("population").loc["Reference"]
    hydro_contrasts = reference["contrasts"].loc[
        reference["contrasts"].comparison.eq(
            "Hydroclimate-and-skill-matched reference minus control"
        )
    ].set_index("metric")
    maximum_balance_smd = float(
        reference["match_balance"].loc[
            reference["match_balance"].design.eq("hydroclimate_skill_matched"),
            "smd_after_matching",
        ].abs().max()
    )
    summary = {
        "scope": "figure-free inferential reanalysis from frozen model outputs",
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "recomputed_inputs": {
            "prediction_skill": args.recompute_skill,
            "observed_recessions": args.recompute_observed_recessions,
            "intervention_fingerprints": args.recompute_fingerprints,
        },
        "bootstrap_draws": args.bootstrap_draws,
        "target_gauges": len(full),
        "primary_skill_qualified_gauges": len(primary),
        "formal_decision": national["formal_decision"],
        "primary_metrics": national["primary_metrics"],
        "model_q0_sensitivity": model_q0.to_dict(orient="records"),
        "reference_gauges": int(reference_metrics.gauges),
        "reference_metrics": {
            metric: float(reference_metrics[metric]) for metric in METRIC_COLUMNS
        },
        "hydroclimate_and_skill_matched_contrasts": {
            "qet_cross_architecture_mad": float(
                hydro_contrasts.loc[
                    "qet_cross_architecture_mad", "reference_minus_control"
                ]
            ),
            "qet_cross_architecture_mad_ci95": [
                float(hydro_contrasts.loc["qet_cross_architecture_mad", "ci_low"]),
                float(hydro_contrasts.loc["qet_cross_architecture_mad", "ci_high"]),
            ],
            "class_agreement": float(
                hydro_contrasts.loc[
                    "all_model_class_agreement_fraction", "reference_minus_control"
                ]
            ),
            "class_agreement_ci95": [
                float(
                    hydro_contrasts.loc[
                        "all_model_class_agreement_fraction", "ci_low"
                    ]
                ),
                float(
                    hydro_contrasts.loc[
                        "all_model_class_agreement_fraction", "ci_high"
                    ]
                ),
            ],
            "maximum_absolute_covariate_smd": maximum_balance_smd,
        },
        "archived_point_estimates_reproduced": comparison["point_estimates_reproduced"],
        "interpretation_boundary": (
            "Reference status and matching are robustness analyses, not randomized "
            "causal contrasts. Persistence among reference gauges shows that the "
            "conditional non-identification result does not depend on retaining "
            "more heavily modified gauges."
        ),
    }
    write_json(output_dir / "audit_summary.json", summary)
    write_json(output_dir / "output_checksums.json", output_checksums(output_dir))

    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"\nAudit outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
