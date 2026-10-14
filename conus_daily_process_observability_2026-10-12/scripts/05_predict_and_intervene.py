#!/usr/bin/env python3
"""Evaluate frozen runs and create common-anchor drainage-response curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
WEEKLY_SOURCE = ROOT / "conus_hydrologic_law_identifiability_2026-09-30/data/interim/weekly_model_data.npz"
QUANTILES = np.linspace(0.05, 0.95, 12, dtype=np.float64)
Q0_THRESHOLD = 0.001


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--intervention-batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o, p = observed[valid], predicted[valid]
    if np.std(o) <= 0 or np.mean(o) <= 0:
        return np.nan
    r = np.corrcoef(o, p)[0, 1]
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(p) / np.mean(o)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o, p = observed[valid], predicted[valid]
    denominator = np.sum((o - np.mean(o)) ** 2)
    return float(1.0 - np.sum((o - p) ** 2) / denominator) if denominator > 0 else np.nan


def correlation(observed: np.ndarray, predicted: np.ndarray, minimum: int = 100) -> tuple[float, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < minimum or np.std(observed[valid]) <= 0 or np.std(predicted[valid]) <= 0:
        return np.nan, np.nan
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1]), float(spearmanr(observed[valid], predicted[valid]).statistic)


def model_soil_fraction(model: torch.nn.Module, state: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    if model.architecture == "dhbv":
        capacity = model.parameters_from_static(static)["capacity"]
    elif model.architecture == "fluxgate":
        _, terms = model.static_terms(static)
        capacity = terms["capacity"]
    else:
        raise ValueError(model.architecture)
    return state[..., 1] / capacity[:, None]


def select_nearest_unique(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    targets = np.quantile(values[candidates], QUANTILES)
    used: set[int] = set()
    selected = []
    for target in targets:
        order = candidates[np.argsort(np.abs(values[candidates] - target), kind="stable")]
        pick = next((int(index) for index in order if int(index) not in used), int(order[0]))
        selected.append(pick); used.add(pick)
    return np.asarray(selected, np.int16)


def build_forcing_anchors(
    target: pd.DataFrame, forcing: np.ndarray, dates: np.ndarray, eval_start: int, eval_end: int,
) -> Path:
    path = PROJECT / "data/frozen/forcing_anchor_indices.npz"
    if path.exists():
        return path
    rows = target.row.to_numpy(int)
    wetness = np.zeros(len(rows), np.float64)
    eval_wetness = np.empty((len(rows), eval_end - eval_start), np.float32)
    for day in range(eval_end):
        local = np.asarray(forcing[rows, day], np.float64)
        wetness = 0.97 * wetness + local[:, 0] - local[:, 1]
        if day >= eval_start:
            eval_wetness[:, day - eval_start] = wetness
    eval_dates = dates[eval_start:eval_end]
    dt = pd.DatetimeIndex(eval_dates.astype("datetime64[ns]"))
    warm_calendar = (dt.month >= 6) & (dt.month <= 9)
    indices = np.empty((len(rows), len(QUANTILES)), np.int16)
    relaxation = np.zeros(len(rows), np.int8)
    for basin, row in enumerate(rows):
        temperature = np.asarray(forcing[row, eval_start:eval_end, 2])
        candidates = np.flatnonzero(warm_calendar & (temperature > 5.0) & np.isfinite(eval_wetness[basin]))
        if len(candidates) < len(QUANTILES):
            candidates = np.flatnonzero(warm_calendar & np.isfinite(eval_wetness[basin]))
            relaxation[basin] = 1
        indices[basin] = select_nearest_unique(eval_wetness[basin], candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, gage_ids=target.GAGE_ID.to_numpy(str), quantiles=QUANTILES.astype(np.float32),
        eval_time_index=indices, relaxation_code=relaxation,
        wetness_definition=np.asarray("W_t = 0.97 W_(t-1) + P_t - PET_t"),
    )
    return path


def intervention_curves(
    model: torch.nn.Module, static: np.ndarray, target_rows: np.ndarray,
    states: np.ndarray, anchor_time: np.ndarray, eval_dates: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    original_dtype = next(model.parameters()).dtype
    model = model.double().eval()
    n_gauges, n_anchors, state_size = states.shape
    basin_flat = np.repeat(np.arange(n_gauges), n_anchors)
    time_flat = anchor_time.reshape(-1)
    state_flat = states.reshape(-1, state_size).astype(np.float64)
    normalized = np.full((n_gauges * n_anchors, 91), np.nan, np.float32)
    residual_max = 0.0
    cumulative_residual_max = 0.0
    with torch.no_grad():
        for first in range(0, len(state_flat), batch_size):
            stop = min(first + batch_size, len(state_flat))
            base_date = eval_dates[time_flat[first:stop]].astype("datetime64[D]")
            experiment_date = base_date[:, None] + np.arange(1, 92).astype("timedelta64[D]")[None, :]
            doy = pd.DatetimeIndex(experiment_date.reshape(-1).astype("datetime64[ns]")).dayofyear.to_numpy().reshape(stop - first, 91)
            experiment = np.zeros((stop - first, 91, 5), np.float64)
            experiment[:, :, 2] = 10.0
            experiment[:, :, 3] = np.sin(2 * np.pi * doy / 365.25)
            experiment[:, :, 4] = np.cos(2 * np.pi * doy / 365.25)
            rows = target_rows[basin_flat[first:stop]]
            result = model(
                torch.from_numpy(experiment), torch.from_numpy(np.asarray(static[rows], np.float64)),
                initial_state=torch.from_numpy(state_flat[first:stop]), return_states=True,
            )
            q = result["q"].cpu().numpy()
            q0 = q[:, 0]
            valid = np.isfinite(q0) & (q0 >= Q0_THRESHOLD)
            local = np.full_like(q, np.nan, dtype=np.float32)
            local[valid] = (q[valid] / q0[valid, None]).astype(np.float32)
            normalized[first:stop] = local
            residual = result["daily_residual"].cpu().numpy()
            residual_max = max(residual_max, float(np.max(np.abs(residual))))
            cumulative_residual_max = max(
                cumulative_residual_max, float(np.max(np.abs(np.cumsum(residual, axis=1))))
            )
    normalized = normalized.reshape(n_gauges, n_anchors, 91)
    with np.errstate(all="ignore"):
        median_curve = np.nanmedian(normalized, axis=1).astype(np.float32)
    valid_anchors = np.isfinite(normalized[:, :, 0]).sum(axis=1).astype(np.int8)
    model.to(dtype=original_dtype)
    return median_curve, valid_anchors, residual_max, cumulative_residual_max


def development_metrics(
    model: torch.nn.Module, stations: pd.DataFrame, dates: np.ndarray,
    forcing: np.ndarray, static: np.ndarray, sm: np.ndarray, batch_size: int,
) -> pd.DataFrame:
    development = stations[stations.role.eq("development")].reset_index(drop=True)
    start = int(np.searchsorted(dates, np.datetime64("2015-01-01")))
    stop = int(np.searchsorted(dates, np.datetime64("2018-12-31"), side="right"))
    et_archive = np.load(PROJECT / "data/model_ready/ssebop_monthly.npz")
    et_ids = et_archive["gage_ids"].astype(str)
    et_lookup = {gage: row for row, gage in enumerate(et_ids)}
    et_rows = np.asarray([et_lookup[gage] for gage in development.GAGE_ID], int)
    block_start = et_archive["block_start"]
    block_stop = et_archive["block_stop"]
    blocks = np.flatnonzero((block_start >= start) & (block_stop <= stop))
    observed_et = et_archive["eta_month_mm"][np.ix_(et_rows, blocks)]
    rows_out = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(development), batch_size):
            local = development.iloc[first:first + batch_size]
            rows = local.row.to_numpy(int)
            s = torch.from_numpy(np.asarray(static[rows])).float()
            out = model(
                torch.from_numpy(np.asarray(forcing[rows, :stop])).float(), s, return_states=True,
            )
            predicted = model_soil_fraction(model, out["state"][:, start:stop], s).cpu().numpy()
            predicted_et = torch.stack([
                out["et"][:, block_start[block]:block_stop[block]].sum(dim=1)
                for block in blocks
            ], dim=1).cpu().numpy()
            observed = np.asarray(sm[rows, start:stop])
            for i, station in enumerate(local.itertuples(index=False)):
                pearson, rank = correlation(observed[i], predicted[i])
                et_observed = observed_et[first + i]
                et_predicted = predicted_et[i]
                valid_et = np.isfinite(et_observed) & np.isfinite(et_predicted)
                et_rank = correlation(et_observed, et_predicted, minimum=20)[1]
                rows_out.append({
                    "GAGE_ID": station.GAGE_ID, "role": "development",
                    "soil_pearson": pearson, "soil_spearman": rank,
                    "et_mae_month_mm": float(np.mean(np.abs(et_predicted[valid_et] - et_observed[valid_et])))
                    if valid_et.any() else np.nan,
                    "et_spearman": et_rank,
                })
    return pd.DataFrame(rows_out)


def main() -> None:
    args = arguments()
    torch.set_num_threads(args.threads)
    run_dir = PROJECT / "runs/frozen_refit" / args.run
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output = PROJECT / "outputs" / args.run
    receipt_path = output / "receipt.json"
    if receipt_path.exists() and not args.force:
        print(f"{args.run}: already complete")
        return
    output.mkdir(parents=True, exist_ok=True)

    base = SOURCE / "data/processed"
    stations = pd.read_csv(base / "station_index.csv", dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    target = stations[stations.role.eq("target")].reset_index(drop=True)
    target_rows = target.row.to_numpy(int)
    dates = np.load(base / "dates.npy", mmap_mode="r")
    forcing = np.load(base / "forcing.npy", mmap_mode="r")
    static = np.load(base / "static.npy", mmap_mode="r")
    sm = np.load(PROJECT / "data/model_ready/nldas_rootzone_saturation.npy", mmap_mode="r")
    observed_q = np.load(SOURCE / "data/evaluation/q_target_2019_2024.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    if observed_q.shape != (len(target), len(eval_dates)):
        raise RuntimeError("Target-Q shape mismatch")
    weekly = np.load(WEEKLY_SOURCE)
    weekly_dates = weekly["dates"].astype("datetime64[D]")
    weekly_time = np.flatnonzero(
        (weekly_dates >= eval_dates[0]) & (weekly_dates <= eval_dates[-1])
    )
    model_weekly_time = np.searchsorted(eval_dates, weekly_dates[weekly_time])
    if not np.array_equal(eval_dates[model_weekly_time], weekly_dates[weekly_time]):
        raise RuntimeError("ESA CCI weekly dates do not align with the daily model calendar")
    weekly_ids = weekly["gage_ids"].astype(str)
    weekly_lookup = {gage: row for row, gage in enumerate(weekly_ids)}
    esa_sm = weekly["esa_sm_m3m3"][:, weekly_time]
    anchor_path = build_forcing_anchors(target, forcing, dates, eval_start, eval_end)
    anchors = np.load(anchor_path)
    if anchors["gage_ids"].astype(str).tolist() != target.GAGE_ID.tolist():
        raise RuntimeError("Forcing-anchor gauge order mismatch")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = make_model(architecture, static.shape[1])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    target_records = []
    anchor_states = np.empty((len(target), len(QUANTILES), model.state_size), np.float32)
    with torch.no_grad():
        for first in range(0, len(target), args.batch_size):
            stop = min(first + args.batch_size, len(target))
            rows = target_rows[first:stop]
            s = torch.from_numpy(np.asarray(static[rows])).float()
            result = model(
                torch.from_numpy(np.asarray(forcing[rows, :eval_end])).float(), s, return_states=True,
            )
            prediction = result["q"][:, eval_start:eval_end].cpu().numpy()
            soil = model_soil_fraction(model, result["state"][:, eval_start:eval_end], s).cpu().numpy()
            state = result["state"][:, eval_start:eval_end].cpu().numpy()
            for i, station in enumerate(target.iloc[first:stop].itertuples(index=False)):
                pearson, rank = correlation(np.asarray(sm[rows[i], eval_start:eval_end]), soil[i])
                esa_row = weekly_lookup.get(station.GAGE_ID)
                if esa_row is None:
                    esa_pearson, esa_rank, esa_count = np.nan, np.nan, 0
                else:
                    esa_observed = np.asarray(esa_sm[esa_row])
                    esa_predicted = soil[i, model_weekly_time]
                    esa_count = int(np.sum(np.isfinite(esa_observed) & np.isfinite(esa_predicted)))
                    esa_pearson, esa_rank = correlation(esa_observed, esa_predicted, minimum=30)
                target_records.append({
                    "GAGE_ID": station.GAGE_ID, "architecture": architecture,
                    "rung": checkpoint.get("rung", "q"), "seed": int(checkpoint["seed"]),
                    "kge": kge(np.asarray(observed_q[first + i]), prediction[i]),
                    "nse": nse(np.asarray(observed_q[first + i]), prediction[i]),
                    "soil_pearson": pearson, "soil_spearman": rank,
                    "esa_cci_soil_pearson": esa_pearson,
                    "esa_cci_soil_spearman": esa_rank,
                    "esa_cci_valid_weeks": esa_count,
                })
                anchor_states[first + i] = state[i, anchors["eval_time_index"][first + i]]
            print(f"{args.run}: target trajectories {stop:,}/{len(target):,}", flush=True)

    target_metrics_path = output / "target_metrics.parquet"
    pd.DataFrame(target_records).to_parquet(target_metrics_path, index=False)
    development = development_metrics(model, stations, dates, forcing, static, sm, args.batch_size)
    development["architecture"] = architecture
    development["rung"] = checkpoint.get("rung", "q")
    development["seed"] = int(checkpoint["seed"])
    development_path = output / "development_process_metrics.parquet"
    development.to_parquet(development_path, index=False)

    median_curve, valid_anchors, residual, cumulative_residual = intervention_curves(
        model, static, target_rows, anchor_states,
        anchors["eval_time_index"], eval_dates, args.intervention_batch_size,
    )
    curve_path = output / "drainage_curves.npz"
    np.savez_compressed(
        curve_path, gage_ids=target.GAGE_ID.to_numpy(str), days=np.arange(91, dtype=np.int16),
        median_normalized_q=median_curve, valid_anchors=valid_anchors,
        r7=median_curve[:, 7], r14=median_curve[:, 14], r30=median_curve[:, 30],
        anchor_quantiles=QUANTILES.astype(np.float32),
        forcing_anchor_file_sha256=np.asarray(sha256(anchor_path)),
    )
    receipt = {
        "run": args.run, "architecture": architecture,
        "rung": checkpoint.get("rung", "q"), "seed": int(checkpoint["seed"]),
        "target_gauges": len(target), "target_discharge_accessed_after_model_freeze": True,
        "common_forcing_anchors": str(anchor_path.relative_to(PROJECT)),
        "anchor_definition": "W_t = 0.97 W_(t-1) + P_t - PET_t; 12 warm-season quantiles",
        "intervention": "P=PET=0, T=10 C, 91 days",
        "maximum_float64_daily_mass_residual_mm": residual,
        "maximum_float64_cumulative_mass_residual_mm": cumulative_residual,
        "model_checkpoint_sha256": sha256(checkpoint_path),
        "target_discharge_sha256": sha256(SOURCE / "data/evaluation/q_target_2019_2024.npy"),
        "nldas_evaluation_target_sha256": sha256(PROJECT / "data/model_ready/nldas_rootzone_saturation.npy"),
        "esa_cci_evaluation_archive_sha256": sha256(WEEKLY_SOURCE),
        "target_metrics_sha256": sha256(target_metrics_path),
        "development_process_metrics_sha256": sha256(development_path),
        "drainage_curves_sha256": sha256(curve_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
