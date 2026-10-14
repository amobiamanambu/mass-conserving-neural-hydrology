#!/usr/bin/env python3
"""Open target discharge only after the frozen registry, then evaluate models."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DAILY_DIR = ROOT / "continental_run/07_matched_daily/daily"
START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2024-12-31")
DATES = pd.date_range(START, END, freq="D")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_target_q(job: tuple[int, str]) -> tuple[int, np.ndarray]:
    row, gage_id = job
    frame = pd.read_csv(
        DAILY_DIR / f"{gage_id}.csv.gz", usecols=["date", "q_mm_day"], parse_dates=["date"]
    ).drop_duplicates("date").set_index("date").sort_index().reindex(DATES)
    return row, pd.to_numeric(frame.q_mm_day, errors="coerce").to_numpy(np.float32)


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator) if denominator > 0 else np.nan


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return {name: np.nan for name in (
            "kge", "nse", "log_nse", "correlation", "variability_ratio", "mean_ratio",
            "percent_bias", "high_flow_bias_fraction", "low_flow_bias_fraction",
            "observed_zero_fraction", "predicted_zero_fraction", "zero_flow_brier",
        )} | {"valid_days": int(valid.sum())}
    o = observed[valid].astype(np.float64); p = predicted[valid].astype(np.float64)
    o_mean, p_mean = o.mean(), p.mean()
    o_std, p_std = o.std(ddof=1), p.std(ddof=1)
    correlation = float(np.corrcoef(o, p)[0, 1]) if o_std > 0 and p_std > 0 else np.nan
    alpha = float(p_std / o_std) if o_std > 0 else np.nan
    beta = float(p_mean / o_mean) if o_mean > 0 else np.nan
    kge = float(1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)) if np.isfinite([correlation, alpha, beta]).all() else np.nan
    high = o >= np.quantile(o, 0.95)
    low = o <= np.quantile(o, 0.20)

    def volume_bias(mask: np.ndarray) -> float:
        denominator = o[mask].sum()
        return float((p[mask].sum() - denominator) / denominator) if denominator > 1e-8 else np.nan

    obs_zero = o < 0.001; pred_zero = p < 0.001
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
        "observed_zero_fraction": float(obs_zero.mean()),
        "predicted_zero_fraction": float(pred_zero.mean()),
        "zero_flow_brier": float(np.mean((pred_zero.astype(float) - obs_zero.astype(float)) ** 2)),
    }


def main() -> None:
    registry_path = PROJECT / "provenance/frozen_prediction_registry.json"
    checksum_path = PROJECT / "provenance/frozen_prediction_registry.sha256"
    registry = json.loads(registry_path.read_text())
    registered = checksum_path.read_text().split()[0]
    if sha256(registry_path) != registered or registry.get("target_discharge_accessed") is not False:
        raise RuntimeError("Frozen prediction registry failed verification")

    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations[stations.role.eq("target")].reset_index(drop=True)
    evaluation_dir = PROJECT / "data/evaluation"
    output_dir = PROJECT / "outputs/evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    q_path = evaluation_dir / "q_target_2019_2024.npy"
    q = np.lib.format.open_memmap(q_path, mode="w+", dtype=np.float32, shape=(len(target), len(DATES)))
    q[:] = np.nan
    jobs = [(row, gage_id) for row, gage_id in enumerate(target.GAGE_ID)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(read_target_q, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            row, values = future.result(); q[row] = values
            if count % 250 == 0 or count == len(futures):
                print(f"target discharge {count:,}/{len(futures):,}", flush=True)
    q.flush()

    predictions: dict[str, np.ndarray] = {}
    run_metadata: dict[str, dict] = {}
    for run in registry["runs"]:
        predictions[run["run"]] = np.load(PROJECT / run["prediction"], mmap_mode="r")
        run_metadata[run["run"]] = run
    ensemble_groups: dict[tuple[str, str], list[str]] = {}
    for run, metadata in run_metadata.items():
        ensemble_groups.setdefault((metadata["architecture"], metadata["rung"]), []).append(run)

    records = []
    for run, prediction in predictions.items():
        metadata = run_metadata[run]
        for basin, station in enumerate(target.itertuples(index=False)):
            records.append({
                "GAGE_ID": station.GAGE_ID, "model": run,
                "architecture": metadata["architecture"], "rung": metadata["rung"],
                "seed": int(metadata["seed"]), "ensemble": False,
                **metrics(q[basin], prediction[basin]),
            })
        print(f"metrics: {run}", flush=True)

    for (architecture, rung), runs in ensemble_groups.items():
        ensemble_name = f"{architecture}_{rung}_ensemble"
        for basin, station in enumerate(target.itertuples(index=False)):
            # Predictive ensembles average the three seed hydrographs first;
            # gauge metrics are then evaluated on that arithmetic-mean series.
            prediction = np.mean(np.stack([predictions[run][basin] for run in runs]), axis=0)
            records.append({
                "GAGE_ID": station.GAGE_ID, "model": ensemble_name,
                "architecture": architecture, "rung": rung,
                "seed": -1, "ensemble": True,
                **metrics(q[basin], prediction),
            })
        print(f"metrics: {ensemble_name}", flush=True)

    result = pd.DataFrame(records).merge(
        target[["GAGE_ID", "ecoregion", "tier_short", "reference_status", "latitude", "longitude", "area_km2", "distance_independent_50km"]],
        on="GAGE_ID", how="left", validate="many_to_one",
    )
    metrics_path = output_dir / "target_model_metrics.parquet"
    result.to_parquet(metrics_path, index=False)
    summary = (
        result.groupby(["model", "architecture", "rung", "seed", "ensemble"], as_index=False)
        .agg(
            gauges=("GAGE_ID", "size"), median_kge=("kge", "median"), q25_kge=("kge", lambda x: x.quantile(0.25)),
            fraction_kge_ge_0=("kge", lambda x: np.mean(x >= 0.0)),
            fraction_kge_ge_03=("kge", lambda x: np.mean(x >= 0.3)),
            fraction_kge_ge_04=("kge", lambda x: np.mean(x >= 0.4)),
            median_nse=("nse", "median"), median_log_nse=("log_nse", "median"),
            median_absolute_percent_bias=("percent_bias", lambda x: np.nanmedian(np.abs(x))),
        )
    )
    summary_path = output_dir / "target_skill_summary.csv"
    summary.to_csv(summary_path, index=False)
    receipt = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_registry_sha256": registered,
        "target_discharge_accessed": True,
        "target_discharge": str(q_path.relative_to(PROJECT)),
        "target_discharge_sha256": sha256(q_path),
        "target_gauges": len(target),
        "target_days": len(DATES),
        "finite_fraction": float(np.isfinite(q).mean()),
        "metrics": str(metrics_path.relative_to(PROJECT)),
        "metrics_sha256": sha256(metrics_path),
        "summary": str(summary_path.relative_to(PROJECT)),
        "summary_sha256": sha256(summary_path),
    }
    (PROJECT / "provenance/target_evaluation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
