#!/usr/bin/env python3
"""Evaluate frozen MCP predictions with the paper's unchanged target metrics."""

from __future__ import annotations

import hashlib
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
evaluation_path = PROJECT / "scripts/06_open_target_discharge_and_evaluate.py"
paper_evaluation = SourceFileLoader("paper_target_evaluation", str(evaluation_path)).load_module()
metrics = paper_evaluation.metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    observed = np.load(PROJECT / "data/evaluation/q_target_2019_2024.npy", mmap_mode="r")
    if observed.shape[0] != len(target):
        raise RuntimeError("Target discharge order/shape mismatch")

    receipts = []
    predictions: dict[str, np.ndarray] = {}
    for path in sorted((EXPERIMENT / "runs/frozen_refit").glob("mcp_*/receipt.json")):
        receipt = json.loads(path.read_text())
        predictions[receipt["run"]] = np.load(
            EXPERIMENT / receipt["prediction"], mmap_mode="r"
        )
        receipts.append(receipt)
    expected = {f"mcp_{rung}_seed{seed}" for rung in ("q", "qet") for seed in (11, 29, 47)}
    if set(predictions) != expected:
        raise RuntimeError(f"Expected six frozen MCP runs, found {sorted(predictions)}")

    records: list[dict] = []
    for receipt in receipts:
        prediction = predictions[receipt["run"]]
        for basin, station in enumerate(target.itertuples(index=False)):
            records.append(
                {
                    "GAGE_ID": station.GAGE_ID,
                    "model": receipt["run"],
                    "architecture": "mcp",
                    "rung": receipt["rung"],
                    "seed": int(receipt["seed"]),
                    "ensemble": False,
                    **metrics(observed[basin], prediction[basin]),
                }
            )
        print(f"metrics: {receipt['run']}", flush=True)

    for rung in ("q", "qet"):
        run_names = [f"mcp_{rung}_seed{seed}" for seed in (11, 29, 47)]
        for basin, station in enumerate(target.itertuples(index=False)):
            prediction = np.mean(
                np.stack([predictions[run][basin] for run in run_names]), axis=0
            )
            records.append(
                {
                    "GAGE_ID": station.GAGE_ID,
                    "model": f"mcp_{rung}_ensemble",
                    "architecture": "mcp",
                    "rung": rung,
                    "seed": -1,
                    "ensemble": True,
                    **metrics(observed[basin], prediction),
                }
            )
        print(f"metrics: mcp_{rung}_ensemble", flush=True)

    result = pd.DataFrame(records).merge(
        target[
            [
                "GAGE_ID",
                "ecoregion",
                "tier_short",
                "reference_status",
                "latitude",
                "longitude",
                "area_km2",
                "distance_independent_50km",
            ]
        ],
        on="GAGE_ID",
        how="left",
        validate="many_to_one",
    )
    output_dir = EXPERIMENT / "outputs/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "target_model_metrics.parquet"
    result.to_parquet(metrics_path, index=False)
    summary = (
        result.groupby(["model", "architecture", "rung", "seed", "ensemble"], as_index=False)
        .agg(
            gauges=("GAGE_ID", "size"),
            median_kge=("kge", "median"),
            q25_kge=("kge", lambda x: x.quantile(0.25)),
            fraction_kge_ge_0=("kge", lambda x: np.mean(x >= 0.0)),
            fraction_kge_ge_03=("kge", lambda x: np.mean(x >= 0.3)),
            fraction_kge_ge_04=("kge", lambda x: np.mean(x >= 0.4)),
            fraction_kge_ge_06=("kge", lambda x: np.mean(x >= 0.6)),
            median_nse=("nse", "median"),
            median_log_nse=("log_nse", "median"),
            median_absolute_percent_bias=(
                "percent_bias", lambda x: np.nanmedian(np.abs(x))
            ),
        )
    )
    summary_path = output_dir / "target_skill_summary.csv"
    summary.to_csv(summary_path, index=False)
    receipt_path = output_dir / "evaluation_receipt.json"
    evaluation_receipt = {
        "target_gauges": int(len(target)),
        "target_days": int(observed.shape[1]),
        "metrics": str(metrics_path.relative_to(EXPERIMENT)),
        "metrics_sha256": sha256(metrics_path),
        "summary": str(summary_path.relative_to(EXPERIMENT)),
        "summary_sha256": sha256(summary_path),
        "ensemble_definition": "arithmetic mean of the three seed hydrographs before metrics",
        "target_discharge_source": str(
            (PROJECT / "data/evaluation/q_target_2019_2024.npy").relative_to(PROJECT)
        ),
    }
    receipt_path.write_text(json.dumps(evaluation_receipt, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(evaluation_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
