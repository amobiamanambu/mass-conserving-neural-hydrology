#!/usr/bin/env python3
"""Evaluate all frozen predictions after the complete prediction registry exists."""

from __future__ import annotations

import hashlib
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

from conserved_models import MODEL_NAMES


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
evaluation_path = PROJECT / "scripts/06_open_target_discharge_and_evaluate.py"
paper_evaluation = SourceFileLoader("paper_target_evaluation", str(evaluation_path)).load_module()
metrics = paper_evaluation.metrics
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    expected = {f"{model}_qet_seed{seed}" for model in MODEL_NAMES for seed in SEEDS}
    receipts, predictions = [], {}
    for path in sorted((EXPERIMENT / "runs/frozen_refit").glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        predictions[receipt["run"]] = np.load(EXPERIMENT / receipt["prediction"], mmap_mode="r")
        receipts.append(receipt)
    if set(predictions) != expected:
        missing = sorted(expected - set(predictions))
        extra = sorted(set(predictions) - expected)
        raise RuntimeError(f"Prediction registry incomplete; missing={missing}, extra={extra}")

    # Target discharge is opened only after the complete 15-run registry check.
    observed = np.load(PROJECT / "data/evaluation/q_target_2019_2024.npy", mmap_mode="r")
    if observed.shape[0] != len(target):
        raise RuntimeError("Target discharge order/shape mismatch")

    records = []
    for receipt in receipts:
        prediction = predictions[receipt["run"]]
        for basin, station in enumerate(target.itertuples(index=False)):
            records.append(
                {
                    "GAGE_ID": station.GAGE_ID,
                    "model": receipt["run"],
                    "architecture": receipt["architecture"],
                    "family": receipt["family"],
                    "seed": int(receipt["seed"]),
                    "ensemble": False,
                    **metrics(observed[basin], prediction[basin]),
                }
            )
        print(f"metrics: {receipt['run']}", flush=True)

    for model in MODEL_NAMES:
        run_names = [f"{model}_qet_seed{seed}" for seed in SEEDS]
        family = next(r["family"] for r in receipts if r["architecture"] == model)
        for basin, station in enumerate(target.itertuples(index=False)):
            prediction = np.mean(
                np.stack([predictions[run][basin] for run in run_names]), axis=0
            )
            records.append(
                {
                    "GAGE_ID": station.GAGE_ID,
                    "model": f"{model}_qet_ensemble",
                    "architecture": model,
                    "family": family,
                    "seed": -1,
                    "ensemble": True,
                    **metrics(observed[basin], prediction),
                }
            )
        print(f"metrics: {model}_qet_ensemble", flush=True)

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
        result.groupby(
            ["model", "architecture", "family", "seed", "ensemble"], as_index=False
        )
        .agg(
            gauges=("GAGE_ID", "size"),
            median_kge=("kge", "median"),
            q25_kge=("kge", lambda x: x.quantile(0.25)),
            fraction_kge_ge_0=("kge", lambda x: np.mean(x >= 0.0)),
            fraction_kge_ge_03=("kge", lambda x: np.mean(x >= 0.3)),
            fraction_kge_ge_04=("kge", lambda x: np.mean(x >= 0.4)),
            fraction_kge_ge_05=("kge", lambda x: np.mean(x >= 0.5)),
            fraction_kge_ge_06=("kge", lambda x: np.mean(x >= 0.6)),
            median_nse=("nse", "median"),
            median_log_nse=("log_nse", "median"),
            median_absolute_percent_bias=(
                "percent_bias", lambda x: np.nanmedian(np.abs(x))
            ),
        )
    )
    summary["passes_global_competence"] = (
        (summary.median_kge >= 0.40) & (summary.fraction_kge_ge_03 >= 0.60)
    )
    summary_path = output_dir / "target_skill_summary.csv"
    summary.to_csv(summary_path, index=False)
    receipt_path = output_dir / "evaluation_receipt.json"
    evaluation_receipt = {
        "target_gauges": int(len(target)),
        "target_days": int(observed.shape[1]),
        "frozen_runs": len(predictions),
        "registry_complete_before_target_open": True,
        "metrics": str(metrics_path.relative_to(EXPERIMENT)),
        "metrics_sha256": sha256(metrics_path),
        "summary": str(summary_path.relative_to(EXPERIMENT)),
        "summary_sha256": sha256(summary_path),
        "ensemble_definition": "arithmetic mean of three seed hydrographs before metrics",
    }
    receipt_path.write_text(json.dumps(evaluation_receipt, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(evaluation_receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
