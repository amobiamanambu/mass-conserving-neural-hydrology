#!/usr/bin/env python3
"""Translate archived response disagreement into hydrologic time-scale metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from r7_diagnostic import persistence_class


PROJECT = Path(__file__).resolve().parents[1]
STRESS = PROJECT / "multimodel_neural_conservation_stress_test_2026-10-03"
OUTPUT = PROJECT / "outputs" / "identification"
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr")
METRICS = ("median_r7", "median_r30", "median_half_life_days")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_path(architecture: str, seed: int) -> Path:
    if architecture in {"dhbv", "fluxgate"}:
        return PROJECT / "outputs" / "law_interventions" / f"{architecture}_qet_seed{seed}_fingerprints.parquet"
    return STRESS / "outputs" / "interventions" / f"{architecture}_qet_seed{seed}_model_specific_fingerprints.parquet"


def ensemble_fingerprint(architecture: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = [fingerprint_path(architecture, seed) for seed in SEEDS]
    frames = [
        pd.read_parquet(path, columns=["GAGE_ID", *METRICS])
        .assign(GAGE_ID=lambda frame: frame.GAGE_ID.astype(str))
        .set_index("GAGE_ID")
        for path in paths
    ]
    index = frames[0].index
    output = pd.DataFrame(index=index)
    for metric in METRICS:
        values = np.column_stack([frame.reindex(index)[metric].to_numpy(float) for frame in frames])
        output[metric] = np.nanmedian(values, axis=1)
    return output, paths


def summarize_pair(
    gauge: pd.DataFrame,
    fingerprints: dict[str, pd.DataFrame],
    first: str,
    second: str,
    threshold: float,
) -> dict[str, float | int | str]:
    eligible = (
        gauge.observed_usable.astype(bool)
        & gauge[f"{first}_kge"].ge(threshold)
        & gauge[f"{second}_kge"].ge(threshold)
    )
    ids = gauge.loc[eligible, "GAGE_ID"].astype(str)
    a = fingerprints[first].reindex(ids)
    b = fingerprints[second].reindex(ids)
    complete = a.notna().all(axis=1) & b.notna().all(axis=1)
    a, b = a.loc[complete], b.loc[complete]
    half_life_gap = np.abs(a.median_half_life_days - b.median_half_life_days)
    different_class = persistence_class(a.median_r7) != persistence_class(b.median_r7)
    return {
        "first": first,
        "second": second,
        "skill_threshold": threshold,
        "n": len(a),
        "median_absolute_r7_difference": float(np.median(np.abs(a.median_r7 - b.median_r7))),
        "median_absolute_r30_difference": float(np.median(np.abs(a.median_r30 - b.median_r30))),
        "first_median_r30": float(np.median(a.median_r30)),
        "second_median_r30": float(np.median(b.median_r30)),
        "median_absolute_half_life_difference_days": float(np.median(half_life_gap)),
        "half_life_difference_q25_days": float(np.quantile(half_life_gap, 0.25)),
        "half_life_difference_q75_days": float(np.quantile(half_life_gap, 0.75)),
        "first_median_half_life_days": float(np.median(a.median_half_life_days)),
        "second_median_half_life_days": float(np.median(b.median_half_life_days)),
        "different_persistence_class_fraction": float(np.mean(different_class)),
    }


def main() -> None:
    gauge_path = STRESS / "outputs" / "comparison" / "gauge_level_multimodel.parquet"
    gauge = pd.read_parquet(gauge_path).assign(GAGE_ID=lambda frame: frame.GAGE_ID.astype(str))
    fingerprints: dict[str, pd.DataFrame] = {}
    inputs = [gauge_path]
    for architecture in ARCHITECTURES:
        fingerprints[architecture], paths = ensemble_fingerprint(architecture)
        inputs.extend(paths)

    rows = []
    for first_index, first in enumerate(ARCHITECTURES):
        for second in ARCHITECTURES[first_index + 1 :]:
            for threshold in (0.30, 0.60):
                rows.append(summarize_pair(gauge, fingerprints, first, second, threshold))
    summary = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "practical_response_metrics.csv"
    receipt_path = OUTPUT / "practical_response_metrics_receipt.json"
    summary.to_csv(summary_path, index=False)
    receipt = {
        "description": "Descriptive time-scale translation of frozen five-model response fingerprints",
        "ensemble_operator": "median across seeds",
        "population_rule": "observed R7 usable and both architecture KGE values at or above threshold",
        "half_life_rule": "first elapsed day with Q(t) no greater than 0.5 Q(0); 91 days if not crossed",
        "inputs": {str(path.relative_to(PROJECT)): sha256(path) for path in inputs},
        "output": str(summary_path.relative_to(PROJECT)),
        "output_sha256": sha256(summary_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(receipt_path)


if __name__ == "__main__":
    main()
