#!/usr/bin/env python3
"""Compare archived float32 interventions with a float64 replay.

This verification does not overwrite the preregistered intervention archive.
It recomputes the primary identification statistics from the replayed
fingerprints and writes a machine-readable reconciliation report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVED = PROJECT / "outputs/law_interventions"
REPLAY = PROJECT / "outputs/law_interventions_float64_audit"
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")
RUNGS = ("q", "qet")
CLASS_BREAKS = (0.5, np.sqrt(0.5))


def rho(x: pd.Series, y: pd.Series) -> float:
    x_values = x.to_numpy(float)
    y_values = y.to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 10 or np.std(x_values[valid]) == 0 or np.std(y_values[valid]) == 0:
        return np.nan
    return float(spearmanr(x_values[valid], y_values[valid]).statistic)


def mad(x: pd.Series, y: pd.Series) -> float:
    difference = np.abs(x.to_numpy(float) - y.to_numpy(float))
    return float(np.nanmedian(difference))


def retention_class(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    classes = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    classes[valid] = np.digitize(values[valid], CLASS_BREAKS)
    return classes


def primary_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    primary = frame.loc[
        frame.observed_usable
        & (frame.dhbv_qet_kge >= 0.30)
        & (frame.fluxgate_qet_kge >= 0.30)
    ].copy()
    classes = np.column_stack(
        [
            retention_class(primary[f"{architecture}_qet_seed{seed}_r7"])
            for architecture in ARCHITECTURES
            for seed in SEEDS
        ]
    )
    complete = np.isfinite(classes).all(axis=1)
    class_agreement = np.full(len(primary), np.nan, dtype=float)
    class_agreement[complete] = np.all(
        classes[complete] == classes[complete][:, [0]], axis=1
    )
    q_mad = mad(primary.dhbv_q_r7, primary.fluxgate_q_r7)
    qet_mad = mad(primary.dhbv_qet_r7, primary.fluxgate_qet_r7)
    return {
        "gauges": int(len(primary)),
        "q_cross_architecture_rho": rho(primary.dhbv_q_r7, primary.fluxgate_q_r7),
        "q_cross_architecture_mad": q_mad,
        "qet_cross_architecture_rho": rho(primary.dhbv_qet_r7, primary.fluxgate_qet_r7),
        "qet_cross_architecture_mad": qet_mad,
        "disagreement_reduction_fraction": float(1.0 - qet_mad / q_mad),
        "dhbv_observed_rho": rho(primary.dhbv_qet_r7, primary.observed_median_r7),
        "dhbv_observed_mae": mad(primary.dhbv_qet_r7, primary.observed_median_r7),
        "fluxgate_observed_rho": rho(primary.fluxgate_qet_r7, primary.observed_median_r7),
        "fluxgate_observed_mae": mad(primary.fluxgate_qet_r7, primary.observed_median_r7),
        "all_model_class_agreement_fraction": float(np.nanmean(class_agreement)),
        "complete_class_rows": int(complete.sum()),
    }


def main() -> None:
    frame = pd.read_parquet(PROJECT / "outputs/identification/gauge_level_identification.parquet")
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    run_rows = []
    replay_residuals = []

    for architecture in ARCHITECTURES:
        for rung in RUNGS:
            seed_columns = []
            for seed in SEEDS:
                run = f"{architecture}_{rung}_seed{seed}"
                old = pd.read_parquet(ARCHIVED / f"{run}_fingerprints.parquet")
                new = pd.read_csv(REPLAY / f"{run}_fingerprints.csv", dtype={"GAGE_ID": str})
                old["GAGE_ID"] = old.GAGE_ID.astype(str).str.zfill(8)
                new["GAGE_ID"] = new.GAGE_ID.astype(str).str.zfill(8)
                comparison = old.merge(new, on="GAGE_ID", suffixes=("_float32", "_float64"), validate="one_to_one")
                row = {"run": run, "gauges": int(len(comparison))}
                for metric in ("median_r7", "median_r30", "median_r90", "median_half_life_days"):
                    difference = np.abs(
                        comparison[f"{metric}_float64"].to_numpy(float)
                        - comparison[f"{metric}_float32"].to_numpy(float)
                    )
                    row[f"median_abs_{metric}_difference"] = float(np.nanmedian(difference))
                    row[f"max_abs_{metric}_difference"] = float(np.nanmax(difference))
                run_rows.append(row)

                column = f"{run}_r7"
                replay = new.set_index("GAGE_ID").reindex(frame.GAGE_ID).median_r7.to_numpy(float)
                frame[column] = replay
                seed_columns.append(column)

                receipt = json.loads((REPLAY / f"{run}_receipt.json").read_text())
                replay_residuals.append(float(receipt["maximum_daily_mass_residual_mm"]))

            frame[f"{architecture}_{rung}_r7"] = frame[seed_columns].median(axis=1)

    replay_metrics = primary_metrics(frame)
    archived_report = json.loads(
        (PROJECT / "outputs/identification/identification_gates.json").read_text()
    )
    archived_metrics = archived_report["primary_metrics"]
    metric_differences = {
        key: float(replay_metrics[key] - archived_metrics[key])
        for key in archived_metrics
        if isinstance(archived_metrics[key], (int, float)) and key in replay_metrics
    }

    run_table = pd.DataFrame(run_rows)
    run_table.to_csv(REPLAY / "float64_fingerprint_differences.csv", index=False)
    report = {
        "purpose": "float64 replay of the 12 archived dHBV and flux-gate interventions",
        "archived_precision": "float32",
        "replay_precision": "float64",
        "maximum_archived_daily_mass_residual_mm": float(
            archived_report["gates"]["numerical_mass_balance"]["value"]
        ),
        "maximum_replay_daily_mass_residual_mm": float(max(replay_residuals)),
        "maximum_absolute_r7_change": float(run_table.max_abs_median_r7_difference.max()),
        "median_absolute_r7_change_across_runs": float(
            run_table.median_abs_median_r7_difference.median()
        ),
        "archived_primary_metrics": archived_metrics,
        "float64_primary_metrics": replay_metrics,
        "float64_minus_archived_primary_metrics": metric_differences,
        "reported_three_decimal_values_unchanged": bool(
            all(
                round(float(replay_metrics[key]), 3) == round(float(archived_metrics[key]), 3)
                for key in archived_metrics
                if isinstance(archived_metrics[key], (int, float)) and key in replay_metrics
            )
        ),
        "run_level_table": "outputs/law_interventions_float64_audit/float64_fingerprint_differences.csv",
    }
    (REPLAY / "float64_replay_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
