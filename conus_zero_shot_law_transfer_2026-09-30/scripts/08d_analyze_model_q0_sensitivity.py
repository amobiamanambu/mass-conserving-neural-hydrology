#!/usr/bin/env python3
"""Rebuild model R7 fingerprints under the prespecified Q(0) thresholds.

This sensitivity changes only the minimum modeled initial-flow threshold used
to admit an archived zero-input trajectory into a basin fingerprint. It does
not rerun or refit a model. The 90-day intervention curves were archived before
target discharge was opened, and observed recession R7 is held at its primary
definition so that this analysis isolates the model-side threshold.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")
RUNGS = ("q", "qet")
MODEL_Q0_THRESHOLDS = (0.0001, 0.001, 0.01)
CLASS_BREAKS = (0.5, float(np.sqrt(0.5)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_ids(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(values, dtype=str).str.replace(r"\.0$", "", regex=True).str.zfill(8)


def rho(x: pd.Series, y: pd.Series) -> float:
    x_values = x.to_numpy(float)
    y_values = y.to_numpy(float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 10 or np.std(x_values[valid]) == 0 or np.std(y_values[valid]) == 0:
        return np.nan
    return float(spearmanr(x_values[valid], y_values[valid]).statistic)


def median_absolute_difference(x: pd.Series, y: pd.Series) -> float:
    return float(np.nanmedian(np.abs(x.to_numpy(float) - y.to_numpy(float))))


def rebuild_run_fingerprint(curve_path: Path, q0_threshold: float) -> pd.DataFrame:
    archive = np.load(curve_path)
    q = archive["q_mm_day"]
    q0 = q[:, :, 0]
    valid = np.isfinite(q0) & (q0 >= q0_threshold)
    r7 = np.full(q0.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Preserve the archived float32 division used in the primary workflow.
        r7[valid] = q[:, :, 7][valid] / q0[valid]
    usable = np.isfinite(r7).any(axis=1)
    median_r7 = np.full(len(q), np.nan, dtype=float)
    median_r7[usable] = np.nanmedian(r7[usable], axis=1)
    return pd.DataFrame({
        "GAGE_ID": normalize_ids(archive["gage_ids"].astype(str)),
        "valid_anchors": np.isfinite(r7).sum(axis=1),
        "median_r7": median_r7,
    })


def supports_nonidentification(record: dict[str, float | int]) -> bool:
    """Gate 6/7 support rule used by the frozen analysis.

    The flag requires failure of the magnitude gate, failure of the expected
    ET-direction criterion, and failure of at least one observed-agreement
    criterion for each architecture. Rank and class stability are reported but
    are not components of this subgroup/sensitivity support flag.
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


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    q_mad = median_absolute_difference(frame.dhbv_q_r7, frame.fluxgate_q_r7)
    qet_mad = median_absolute_difference(frame.dhbv_qet_r7, frame.fluxgate_qet_r7)
    return {
        "gauges": int(len(frame)),
        "q_cross_architecture_rho": rho(frame.dhbv_q_r7, frame.fluxgate_q_r7),
        "q_cross_architecture_mad": q_mad,
        "qet_cross_architecture_rho": rho(frame.dhbv_qet_r7, frame.fluxgate_qet_r7),
        "qet_cross_architecture_mad": qet_mad,
        "disagreement_reduction_fraction": float(1.0 - qet_mad / q_mad),
        "dhbv_observed_rho": rho(frame.dhbv_qet_r7, frame.observed_median_r7),
        "dhbv_observed_mae": median_absolute_difference(
            frame.dhbv_qet_r7, frame.observed_median_r7
        ),
        "fluxgate_observed_rho": rho(
            frame.fluxgate_qet_r7, frame.observed_median_r7
        ),
        "fluxgate_observed_mae": median_absolute_difference(
            frame.fluxgate_qet_r7, frame.observed_median_r7
        ),
        "all_model_class_agreement_fraction": float(frame.class_agreement.mean()),
    }


def main() -> None:
    primary_path = PROJECT / "outputs/identification/primary_identification_population.parquet"
    primary = pd.read_parquet(primary_path)
    primary["GAGE_ID"] = normalize_ids(primary.GAGE_ID)
    primary = primary.set_index("GAGE_ID")
    records = []
    curve_checksums: dict[str, str] = {}

    for threshold in MODEL_Q0_THRESHOLDS:
        frame = pd.DataFrame(index=primary.index)
        qet_seed_columns: list[str] = []
        ensemble_columns: list[str] = []
        for architecture in ARCHITECTURES:
            for rung in RUNGS:
                seed_columns = []
                for seed in SEEDS:
                    run = f"{architecture}_{rung}_seed{seed}"
                    curve_path = PROJECT / f"outputs/law_interventions/{run}_curves.npz"
                    curve_checksums[str(curve_path.relative_to(PROJECT))] = sha256(curve_path)
                    table = rebuild_run_fingerprint(curve_path, threshold).set_index("GAGE_ID")
                    column = f"{run}_r7"
                    frame[column] = table.reindex(frame.index).median_r7.to_numpy(float)
                    seed_columns.append(column)
                    if rung == "qet":
                        qet_seed_columns.append(column)
                ensemble_column = f"{architecture}_{rung}_r7"
                # A model-side sensitivity gauge is complete only when every
                # seed contributes a fingerprint, matching the primary design.
                frame[ensemble_column] = frame[seed_columns].median(axis=1, skipna=False)
                ensemble_columns.append(ensemble_column)

        classes = np.column_stack([
            np.digitize(frame[column].to_numpy(float), CLASS_BREAKS)
            for column in qet_seed_columns
        ])
        frame["class_agreement"] = np.all(classes == classes[:, [0]], axis=1)
        complete_columns = [*ensemble_columns, *qet_seed_columns]
        complete = np.isfinite(frame[complete_columns]).all(axis=1)
        analysis = frame.loc[complete].join(primary[["observed_median_r7"]])
        metrics = calculate_metrics(analysis)
        record = {
            "model_q0_threshold_mm_day": threshold,
            "primary_gauges": int(len(primary)),
            "gauges_excluded_for_q0": int((~complete).sum()),
            **metrics,
        }
        record["supports_nonidentification"] = supports_nonidentification(record)
        records.append(record)

    output = pd.DataFrame(records)
    output_path = PROJECT / "outputs/identification/model_q0_sensitivity.csv"
    output.to_csv(output_path, index=False)
    gate_path = PROJECT / "outputs/identification/identification_gates.json"
    if gate_path.exists():
        gates = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_7 = gates["gates"]["gate_7_sensitivity"]
        observed_tests = int(gate_7.get("observed_recession_skill_grid_tests", 25))
        observed_support = int(
            gate_7.get("observed_recession_skill_grid_supporting_nonidentification", 25)
        )
        alternatives = output.loc[output.model_q0_threshold_mm_day.ne(0.001)]
        alternative_support = int(alternatives.supports_nonidentification.sum())
        gate_7.update({
            "observed_recession_skill_grid_tests": observed_tests,
            "observed_recession_skill_grid_supporting_nonidentification": observed_support,
            "alternative_model_q0_tests": int(len(alternatives)),
            "alternative_model_q0_supporting_nonidentification": alternative_support,
            "tests": observed_tests + int(len(alternatives)),
            "tests_supporting_nonidentification": observed_support + alternative_support,
            "supports_nonidentification": bool(
                observed_support == observed_tests
                and alternatives.supports_nonidentification.all()
            ),
        })
        gates.setdefault("files", {})["model_q0_sensitivity"] = str(
            output_path.relative_to(PROJECT)
        )
        gate_path.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "model-side initial-flow threshold sensitivity from archived intervention curves",
        "observed_recession_definition": "primary and held fixed",
        "thresholds_mm_day": list(MODEL_Q0_THRESHOLDS),
        "primary_population": str(primary_path.relative_to(PROJECT)),
        "primary_population_sha256": sha256(primary_path),
        "curve_sha256": curve_checksums,
        "output": str(output_path.relative_to(PROJECT)),
        "output_sha256": sha256(output_path),
        "gate_file_updated": gate_path.exists(),
        "gate_file_sha256": sha256(gate_path) if gate_path.exists() else None,
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    receipt_path = PROJECT / "provenance/model_q0_sensitivity_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(output.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
