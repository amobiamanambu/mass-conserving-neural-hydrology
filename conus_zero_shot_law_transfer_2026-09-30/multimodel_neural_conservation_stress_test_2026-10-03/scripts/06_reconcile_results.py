#!/usr/bin/env python3
"""Independent reconciliation checks for the completed stress test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conserved_models import MODEL_NAMES


EXPERIMENT = Path(__file__).resolve().parents[1]
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    expected = {f"{model}_qet_seed{seed}" for model in MODEL_NAMES for seed in SEEDS}
    run_receipts = [
        json.loads(path.read_text())
        for path in sorted((EXPERIMENT / "runs/frozen_refit").glob("*/receipt.json"))
    ]
    assert {item["run"] for item in run_receipts} == expected
    assert all(not item["target_discharge_accessed_by_script"] for item in run_receipts)
    for item in run_receipts:
        for key, digest_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("prediction", "prediction_sha256"),
            ("model_specific_states", "model_specific_states_sha256"),
            ("common_date_states", "common_date_states_sha256"),
        ):
            assert sha256(EXPERIMENT / item[key]) == item[digest_key]

    intervention_receipts = [
        json.loads(path.read_text())
        for path in sorted((EXPERIMENT / "outputs/interventions").glob("*_receipt.json"))
    ]
    assert len(intervention_receipts) == 30
    assert {(item["run"], item["anchor_design"]) for item in intervention_receipts} == {
        (run, design) for run in expected for design in ("model_specific", "common_date")
    }
    for item in intervention_receipts:
        assert sha256(EXPERIMENT / item["curves"]) == item["curves_sha256"]
        assert sha256(EXPERIMENT / item["fingerprints"]) == item["fingerprints_sha256"]
        table = pd.read_parquet(EXPERIMENT / item["fingerprints"])
        assert len(table) == 3982 and table.GAGE_ID.nunique() == 3982

    metrics = pd.read_parquet(EXPERIMENT / "outputs/evaluation/target_model_metrics.parquet")
    assert len(metrics) == 3982 * 20
    assert metrics.loc[metrics.ensemble].groupby("architecture").GAGE_ID.nunique().eq(3982).all()
    summary = pd.read_csv(EXPERIMENT / "outputs/evaluation/target_skill_summary.csv")
    ensemble = summary.loc[summary.ensemble]
    recomputed = (
        metrics.loc[metrics.ensemble]
        .groupby("architecture")
        .agg(
            median_kge=("kge", "median"),
            fraction_kge_ge_03=("kge", lambda x: np.mean(x >= 0.30)),
        )
    )
    aligned = ensemble.set_index("architecture").loc[recomputed.index]
    assert np.allclose(recomputed.median_kge, aligned.median_kge, atol=1e-12)
    assert np.allclose(
        recomputed.fraction_kge_ge_03, aligned.fraction_kge_ge_03, atol=1e-12
    )

    mass = pd.read_csv(EXPERIMENT / "outputs/comparison/mass_balance_summary.csv")
    assert len(mass) == 15 and mass.passes_0_005_mm_gate.all()
    pairs = pd.read_csv(EXPERIMENT / "outputs/comparison/pairwise_response_agreement.csv")
    primary = pairs.loc[
        pairs.anchor_design.eq("model_specific")
        & pairs.population.eq("all")
        & pairs.skill_threshold.eq(0.30)
        & (pairs.n >= 500)
    ]
    common = pairs.loc[
        pairs.anchor_design.eq("common_date")
        & pairs.population.eq("all")
        & pairs.skill_threshold.eq(0.30)
        & (pairs.n >= 500)
    ]
    high = pairs.loc[
        pairs.anchor_design.eq("model_specific")
        & pairs.population.eq("all")
        & pairs.skill_threshold.eq(0.60)
        & (pairs.n >= 30)
    ]
    reference = pairs.loc[
        pairs.anchor_design.eq("model_specific")
        & pairs.population.eq("reference")
        & pairs.skill_threshold.eq(0.30)
        & (pairs.n >= 30)
    ]

    result = {
        "status": "pass",
        "frozen_prediction_receipts": len(run_receipts),
        "intervention_receipts": len(intervention_receipts),
        "target_metric_rows": len(metrics),
        "all_prediction_files_match_receipts": True,
        "all_intervention_files_match_receipts": True,
        "all_mass_gates_pass": True,
        "maximum_refit_daily_residual_mm": float(mass.refit_daily_max_abs_mm.max()),
        "maximum_refit_cumulative_residual_mm": float(
            mass.refit_cumulative_max_abs_mm.max()
        ),
        "maximum_intervention_daily_residual_mm": float(
            mass.intervention_daily_max_abs_mm.max()
        ),
        "primary_pairs": int(len(primary)),
        "primary_disagreements": int(primary.material_disagreement.sum()),
        "common_date_pairs": int(len(common)),
        "common_date_disagreements": int(common.material_disagreement.sum()),
        "high_skill_pairs": int(len(high)),
        "high_skill_disagreements": int(high.material_disagreement.sum()),
        "reference_pairs": int(len(reference)),
        "reference_disagreements": int(reference.material_disagreement.sum()),
    }
    output = EXPERIMENT / "outputs/comparison/reconciliation_receipt.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
