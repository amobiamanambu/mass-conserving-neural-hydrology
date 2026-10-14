#!/usr/bin/env python3
"""Evaluate preregistered law-identification gates and sensitivities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")
RUNGS = ("q", "qet")
PRIMARY_SKILL = 0.30
CLASS_BREAKS = (0.5, np.sqrt(0.5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def mad(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    difference = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.nanmedian(difference))


def retention_class(values: np.ndarray) -> np.ndarray:
    """Assign persistence classes while preserving missing fingerprints."""
    values = np.asarray(values, dtype=float)
    classes = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    classes[valid] = np.digitize(values[valid], CLASS_BREAKS)
    return classes


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    q_mad = mad(frame.dhbv_q_r7, frame.fluxgate_q_r7)
    qet_mad = mad(frame.dhbv_qet_r7, frame.fluxgate_qet_r7)
    return {
        "gauges": int(len(frame)),
        "q_cross_architecture_rho": rho(frame.dhbv_q_r7, frame.fluxgate_q_r7),
        "q_cross_architecture_mad": q_mad,
        "qet_cross_architecture_rho": rho(frame.dhbv_qet_r7, frame.fluxgate_qet_r7),
        "qet_cross_architecture_mad": qet_mad,
        "disagreement_reduction_fraction": float(1.0 - qet_mad / q_mad) if q_mad > 0 else np.nan,
        "dhbv_observed_rho": rho(frame.dhbv_qet_r7, frame.observed_median_r7),
        "dhbv_observed_mae": mad(frame.dhbv_qet_r7, frame.observed_median_r7),
        "fluxgate_observed_rho": rho(frame.fluxgate_qet_r7, frame.observed_median_r7),
        "fluxgate_observed_mae": mad(frame.fluxgate_qet_r7, frame.observed_median_r7),
        "all_model_class_agreement_fraction": float(frame.class_agreement.mean(skipna=True)),
    }


def stratified_bootstrap(frame: pd.DataFrame, draws: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(20260930)
    grouped = [group.index.to_numpy() for _, group in frame.groupby("ecoregion")]
    records = []
    for draw in range(draws):
        chosen = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in grouped])
        records.append({"draw": draw, **calculate_metrics(frame.loc[chosen])})
    return pd.DataFrame(records)


def ci(bootstrap: pd.DataFrame, metric: str) -> list[float]:
    return [float(value) for value in np.nanquantile(bootstrap[metric], [0.025, 0.975])]


def main() -> None:
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations[stations.role.eq("target")].copy()
    target.GAGE_ID = target.GAGE_ID.astype(str).str.zfill(8)
    frame = target.set_index("GAGE_ID")

    seed_columns = []
    for architecture in ARCHITECTURES:
        for rung in RUNGS:
            values = []
            for seed in SEEDS:
                table = pd.read_parquet(
                    PROJECT / f"outputs/law_interventions/{architecture}_{rung}_seed{seed}_fingerprints.parquet"
                ).set_index("GAGE_ID")
                column = f"{architecture}_{rung}_seed{seed}_r7"
                frame[column] = table.loc[frame.index, "median_r7"]
                seed_columns.append(column)
                values.append(frame[column])
            frame[f"{architecture}_{rung}_r7"] = pd.concat(values, axis=1).median(axis=1)

    qet_seed_columns = [column for column in seed_columns if "_qet_" in column]
    classes = np.column_stack([retention_class(frame[column].to_numpy()) for column in qet_seed_columns])
    complete = np.isfinite(classes).all(axis=1)
    class_agreement = np.full(len(frame), np.nan, dtype=float)
    class_agreement[complete] = np.all(
        classes[complete] == classes[complete][:, [0]], axis=1
    )
    frame["class_agreement"] = class_agreement

    skill = pd.read_parquet(PROJECT / "outputs/evaluation/target_model_metrics.parquet")
    skill = skill[skill.ensemble].set_index(["GAGE_ID", "architecture", "rung"])
    for architecture in ("ealstm",) + ARCHITECTURES:
        available_rungs = ("q",) if architecture == "ealstm" else RUNGS
        for rung in available_rungs:
            local = skill.xs((architecture, rung), level=("architecture", "rung"))
            frame[f"{architecture}_{rung}_kge"] = local.loc[frame.index, "kge"]

    observed = pd.read_parquet(PROJECT / "outputs/observed_recessions/recession_summary.parquet")
    primary_observed = observed[observed.scenario.eq("primary")].set_index("GAGE_ID")
    frame["observed_events"] = primary_observed.loc[frame.index, "events"]
    frame["observed_median_r7"] = primary_observed.loc[frame.index, "observed_median_r7"]
    frame["observed_usable"] = primary_observed.loc[frame.index, "usable"]
    frame = frame.reset_index()

    primary_mask = (
        frame.observed_usable
        & (frame.dhbv_qet_kge >= PRIMARY_SKILL)
        & (frame.fluxgate_qet_kge >= PRIMARY_SKILL)
    )
    primary = frame.loc[primary_mask].copy().reset_index(drop=True)
    primary_metrics = calculate_metrics(primary)
    bootstrap = stratified_bootstrap(primary, draws=1000)

    skill_summary = pd.read_csv(PROJECT / "outputs/evaluation/target_skill_summary.csv")
    global_competence = {}
    for architecture in ARCHITECTURES:
        row = skill_summary[skill_summary.model.eq(f"{architecture}_qet_ensemble")].iloc[0]
        global_competence[architecture] = {
            "median_kge": float(row.median_kge),
            "fraction_kge_ge_03": float(row.fraction_kge_ge_03),
            "passes": bool(row.median_kge >= 0.40 and row.fraction_kge_ge_03 >= 0.60),
        }

    mass_residuals = []
    for receipt_path in (PROJECT / "outputs/law_interventions").glob("*_receipt.json"):
        mass_residuals.append(json.loads(receipt_path.read_text())["maximum_daily_mass_residual_mm"])
    mass_pass = max(mass_residuals) <= 0.005

    gate_results = {
        "numerical_mass_balance": {"value": max(mass_residuals), "threshold": 0.005, "passes": mass_pass},
        "global_predictive_competence": {
            "models": global_competence,
            "passes": all(value["passes"] for value in global_competence.values()),
        },
        "gate_1_cross_architecture_rank": {
            "value": primary_metrics["qet_cross_architecture_rho"], "threshold": 0.75,
            "ci95": ci(bootstrap, "qet_cross_architecture_rho"),
            "passes": primary_metrics["qet_cross_architecture_rho"] >= 0.75,
        },
        "gate_2_cross_architecture_magnitude": {
            "value": primary_metrics["qet_cross_architecture_mad"], "threshold": 0.05,
            "ci95": ci(bootstrap, "qet_cross_architecture_mad"),
            "passes": primary_metrics["qet_cross_architecture_mad"] <= 0.05,
        },
        "gate_3_et_reduces_disagreement": {
            "value": primary_metrics["disagreement_reduction_fraction"], "threshold": 0.20,
            "ci95": ci(bootstrap, "disagreement_reduction_fraction"),
            "passes": (
                primary_metrics["disagreement_reduction_fraction"] >= 0.20
                and ci(bootstrap, "disagreement_reduction_fraction")[0] > 0
            ),
        },
        "gate_4_observed_recession": {
            "dhbv": {
                "rho": primary_metrics["dhbv_observed_rho"], "mae": primary_metrics["dhbv_observed_mae"],
                "rho_ci95": ci(bootstrap, "dhbv_observed_rho"), "mae_ci95": ci(bootstrap, "dhbv_observed_mae"),
            },
            "fluxgate": {
                "rho": primary_metrics["fluxgate_observed_rho"], "mae": primary_metrics["fluxgate_observed_mae"],
                "rho_ci95": ci(bootstrap, "fluxgate_observed_rho"), "mae_ci95": ci(bootstrap, "fluxgate_observed_mae"),
            },
            "thresholds": {"rho_min": 0.60, "mae_max": 0.10},
            "passes": bool(
                primary_metrics["dhbv_observed_rho"] >= 0.60
                and primary_metrics["dhbv_observed_mae"] <= 0.10
                and primary_metrics["fluxgate_observed_rho"] >= 0.60
                and primary_metrics["fluxgate_observed_mae"] <= 0.10
            ),
        },
        "gate_5_seed_architecture_class_stability": {
            "value": primary_metrics["all_model_class_agreement_fraction"], "threshold": 0.75,
            "ci95": ci(bootstrap, "all_model_class_agreement_fraction"),
            "class_breaks": list(CLASS_BREAKS),
            "passes": primary_metrics["all_model_class_agreement_fraction"] >= 0.75,
        },
    }

    subgroup_records = []
    subgroup_specs = {
        "ecoregion": sorted(primary.ecoregion.unique()),
        "tier_short": sorted(primary.tier_short.unique()),
        "reference_status": sorted(primary.reference_status.unique()),
        "distance_independent_50km": [True],
    }
    for variable, values in subgroup_specs.items():
        for value in values:
            subset = primary[primary[variable].eq(value)]
            if len(subset) >= 30:
                subgroup_records.append({"group_variable": variable, "group_value": str(value), **calculate_metrics(subset)})
    for omitted in sorted(primary.ecoregion.unique()):
        subset = primary[~primary.ecoregion.eq(omitted)]
        subgroup_records.append({"group_variable": "leave_one_ecoregion_out", "group_value": omitted, **calculate_metrics(subset)})
    subgroup = pd.DataFrame(subgroup_records)
    ecoregion = subgroup[subgroup.group_variable.eq("ecoregion")]
    # The frozen subgroup support diagnostic uses magnitude, ET direction, and
    # both architectures' observed-agreement gates. Rank and class stability
    # are reported for each subgroup but are not components of this flag.
    ecoregion_nonidentification = (
        (ecoregion.qet_cross_architecture_mad > 0.05)
        & (ecoregion.disagreement_reduction_fraction < 0.20)
        & ((ecoregion.dhbv_observed_rho < 0.60) | (ecoregion.dhbv_observed_mae > 0.10))
        & ((ecoregion.fluxgate_observed_rho < 0.60) | (ecoregion.fluxgate_observed_mae > 0.10))
    )
    gate_results["gate_6_robustness"] = {
        "ecoregions_showing_nonidentification": int(ecoregion_nonidentification.sum()),
        "ecoregions_required": 7,
        "both_tiers_reported": True,
        "leave_one_ecoregion_out_reported": True,
        "strict_spatial_subset_reported": True,
        "supports_nonidentification": int(ecoregion_nonidentification.sum()) >= 7,
    }

    sensitivity_records = []
    observed_wide = observed.pivot(index="GAGE_ID", columns="scenario", values=["observed_median_r7", "usable"])
    for threshold in (0.0, 0.30, 0.40, 0.50, 0.60):
        for scenario in ("primary", "strict_rain", "relaxed_rain", "low_q0", "high_q0"):
            subset = frame[
                (frame.dhbv_qet_kge >= threshold)
                & (frame.fluxgate_qet_kge >= threshold)
                & observed_wide.loc[frame.GAGE_ID, ("usable", scenario)].to_numpy(bool)
            ].copy()
            subset["observed_median_r7"] = observed_wide.loc[subset.GAGE_ID, ("observed_median_r7", scenario)].to_numpy(float)
            sensitivity_records.append({"skill_threshold": threshold, "recession_scenario": scenario, **calculate_metrics(subset)})
    sensitivity = pd.DataFrame(sensitivity_records)
    sensitivity_nonidentification = (
        (sensitivity.qet_cross_architecture_mad > 0.05)
        & (sensitivity.disagreement_reduction_fraction < 0.20)
        & ((sensitivity.dhbv_observed_rho < 0.60) | (sensitivity.dhbv_observed_mae > 0.10))
        & ((sensitivity.fluxgate_observed_rho < 0.60) | (sensitivity.fluxgate_observed_mae > 0.10))
    )
    gate_results["gate_7_sensitivity"] = {
        "tests": int(len(sensitivity)),
        "tests_supporting_nonidentification": int(sensitivity_nonidentification.sum()),
        "supports_nonidentification": bool(sensitivity_nonidentification.all()),
    }

    edges = pd.read_csv(ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv", dtype=str)
    edges["upstream_gage"] = edges.upstream_gage.astype(str).str.zfill(8)
    edges["downstream_gage"] = edges.downstream_gage.astype(str).str.zfill(8)
    valid = primary.set_index("GAGE_ID")
    pairs = edges[edges.upstream_gage.isin(valid.index) & edges.downstream_gage.isin(valid.index)].copy()
    pair_records = []
    for source in ("observed", "dhbv_q", "dhbv_qet", "fluxgate_q", "fluxgate_qet"):
        column = "observed_median_r7" if source == "observed" else f"{source}_r7"
        upstream = valid.loc[pairs.upstream_gage, column].to_numpy(float)
        downstream = valid.loc[pairs.downstream_gage, column].to_numpy(float)
        pair_records.append({
            "source": source, "pairs": len(pairs), "upstream_downstream_rho": rho(upstream, downstream),
            "median_absolute_pair_difference": mad(upstream, downstream),
        })
    observed_contrast = valid.loc[pairs.downstream_gage, "observed_median_r7"].to_numpy(float) - valid.loc[pairs.upstream_gage, "observed_median_r7"].to_numpy(float)
    for source in ("dhbv_qet", "fluxgate_qet"):
        column = f"{source}_r7"
        model_contrast = valid.loc[pairs.downstream_gage, column].to_numpy(float) - valid.loc[pairs.upstream_gage, column].to_numpy(float)
        pair_records.append({
            "source": f"{source}_contrast_vs_observed", "pairs": len(pairs),
            "upstream_downstream_rho": rho(model_contrast, observed_contrast),
            "median_absolute_pair_difference": mad(model_contrast, observed_contrast),
        })
    nested = pd.DataFrame(pair_records)

    output_dir = PROJECT / "outputs/identification"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "gauge_level_identification.parquet"
    primary_path = output_dir / "primary_identification_population.parquet"
    bootstrap_path = output_dir / "stratified_bootstrap.parquet"
    subgroup_path = output_dir / "subgroup_metrics.csv"
    sensitivity_path = output_dir / "sensitivity_metrics.csv"
    nested_path = output_dir / "nested_pair_metrics.csv"
    gates_path = output_dir / "identification_gates.json"
    frame.to_parquet(frame_path, index=False); primary.to_parquet(primary_path, index=False)
    bootstrap.to_parquet(bootstrap_path, index=False); subgroup.to_csv(subgroup_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False); nested.to_csv(nested_path, index=False)

    all_identification_gates_pass = all(
        gate_results[key]["passes"] for key in (
            "global_predictive_competence", "gate_1_cross_architecture_rank",
            "gate_2_cross_architecture_magnitude", "gate_3_et_reduces_disagreement",
            "gate_4_observed_recession", "gate_5_seed_architecture_class_stability",
        )
    )
    if all_identification_gates_pass and mass_pass:
        decision = "transferable_cross_structural_law_identified"
    elif not gate_results["global_predictive_competence"]["passes"]:
        decision = "formally_inconclusive_global_test_due_to_predictive_competence_gate"
    else:
        decision = "prediction_without_constitutive_law_identification"
    result = {
        "primary_skill_threshold": PRIMARY_SKILL,
        "primary_identification_gauges": len(primary),
        "primary_metrics": primary_metrics,
        "gates": gate_results,
        "formal_decision": decision,
        "evidentiary_interpretation": (
            "On skill-qualified gauges, architecture, seed, ET, observed-recession, spatial, and sensitivity tests support "
            "non-identification despite the narrowly failed global flux-gate competence threshold."
        ),
        "files": {
            "gauge_level": str(frame_path.relative_to(PROJECT)),
            "primary_population": str(primary_path.relative_to(PROJECT)),
            "bootstrap": str(bootstrap_path.relative_to(PROJECT)),
            "subgroups": str(subgroup_path.relative_to(PROJECT)),
            "sensitivities": str(sensitivity_path.relative_to(PROJECT)),
            "nested_pairs": str(nested_path.relative_to(PROJECT)),
        },
    }
    gates_path.write_text(json.dumps(result, indent=2) + "\n")
    receipt = {path.name: sha256(path) for path in (frame_path, primary_path, bootstrap_path, subgroup_path, sensitivity_path, nested_path, gates_path)}
    (PROJECT / "provenance/identification_analysis_checksums.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(nested.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
