#!/usr/bin/env python3
"""Compare MCP response fingerprints with both paper architectures and observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
SEEDS = (11, 29, 47)
CLASS_BREAKS = (0.5, np.sqrt(0.5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def median_absolute_difference(x: pd.Series, y: pd.Series) -> float:
    return float(np.nanmedian(np.abs(x.to_numpy(float) - y.to_numpy(float))))


def retention_class(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    classes = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    classes[valid] = np.digitize(values[valid], CLASS_BREAKS).astype(np.int8)
    return classes


def pair_metrics(frame: pd.DataFrame, first: str, second: str) -> dict:
    first_class = retention_class(frame[first].to_numpy(float))
    second_class = retention_class(frame[second].to_numpy(float))
    valid_class = (first_class >= 0) & (second_class >= 0)
    return {
        "n": int(len(frame)),
        "first": first,
        "second": second,
        "cross_architecture_rho": rho(frame[first], frame[second]),
        "median_absolute_r7_difference": median_absolute_difference(
            frame[first], frame[second]
        ),
        "first_observed_rho": rho(frame[first], frame.observed_median_r7),
        "second_observed_rho": rho(frame[second], frame.observed_median_r7),
        "first_observed_mae": median_absolute_difference(
            frame[first], frame.observed_median_r7
        ),
        "second_observed_mae": median_absolute_difference(
            frame[second], frame.observed_median_r7
        ),
        "same_retention_class_fraction": (
            float(np.mean(first_class[valid_class] == second_class[valid_class]))
            if valid_class.any()
            else np.nan
        ),
    }


def bootstrap_pair(
    frame: pd.DataFrame, first: str, second: str, draws: int = 1000
) -> dict[str, list[float]]:
    rng = np.random.default_rng(20261003)
    groups = [group.index.to_numpy() for _, group in frame.groupby("ecoregion")]
    records = []
    for _ in range(draws):
        chosen = np.concatenate(
            [rng.choice(index, size=len(index), replace=True) for index in groups]
        )
        records.append(pair_metrics(frame.loc[chosen], first, second))
    table = pd.DataFrame(records)
    columns = [
        "cross_architecture_rho",
        "median_absolute_r7_difference",
        "first_observed_rho",
        "second_observed_rho",
        "first_observed_mae",
        "second_observed_mae",
        "same_retention_class_fraction",
    ]
    return {
        column: [float(value) for value in np.nanquantile(table[column], [0.025, 0.975])]
        for column in columns
    }


def load_mcp_fingerprints(anchor_design: str) -> pd.DataFrame:
    seed_tables = []
    for rung in ("q", "qet"):
        for seed in SEEDS:
            path = (
                EXPERIMENT
                / f"outputs/interventions/mcp_{rung}_seed{seed}_{anchor_design}_fingerprints.parquet"
            )
            table = pd.read_parquet(path, columns=["GAGE_ID", "median_r7"])
            table["GAGE_ID"] = table.GAGE_ID.astype(str).str.zfill(8)
            table = table.rename(columns={"median_r7": f"mcp_{rung}_seed{seed}_r7"})
            seed_tables.append(table.set_index("GAGE_ID"))
    frame = pd.concat(seed_tables, axis=1)
    for rung in ("q", "qet"):
        columns = [f"mcp_{rung}_seed{seed}_r7" for seed in SEEDS]
        frame[f"mcp_{rung}_r7"] = frame[columns].median(axis=1)
    return frame.reset_index()


def load_common_paper_fingerprints() -> pd.DataFrame:
    trace = pd.read_parquet(
        PROJECT / "outputs/identification/strengthened_test/common_date_flux_trace.parquet",
        columns=["architecture", "seed", "GAGE_ID", "r7"],
    )
    trace["GAGE_ID"] = trace.GAGE_ID.astype(str).str.zfill(8)
    seed = (
        trace.groupby(["architecture", "seed", "GAGE_ID"], as_index=False).r7.median()
    )
    architecture = (
        seed.groupby(["architecture", "GAGE_ID"], as_index=False).r7.median()
        .pivot(index="GAGE_ID", columns="architecture", values="r7")
        .rename(columns={"dhbv": "dhbv_qet_r7", "fluxgate": "fluxgate_qet_r7"})
        .reset_index()
    )
    return architecture


def main() -> None:
    output_dir = EXPERIMENT / "outputs/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(PROJECT / "outputs/identification/gauge_level_identification.parquet")
    base["GAGE_ID"] = base.GAGE_ID.astype(str).str.zfill(8)
    model_specific = load_mcp_fingerprints("model_specific")
    frame = base.merge(model_specific, on="GAGE_ID", how="left", validate="one_to_one")

    mcp_skill = pd.read_parquet(
        EXPERIMENT / "outputs/evaluation/target_model_metrics.parquet"
    )
    mcp_skill = mcp_skill.loc[mcp_skill.ensemble].set_index(["GAGE_ID", "rung"])
    for rung in ("q", "qet"):
        local = mcp_skill.xs(rung, level="rung")
        frame[f"mcp_{rung}_kge"] = local.loc[frame.GAGE_ID, "kge"].to_numpy(float)

    qet_seed_columns = [
        f"{architecture}_qet_seed{seed}_r7"
        for architecture in ("dhbv", "fluxgate", "mcp")
        for seed in SEEDS
    ]
    classes = np.column_stack(
        [retention_class(frame[column].to_numpy(float)) for column in qet_seed_columns]
    )
    finite_classes = np.column_stack(
        [np.isfinite(frame[column].to_numpy(float)) for column in qet_seed_columns]
    )
    frame["three_architecture_nine_member_class_agreement"] = finite_classes.all(
        axis=1
    ) & np.all(classes == classes[:, [0]], axis=1)
    three_values = frame[["dhbv_qet_r7", "fluxgate_qet_r7", "mcp_qet_r7"]]
    frame["three_architecture_r7_span"] = three_values.max(axis=1) - three_values.min(axis=1)

    populations = []
    pair_rows = []
    sensitivity_rows = []
    for threshold in (0.0, 0.30, 0.40, 0.50, 0.60):
        triple = frame.loc[
            frame.observed_usable
            & (frame.dhbv_qet_kge >= threshold)
            & (frame.fluxgate_qet_kge >= threshold)
            & (frame.mcp_qet_kge >= threshold)
            & frame.mcp_qet_r7.notna()
        ].copy()
        sensitivity_rows.append(
            {
                "population": "all",
                "skill_threshold": threshold,
                "n": int(len(triple)),
                "mcp_dhbv_rho": rho(triple.mcp_qet_r7, triple.dhbv_qet_r7),
                "mcp_dhbv_mad": median_absolute_difference(
                    triple.mcp_qet_r7, triple.dhbv_qet_r7
                ) if len(triple) else np.nan,
                "mcp_fluxgate_rho": rho(triple.mcp_qet_r7, triple.fluxgate_qet_r7),
                "mcp_fluxgate_mad": median_absolute_difference(
                    triple.mcp_qet_r7, triple.fluxgate_qet_r7
                ) if len(triple) else np.nan,
                "mcp_observed_rho": rho(triple.mcp_qet_r7, triple.observed_median_r7),
                "mcp_observed_mae": median_absolute_difference(
                    triple.mcp_qet_r7, triple.observed_median_r7
                ) if len(triple) else np.nan,
                "nine_member_class_agreement_fraction": float(
                    triple.three_architecture_nine_member_class_agreement.mean()
                ) if len(triple) else np.nan,
                "median_three_architecture_r7_span": float(
                    triple.three_architecture_r7_span.median()
                ) if len(triple) else np.nan,
            }
        )
        reference = triple.loc[triple.reference_status.eq("Reference")]
        sensitivity_rows.append(
            {
                "population": "reference",
                "skill_threshold": threshold,
                "n": int(len(reference)),
                "mcp_dhbv_rho": rho(reference.mcp_qet_r7, reference.dhbv_qet_r7),
                "mcp_dhbv_mad": median_absolute_difference(
                    reference.mcp_qet_r7, reference.dhbv_qet_r7
                ) if len(reference) else np.nan,
                "mcp_fluxgate_rho": rho(
                    reference.mcp_qet_r7, reference.fluxgate_qet_r7
                ),
                "mcp_fluxgate_mad": median_absolute_difference(
                    reference.mcp_qet_r7, reference.fluxgate_qet_r7
                ) if len(reference) else np.nan,
                "mcp_observed_rho": rho(
                    reference.mcp_qet_r7, reference.observed_median_r7
                ),
                "mcp_observed_mae": median_absolute_difference(
                    reference.mcp_qet_r7, reference.observed_median_r7
                ) if len(reference) else np.nan,
                "nine_member_class_agreement_fraction": float(
                    reference.three_architecture_nine_member_class_agreement.mean()
                ) if len(reference) else np.nan,
                "median_three_architecture_r7_span": float(
                    reference.three_architecture_r7_span.median()
                ) if len(reference) else np.nan,
            }
        )
        if threshold == 0.30:
            populations.append(("all_primary", triple))
            populations.append(("reference_primary", reference))
            for population_name, population in populations:
                for first, second in (
                    ("mcp_qet_r7", "dhbv_qet_r7"),
                    ("mcp_qet_r7", "fluxgate_qet_r7"),
                    ("dhbv_qet_r7", "fluxgate_qet_r7"),
                ):
                    values = pair_metrics(population, first, second)
                    values["population"] = population_name
                    pair_rows.append(values)

    primary = dict(populations)["all_primary"]
    reference_primary = dict(populations)["reference_primary"]
    ci = {
        "mcp_vs_dhbv": bootstrap_pair(primary, "mcp_qet_r7", "dhbv_qet_r7"),
        "mcp_vs_fluxgate": bootstrap_pair(
            primary, "mcp_qet_r7", "fluxgate_qet_r7"
        ),
    }

    q_mad_dhbv = median_absolute_difference(primary.mcp_q_r7, primary.dhbv_q_r7)
    qet_mad_dhbv = median_absolute_difference(primary.mcp_qet_r7, primary.dhbv_qet_r7)
    q_mad_flux = median_absolute_difference(primary.mcp_q_r7, primary.fluxgate_q_r7)
    qet_mad_flux = median_absolute_difference(
        primary.mcp_qet_r7, primary.fluxgate_qet_r7
    )

    common = (
        base[
            [
                "GAGE_ID",
                "ecoregion",
                "reference_status",
                "observed_usable",
                "observed_median_r7",
                "dhbv_qet_kge",
                "fluxgate_qet_kge",
            ]
        ]
        .merge(load_common_paper_fingerprints(), on="GAGE_ID", how="left")
        .merge(
            load_mcp_fingerprints("common_date")[
                ["GAGE_ID", "mcp_qet_r7"]
            ],
            on="GAGE_ID",
            how="left",
        )
    )
    common["mcp_qet_kge"] = frame.set_index("GAGE_ID").loc[
        common.GAGE_ID, "mcp_qet_kge"
    ].to_numpy(float)
    common_primary = common.loc[
        common.observed_usable
        & (common.dhbv_qet_kge >= 0.30)
        & (common.fluxgate_qet_kge >= 0.30)
        & (common.mcp_qet_kge >= 0.30)
        & common.mcp_qet_r7.notna()
    ].copy()
    common_metrics = {
        "mcp_vs_dhbv": pair_metrics(
            common_primary, "mcp_qet_r7", "dhbv_qet_r7"
        ),
        "mcp_vs_fluxgate": pair_metrics(
            common_primary, "mcp_qet_r7", "fluxgate_qet_r7"
        ),
    }

    edges = pd.read_csv(
        ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv",
        dtype=str,
    )
    nested_source = primary.set_index("GAGE_ID")
    pairs = edges.loc[
        edges.upstream_gage.isin(nested_source.index)
        & edges.downstream_gage.isin(nested_source.index)
    ].copy()
    observed_contrast = (
        nested_source.loc[pairs.downstream_gage, "observed_median_r7"].to_numpy(float)
        - nested_source.loc[pairs.upstream_gage, "observed_median_r7"].to_numpy(float)
    )
    nested_rows = []
    for architecture in ("dhbv", "fluxgate", "mcp"):
        column = f"{architecture}_qet_r7"
        upstream = nested_source.loc[pairs.upstream_gage, column].to_numpy(float)
        downstream = nested_source.loc[pairs.downstream_gage, column].to_numpy(float)
        contrast = downstream - upstream
        nested_rows.append(
            {
                "architecture": architecture,
                "pairs": int(len(pairs)),
                "upstream_downstream_rho": rho(upstream, downstream),
                "contrast_vs_observed_rho": rho(contrast, observed_contrast),
                "contrast_vs_observed_mae": float(
                    np.nanmedian(np.abs(contrast - observed_contrast))
                ),
            }
        )

    skill_summary = pd.read_csv(
        EXPERIMENT / "outputs/evaluation/target_skill_summary.csv"
    )
    mcp_skill_summary = skill_summary.loc[
        skill_summary.model.eq("mcp_qet_ensemble")
    ].iloc[0]
    receipts = [
        json.loads(path.read_text())
        for path in (EXPERIMENT / "runs/frozen_refit").glob("mcp_*/receipt.json")
    ]
    intervention_receipts = [
        json.loads(path.read_text())
        for path in (EXPERIMENT / "outputs/interventions").glob("*_receipt.json")
    ]
    max_daily_mass_error = max(
        [receipt["mass_balance"]["daily_max_abs_mm"] for receipt in receipts]
        + [receipt["maximum_daily_mass_residual_mm"] for receipt in intervention_receipts]
    )
    max_cumulative_mass_error = max(
        receipt["mass_balance"]["cumulative_max_abs_mm"] for receipt in receipts
    )

    primary_pair = {
        row["second"]: row
        for row in pair_rows
        if row["population"] == "all_primary" and row["first"] == "mcp_qet_r7"
    }
    architecture_disagreement = all(
        (value["cross_architecture_rho"] < 0.75)
        or (value["median_absolute_r7_difference"] > 0.05)
        for value in primary_pair.values()
    )
    observed_failure = (
        rho(primary.mcp_qet_r7, primary.observed_median_r7) < 0.60
        or median_absolute_difference(
            primary.mcp_qet_r7, primary.observed_median_r7
        ) > 0.10
    )
    reference_support = (
        len(reference_primary) >= 30
        and (
            median_absolute_difference(
                reference_primary.mcp_qet_r7, reference_primary.dhbv_qet_r7
            )
            > 0.05
            or rho(reference_primary.mcp_qet_r7, reference_primary.dhbv_qet_r7)
            < 0.75
        )
        and (
            median_absolute_difference(
                reference_primary.mcp_qet_r7, reference_primary.fluxgate_qet_r7
            )
            > 0.05
            or rho(
                reference_primary.mcp_qet_r7, reference_primary.fluxgate_qet_r7
            )
            < 0.75
        )
    )
    global_competence = bool(
        mcp_skill_summary.median_kge >= 0.40
        and mcp_skill_summary.fraction_kge_ge_03 >= 0.60
    )
    conditional_support = bool(
        len(primary) >= 500
        and architecture_disagreement
        and observed_failure
        and reference_support
        and max_daily_mass_error <= 0.005
    )
    if global_competence and conditional_support:
        verdict = "third_architecture_confirms_nonidentification"
    elif conditional_support:
        verdict = (
            "conditional_third_architecture_support_but_global_test_inconclusive_due_to_mcp_skill"
        )
    else:
        verdict = "third_architecture_does_not_yet_support_nonidentification"

    pair_path = output_dir / "pairwise_metrics.csv"
    sensitivity_path = output_dir / "skill_and_reference_sensitivity.csv"
    nested_path = output_dir / "nested_pair_metrics.csv"
    gauge_path = output_dir / "gauge_level_mcp_comparison.parquet"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    pd.DataFrame(nested_rows).to_csv(nested_path, index=False)
    frame.to_parquet(gauge_path, index=False)

    summary = {
        "architecture": "regional_12_cell_mass_conserving_perceptron",
        "population": {
            "targets": 3982,
            "triple_skill_observed_primary": int(len(primary)),
            "triple_skill_reference_primary": int(len(reference_primary)),
            "triple_skill_common_date_primary": int(len(common_primary)),
        },
        "predictive_skill": {
            "median_kge": float(mcp_skill_summary.median_kge),
            "fraction_kge_ge_03": float(mcp_skill_summary.fraction_kge_ge_03),
            "fraction_kge_ge_04": float(mcp_skill_summary.fraction_kge_ge_04),
            "passes_original_global_competence_gate": global_competence,
        },
        "mass_balance": {
            "maximum_daily_absolute_residual_mm": float(max_daily_mass_error),
            "maximum_cumulative_absolute_residual_mm": float(
                max_cumulative_mass_error
            ),
            "passes_0_005_mm_gate": bool(max_daily_mass_error <= 0.005),
        },
        "model_specific_primary_pairwise": primary_pair,
        "model_specific_primary_ci95": ci,
        "common_date_primary_pairwise": common_metrics,
        "mcp_observed": {
            "rho": rho(primary.mcp_qet_r7, primary.observed_median_r7),
            "mae": median_absolute_difference(
                primary.mcp_qet_r7, primary.observed_median_r7
            ),
        },
        "three_architecture": {
            "median_r7_span": float(primary.three_architecture_r7_span.median()),
            "fraction_span_le_0_05": float(
                np.mean(primary.three_architecture_r7_span <= 0.05)
            ),
            "nine_member_class_agreement_fraction": float(
                primary.three_architecture_nine_member_class_agreement.mean()
            ),
        },
        "et_supervision": {
            "mcp_dhbv_q_mad": q_mad_dhbv,
            "mcp_dhbv_qet_mad": qet_mad_dhbv,
            "mcp_dhbv_disagreement_reduction_fraction": float(
                1.0 - qet_mad_dhbv / q_mad_dhbv
            ),
            "mcp_fluxgate_q_mad": q_mad_flux,
            "mcp_fluxgate_qet_mad": qet_mad_flux,
            "mcp_fluxgate_disagreement_reduction_fraction": float(
                1.0 - qet_mad_flux / q_mad_flux
            ),
        },
        "reference_primary": {
            "n": int(len(reference_primary)),
            "mcp_observed_rho": rho(
                reference_primary.mcp_qet_r7,
                reference_primary.observed_median_r7,
            ),
            "mcp_observed_mae": median_absolute_difference(
                reference_primary.mcp_qet_r7,
                reference_primary.observed_median_r7,
            ) if len(reference_primary) else np.nan,
            "supports_same_direction": reference_support,
        },
        "decision_components": {
            "architecture_disagreement": architecture_disagreement,
            "observed_agreement_fails": observed_failure,
            "reference_support": reference_support,
            "conditional_support": conditional_support,
        },
        "verdict": verdict,
        "interpretation": (
            "Global and conditional conclusions are separated. A weak MCP cannot be used as "
            "a universal counterexample merely because its response differs; conditional support "
            "is evaluated only where all three models have KGE >= 0.30 and an observed recession."
        ),
        "files": {
            "pairwise": str(pair_path.relative_to(EXPERIMENT)),
            "sensitivity": str(sensitivity_path.relative_to(EXPERIMENT)),
            "nested_pairs": str(nested_path.relative_to(EXPERIMENT)),
            "gauge_level": str(gauge_path.relative_to(EXPERIMENT)),
        },
    }
    summary_path = output_dir / "mcp_stress_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    checksums = {
        path.name: sha256(path)
        for path in (pair_path, sensitivity_path, nested_path, gauge_path, summary_path)
    }
    (output_dir / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(pd.DataFrame(sensitivity_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(nested_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
