#!/usr/bin/env python3
"""Synthesize the frozen multi-architecture conservation stress test.

This script is deliberately downstream of prediction and intervention. It
does not train or tune a model. It combines the five new architectures with
the archived dHBV, serial neural flux-gate, and 12-cell MCP results and reports
global competence separately from conditional response agreement.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Keep the post-processing step independent of PyTorch.  These are experiment
# design labels, not reported outcomes; model construction remains defined in
# ``conserved_models.py``.
MODEL_NAMES = ("mclstm_dr", "mc_node", "mcp_ma3", "mcp_ma4", "mcp_ma5")


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
MCP_EXPERIMENT = PROJECT / "independent_mcp_stress_test_2026-10-03"
SEEDS = (11, 29, 47)
THRESHOLDS = (0.0, 0.30, 0.40, 0.50, 0.60)
CLASS_BREAKS = (0.50, np.sqrt(0.50))
ARCHITECTURES = (
    "dhbv",
    "fluxgate",
    "mcp12",
    "mclstm_dr",
    "mc_node",
    "mcp_ma3",
    "mcp_ma4",
    "mcp_ma5",
)
FAMILIES = {
    "dhbv": "neural_parameterized_conceptual",
    "fluxgate": "serial_neural_flux_gate",
    "mcp12": "mass_conserving_perceptron",
    "mclstm_dr": "mass_conserving_lstm",
    "mc_node": "conservative_neural_ode",
    "mcp_ma3": "mass_conserving_perceptron",
    "mcp_ma4": "mass_conserving_perceptron",
    "mcp_ma5": "mass_conserving_perceptron",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(x, y, minimum: int = 10) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def mad(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.median(np.abs(x[valid] - y[valid]))) if valid.any() else np.nan


def retention_class(values) -> np.ndarray:
    """Assign persistence classes while preserving missing fingerprints."""
    values = np.asarray(values, dtype=float)
    classes = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    classes[valid] = np.digitize(values[valid], CLASS_BREAKS)
    return classes


def normalize_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    return frame


def load_new_fingerprints(anchor_design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return architecture medians and individual-seed R7 columns."""
    seed_tables = []
    for architecture in MODEL_NAMES:
        columns = []
        for seed in SEEDS:
            path = (
                EXPERIMENT
                / "outputs/interventions"
                / f"{architecture}_qet_seed{seed}_{anchor_design}_fingerprints.parquet"
            )
            table = normalize_id(pd.read_parquet(path, columns=["GAGE_ID", "median_r7"]))
            column = f"{architecture}_seed{seed}_r7"
            columns.append(column)
            seed_tables.append(table.rename(columns={"median_r7": column}).set_index("GAGE_ID"))
    seeds = pd.concat(seed_tables, axis=1)
    medians = pd.DataFrame(index=seeds.index)
    for architecture in MODEL_NAMES:
        columns = [f"{architecture}_seed{seed}_r7" for seed in SEEDS]
        medians[f"{architecture}_r7"] = seeds[columns].median(axis=1)
    return medians.reset_index(), seeds.reset_index()


def load_mcp12_fingerprints(anchor_design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_tables = []
    for seed in SEEDS:
        path = (
            MCP_EXPERIMENT
            / "outputs/interventions"
            / f"mcp_qet_seed{seed}_{anchor_design}_fingerprints.parquet"
        )
        table = normalize_id(pd.read_parquet(path, columns=["GAGE_ID", "median_r7"]))
        seed_tables.append(
            table.rename(columns={"median_r7": f"mcp12_seed{seed}_r7"}).set_index("GAGE_ID")
        )
    seeds = pd.concat(seed_tables, axis=1)
    medians = seeds.median(axis=1).rename("mcp12_r7").reset_index()
    return medians, seeds.reset_index()


def load_paper_common() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = normalize_id(
        pd.read_parquet(
            PROJECT / "outputs/identification/strengthened_test/common_date_flux_trace.parquet",
            columns=["architecture", "seed", "GAGE_ID", "r7"],
        )
    )
    seed_values = (
        trace.groupby(["architecture", "seed", "GAGE_ID"], as_index=False).r7.median()
    )
    wide_seed = seed_values.pivot(index="GAGE_ID", columns=["architecture", "seed"], values="r7")
    wide_seed.columns = [f"{architecture}_seed{seed}_r7" for architecture, seed in wide_seed.columns]
    wide_seed = wide_seed.reset_index()
    median = (
        seed_values.groupby(["architecture", "GAGE_ID"], as_index=False).r7.median()
        .pivot(index="GAGE_ID", columns="architecture", values="r7")
        .rename(columns={"dhbv": "dhbv_r7", "fluxgate": "fluxgate_r7"})
        .reset_index()
    )
    return median, wide_seed


def build_response_frame(anchor_design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = normalize_id(
        pd.read_parquet(PROJECT / "outputs/identification/gauge_level_identification.parquet")
    )
    metadata = base[
        [
            "GAGE_ID", "ecoregion", "tier_short", "reference_status", "latitude",
            "longitude", "area_km2", "observed_events", "observed_median_r7",
            "observed_usable", "dhbv_qet_kge", "fluxgate_qet_kge",
        ]
    ].rename(columns={"dhbv_qet_kge": "dhbv_kge", "fluxgate_qet_kge": "fluxgate_kge"})

    if anchor_design == "model_specific":
        old_median = base[
            ["GAGE_ID", "dhbv_qet_r7", "fluxgate_qet_r7"]
        ].rename(columns={"dhbv_qet_r7": "dhbv_r7", "fluxgate_qet_r7": "fluxgate_r7"})
        old_seeds = base[
            ["GAGE_ID"]
            + [f"{architecture}_qet_seed{seed}_r7" for architecture in ("dhbv", "fluxgate") for seed in SEEDS]
        ].rename(
            columns={
                f"{architecture}_qet_seed{seed}_r7": f"{architecture}_seed{seed}_r7"
                for architecture in ("dhbv", "fluxgate") for seed in SEEDS
            }
        )
    else:
        old_median, old_seeds = load_paper_common()

    mcp_median, mcp_seeds = load_mcp12_fingerprints(anchor_design)
    new_median, new_seeds = load_new_fingerprints(anchor_design)
    frame = metadata.merge(old_median, on="GAGE_ID", validate="one_to_one")
    frame = frame.merge(mcp_median, on="GAGE_ID", how="left", validate="one_to_one")
    frame = frame.merge(new_median, on="GAGE_ID", how="left", validate="one_to_one")

    mcp_gauge = normalize_id(
        pd.read_parquet(
            MCP_EXPERIMENT / "outputs/comparison/gauge_level_mcp_comparison.parquet",
            columns=["GAGE_ID", "mcp_qet_kge"],
        )
    ).rename(columns={"mcp_qet_kge": "mcp12_kge"})
    frame = frame.merge(mcp_gauge, on="GAGE_ID", how="left", validate="one_to_one")
    new_skill = normalize_id(
        pd.read_parquet(EXPERIMENT / "outputs/evaluation/target_model_metrics.parquet")
    )
    new_skill = (
        new_skill.loc[new_skill.ensemble, ["GAGE_ID", "architecture", "kge"]]
        .pivot(index="GAGE_ID", columns="architecture", values="kge")
        .rename(columns={architecture: f"{architecture}_kge" for architecture in MODEL_NAMES})
        .reset_index()
    )
    frame = frame.merge(new_skill, on="GAGE_ID", how="left", validate="one_to_one")

    seeds = old_seeds.merge(mcp_seeds, on="GAGE_ID", how="outer", validate="one_to_one")
    seeds = seeds.merge(new_seeds, on="GAGE_ID", how="outer", validate="one_to_one")
    return frame, seeds


def architecture_summary(frame: pd.DataFrame, seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURES:
        kge = frame[f"{architecture}_kge"].to_numpy(float)
        r7 = frame[f"{architecture}_r7"].to_numpy(float)
        conditional = frame.observed_usable.to_numpy(bool) & (kge >= 0.30) & np.isfinite(r7)
        reference = conditional & frame.reference_status.eq("Reference").to_numpy(bool)
        seed_columns = [f"{architecture}_seed{seed}_r7" for seed in SEEDS]
        seed_rhos, seed_mads = [], []
        for first, second in itertools.combinations(seed_columns, 2):
            x = seed_frame.set_index("GAGE_ID").loc[frame.GAGE_ID, first].to_numpy(float)
            y = seed_frame.set_index("GAGE_ID").loc[frame.GAGE_ID, second].to_numpy(float)
            seed_rhos.append(rho(x[conditional], y[conditional]))
            seed_mads.append(mad(x[conditional], y[conditional]))
        rows.append(
            {
                "architecture": architecture,
                "family": FAMILIES[architecture],
                "targets": int(len(frame)),
                "median_kge": float(np.nanmedian(kge)),
                "q25_kge": float(np.nanquantile(kge, 0.25)),
                "fraction_kge_ge_03": float(np.nanmean(kge >= 0.30)),
                "fraction_kge_ge_04": float(np.nanmean(kge >= 0.40)),
                "fraction_kge_ge_05": float(np.nanmean(kge >= 0.50)),
                "fraction_kge_ge_06": float(np.nanmean(kge >= 0.60)),
                "passes_global_competence": bool(
                    np.nanmedian(kge) >= 0.40 and np.nanmean(kge >= 0.30) >= 0.60
                ),
                "conditional_n": int(conditional.sum()),
                "observed_rho": rho(r7[conditional], frame.observed_median_r7.to_numpy(float)[conditional]),
                "observed_mae": mad(r7[conditional], frame.observed_median_r7.to_numpy(float)[conditional]),
                "reference_n": int(reference.sum()),
                "reference_observed_rho": rho(r7[reference], frame.observed_median_r7.to_numpy(float)[reference]),
                "reference_observed_mae": mad(r7[reference], frame.observed_median_r7.to_numpy(float)[reference]),
                "median_seed_pair_rho": float(np.nanmedian(seed_rhos)),
                "median_seed_pair_mad": float(np.nanmedian(seed_mads)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_summary(frame: pd.DataFrame, anchor_design: str) -> pd.DataFrame:
    rows = []
    observed = frame.observed_median_r7.to_numpy(float)
    observed_usable = frame.observed_usable.to_numpy(bool)
    for population in ("all", "reference"):
        population_mask = np.ones(len(frame), dtype=bool)
        if population == "reference":
            population_mask = frame.reference_status.eq("Reference").to_numpy(bool)
        for threshold in THRESHOLDS:
            for first, second in itertools.combinations(ARCHITECTURES, 2):
                x = frame[f"{first}_r7"].to_numpy(float)
                y = frame[f"{second}_r7"].to_numpy(float)
                mask = (
                    population_mask
                    & observed_usable
                    & (frame[f"{first}_kge"].to_numpy(float) >= threshold)
                    & (frame[f"{second}_kge"].to_numpy(float) >= threshold)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )
                n = int(mask.sum())
                cross_rho = rho(x[mask], y[mask])
                cross_mad = mad(x[mask], y[mask])
                rows.append(
                    {
                        "anchor_design": anchor_design,
                        "population": population,
                        "skill_threshold": threshold,
                        "first": first,
                        "second": second,
                        "first_family": FAMILIES[first],
                        "second_family": FAMILIES[second],
                        "same_family": FAMILIES[first] == FAMILIES[second],
                        "n": n,
                        "cross_architecture_rho": cross_rho,
                        "median_absolute_r7_difference": cross_mad,
                        "same_retention_class_fraction": (
                            float(np.mean(retention_class(x[mask]) == retention_class(y[mask])))
                            if n else np.nan
                        ),
                        "first_observed_rho": rho(x[mask], observed[mask]),
                        "second_observed_rho": rho(y[mask], observed[mask]),
                        "first_observed_mae": mad(x[mask], observed[mask]),
                        "second_observed_mae": mad(y[mask], observed[mask]),
                        "material_disagreement": bool(
                            n >= 10 and (cross_rho < 0.75 or cross_mad > 0.05)
                        ) if np.isfinite(cross_rho) and np.isfinite(cross_mad) else False,
                    }
                )
    return pd.DataFrame(rows)


def multimodel_span(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS[1:]:
        values = np.column_stack([frame[f"{architecture}_r7"] for architecture in ARCHITECTURES])
        skill = np.column_stack([frame[f"{architecture}_kge"] >= threshold for architecture in ARCHITECTURES])
        valid = skill & np.isfinite(values)
        masked = np.where(valid, values, np.nan)
        count = valid.sum(axis=1)
        span = np.full(len(frame), np.nan, dtype=float)
        iqr = np.full(len(frame), np.nan, dtype=float)
        evaluable = count > 0
        span[evaluable] = (
            np.nanmax(masked[evaluable], axis=1)
            - np.nanmin(masked[evaluable], axis=1)
        )
        iqr[evaluable] = (
            np.nanquantile(masked[evaluable], 0.75, axis=1)
            - np.nanquantile(masked[evaluable], 0.25, axis=1)
        )
        for population in ("all", "reference"):
            mask = frame.observed_usable.to_numpy(bool) & (count >= 3)
            if population == "reference":
                mask &= frame.reference_status.eq("Reference").to_numpy(bool)
            rows.append(
                {
                    "population": population,
                    "skill_threshold": threshold,
                    "minimum_models": 3,
                    "n": int(mask.sum()),
                    "median_number_models": float(np.median(count[mask])) if mask.any() else np.nan,
                    "median_r7_span": float(np.nanmedian(span[mask])) if mask.any() else np.nan,
                    "fraction_span_le_005": float(np.nanmean(span[mask] <= 0.05)) if mask.any() else np.nan,
                    "median_r7_iqr": float(np.nanmedian(iqr[mask])) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def nested_summary(frame: pd.DataFrame) -> pd.DataFrame:
    edges = pd.read_csv(
        ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv",
        dtype=str,
    )
    edges["upstream_gage"] = edges.upstream_gage.astype(str).str.zfill(8)
    edges["downstream_gage"] = edges.downstream_gage.astype(str).str.zfill(8)
    indexed = frame.set_index("GAGE_ID")
    rows = []
    for threshold in (0.30, 0.60):
        for architecture in ARCHITECTURES:
            eligible = frame.loc[
                frame.observed_usable
                & (frame[f"{architecture}_kge"] >= threshold)
                & frame[f"{architecture}_r7"].notna(),
                "GAGE_ID",
            ]
            eligible = set(eligible)
            pairs = edges.loc[
                edges.upstream_gage.isin(eligible) & edges.downstream_gage.isin(eligible)
            ]
            upstream = indexed.loc[pairs.upstream_gage]
            downstream = indexed.loc[pairs.downstream_gage]
            model_contrast = (
                downstream[f"{architecture}_r7"].to_numpy(float)
                - upstream[f"{architecture}_r7"].to_numpy(float)
            )
            observed_contrast = (
                downstream.observed_median_r7.to_numpy(float)
                - upstream.observed_median_r7.to_numpy(float)
            )
            rows.append(
                {
                    "architecture": architecture,
                    "skill_threshold": threshold,
                    "pairs": int(len(pairs)),
                    "upstream_downstream_rho": rho(
                        upstream[f"{architecture}_r7"], downstream[f"{architecture}_r7"]
                    ),
                    "contrast_vs_observed_rho": rho(model_contrast, observed_contrast),
                    "contrast_vs_observed_mae": mad(model_contrast, observed_contrast),
                }
            )
    return pd.DataFrame(rows)


def mass_balance_summary() -> pd.DataFrame:
    rows = []
    for architecture in MODEL_NAMES:
        for seed in SEEDS:
            receipt = json.loads(
                (EXPERIMENT / "runs/frozen_refit" / f"{architecture}_qet_seed{seed}" / "receipt.json").read_text()
            )
            intervention_receipts = [
                json.loads(
                    (EXPERIMENT / "outputs/interventions" / f"{architecture}_qet_seed{seed}_{design}_receipt.json").read_text()
                )
                for design in ("model_specific", "common_date")
            ]
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "refit_daily_max_abs_mm": receipt["mass_balance"]["daily_max_abs_mm"],
                    "refit_cumulative_max_abs_mm": receipt["mass_balance"]["cumulative_max_abs_mm"],
                    "intervention_daily_max_abs_mm": max(
                        item["maximum_daily_mass_residual_mm"] for item in intervention_receipts
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["passes_0_005_mm_gate"] = (
        result[["refit_daily_max_abs_mm", "intervention_daily_max_abs_mm"]].max(axis=1)
        <= 0.005
    )
    return result


def main() -> None:
    output_dir = EXPERIMENT / "outputs/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_frame, model_seeds = build_response_frame("model_specific")
    common_frame, common_seeds = build_response_frame("common_date")
    architecture = architecture_summary(model_frame, model_seeds)
    common_architecture = architecture_summary(common_frame, common_seeds)
    pairwise = pd.concat(
        [
            pairwise_summary(model_frame, "model_specific"),
            pairwise_summary(common_frame, "common_date"),
        ],
        ignore_index=True,
    )
    spans = multimodel_span(model_frame)
    nested = nested_summary(model_frame)
    mass = mass_balance_summary()

    primary_pairs = pairwise.loc[
        pairwise.anchor_design.eq("model_specific")
        & pairwise.population.eq("all")
        & pairwise.skill_threshold.eq(0.30)
        & (pairwise.n >= 500)
    ]
    high_skill_pairs = pairwise.loc[
        pairwise.anchor_design.eq("model_specific")
        & pairwise.population.eq("all")
        & pairwise.skill_threshold.eq(0.60)
        & (pairwise.n >= 30)
    ]
    reference_pairs = pairwise.loc[
        pairwise.anchor_design.eq("model_specific")
        & pairwise.population.eq("reference")
        & pairwise.skill_threshold.eq(0.30)
        & (pairwise.n >= 30)
    ]

    new_support = []
    for model in MODEL_NAMES:
        relevant = primary_pairs.loc[(primary_pairs["first"] == model) | (primary_pairs["second"] == model)]
        reference_relevant = reference_pairs.loc[
            (reference_pairs["first"] == model) | (reference_pairs["second"] == model)
        ]
        row = architecture.loc[architecture.architecture.eq(model)].iloc[0]
        observed_failure = bool(row.observed_rho < 0.60 or row.observed_mae > 0.10)
        new_support.append(
            {
                "architecture": model,
                "passes_global_competence": bool(row.passes_global_competence),
                "primary_pairs_n_ge_500": int(len(relevant)),
                "primary_pairs_materially_disagree": int(relevant.material_disagreement.sum()),
                "reference_pairs_n_ge_30": int(len(reference_relevant)),
                "reference_pairs_materially_disagree": int(reference_relevant.material_disagreement.sum()),
                "observed_identification_fails": observed_failure,
                "conditional_support": bool(
                    len(relevant) > 0
                    and relevant.material_disagreement.any()
                    and len(reference_relevant) > 0
                    and reference_relevant.material_disagreement.any()
                    and observed_failure
                ),
            }
        )
    new_support = pd.DataFrame(new_support)

    paths = {
        "architecture": output_dir / "architecture_summary.csv",
        "common_architecture": output_dir / "common_date_architecture_summary.csv",
        "pairwise": output_dir / "pairwise_response_agreement.csv",
        "spans": output_dir / "multimodel_span_summary.csv",
        "nested": output_dir / "nested_pair_summary.csv",
        "mass": output_dir / "mass_balance_summary.csv",
        "support": output_dir / "new_architecture_support.csv",
        "gauge": output_dir / "gauge_level_multimodel.parquet",
        "common_gauge": output_dir / "gauge_level_multimodel_common_date.parquet",
    }
    architecture.to_csv(paths["architecture"], index=False)
    common_architecture.to_csv(paths["common_architecture"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    spans.to_csv(paths["spans"], index=False)
    nested.to_csv(paths["nested"], index=False)
    mass.to_csv(paths["mass"], index=False)
    new_support.to_csv(paths["support"], index=False)
    model_frame.to_parquet(paths["gauge"], index=False)
    common_frame.to_parquet(paths["common_gauge"], index=False)

    globally_competent = architecture.loc[architecture.passes_global_competence, "architecture"].tolist()
    globally_competent_pairs = primary_pairs.loc[
        primary_pairs["first"].isin(globally_competent)
        & primary_pairs["second"].isin(globally_competent)
    ]
    summary = {
        "architectures_compared": list(ARCHITECTURES),
        "new_architectures": list(MODEL_NAMES),
        "target_gauges": int(len(model_frame)),
        "globally_competent_architectures": globally_competent,
        "new_globally_competent_count": int(
            architecture.loc[architecture.architecture.isin(MODEL_NAMES), "passes_global_competence"].sum()
        ),
        "new_exact_conservation_count": int(
            mass.groupby("architecture").passes_0_005_mm_gate.all().sum()
        ),
        "primary_pairwise": {
            "eligible_pairs_n_ge_500": int(len(primary_pairs)),
            "materially_disagreeing_pairs": int(primary_pairs.material_disagreement.sum()),
            "different_family_pairs": int((~primary_pairs.same_family).sum()),
            "different_family_materially_disagreeing_pairs": int(
                primary_pairs.loc[~primary_pairs.same_family, "material_disagreement"].sum()
            ),
        },
        "high_skill_pairwise": {
            "eligible_pairs_n_ge_30": int(len(high_skill_pairs)),
            "materially_disagreeing_pairs": int(high_skill_pairs.material_disagreement.sum()),
        },
        "reference_pairwise": {
            "eligible_pairs_n_ge_30": int(len(reference_pairs)),
            "materially_disagreeing_pairs": int(reference_pairs.material_disagreement.sum()),
        },
        "globally_competent_pairwise": {
            "eligible_pairs_n_ge_500": int(len(globally_competent_pairs)),
            "materially_disagreeing_pairs": int(globally_competent_pairs.material_disagreement.sum()),
        },
        "new_architecture_conditional_support_count": int(new_support.conditional_support.sum()),
        "interpretation_rule": (
            "Weak architectures may provide conditional sensitivity evidence but are not counted as "
            "globally competent counterexamples. Multiple MCP graphs are one architectural family."
        ),
        "files": {key: str(value.relative_to(EXPERIMENT)) for key, value in paths.items()},
    }
    summary_path = output_dir / "multimodel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    checksums = {path.name: sha256(path) for path in [*paths.values(), summary_path]}
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")

    print(architecture.to_string(index=False), flush=True)
    print(new_support.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
