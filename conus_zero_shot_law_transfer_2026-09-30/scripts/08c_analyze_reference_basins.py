#!/usr/bin/env python3
"""Expand the prespecified GAGES-II reference-basin robustness test.

This post-evaluation analysis does not refit any hydrologic model. It reads the
frozen gauge-level predictions and intervention fingerprints, quantifies
uncertainty within reference and quality-tier subsets, and compares reference
gauges with non-reference controls matched without replacement within
ecoregion and GAGES-II quality tier. Matching never uses a law fingerprint or
observed recession outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs/identification/reference_expansion"
SEED = 20261001
BOOTSTRAP_DRAWS = 1000
SKILL_THRESHOLDS = (0.0, 0.30, 0.40, 0.50, 0.60)

METRIC_COLUMNS = (
    "q_cross_architecture_rho",
    "q_cross_architecture_mad",
    "qet_cross_architecture_rho",
    "qet_cross_architecture_mad",
    "disagreement_reduction_fraction",
    "dhbv_observed_rho",
    "dhbv_observed_mae",
    "fluxgate_observed_rho",
    "fluxgate_observed_mae",
    "all_model_class_agreement_fraction",
)

ATTRIBUTE_COLUMNS = {
    "Climate__PPTAVG_BASIN": "pptavg_basin",
    "Climate__T_AVG_BASIN": "tavg_basin",
    "Climate__PET": "pet",
    "Climate__SNOW_PCT_PRECIP": "snow_pct_precip",
    "Climate__PRECIP_SEAS_IND": "precip_seasonality",
}


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


def mad(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    difference = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.nanmedian(difference))


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
        "all_model_class_agreement_fraction": float(np.mean(frame.class_agreement)),
    }


def stratified_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = [group.index.to_numpy() for _, group in frame.groupby("ecoregion", sort=True)]
    return np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])


def bootstrap_group(frame: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    for draw in range(draws):
        chosen = stratified_indices(frame, rng)
        records.append({"draw": draw, **calculate_metrics(frame.loc[chosen])})
    return pd.DataFrame(records)


def group_summary(label: str, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    point = calculate_metrics(frame)
    bootstrap = bootstrap_group(frame)
    record = {"population": label, **point}
    for metric in METRIC_COLUMNS:
        lo, hi = bootstrap[metric].quantile([0.025, 0.975])
        record[f"{metric}_ci_low"] = float(lo)
        record[f"{metric}_ci_high"] = float(hi)
    bootstrap.insert(1, "population", label)
    return record, bootstrap


def bootstrap_unmatched_contrast(reference: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    records = []
    for draw in range(BOOTSTRAP_DRAWS):
        ref_metrics = calculate_metrics(reference.loc[stratified_indices(reference, rng)])
        control_metrics = calculate_metrics(control.loc[stratified_indices(control, rng)])
        record = {"draw": draw, "comparison": "unmatched_reference_minus_non_reference"}
        for metric in METRIC_COLUMNS:
            record[metric] = ref_metrics[metric] - control_metrics[metric]
        records.append(record)
    return pd.DataFrame(records)


def load_hydroclimate(primary: pd.DataFrame) -> pd.DataFrame:
    attributes = pd.read_csv(
        PROJECT / "data/interim/gagesii_all_attributes.csv.gz",
        dtype={"STAID": str},
        usecols=["STAID", *ATTRIBUTE_COLUMNS],
    ).rename(columns={"STAID": "GAGE_ID", **ATTRIBUTE_COLUMNS})
    attributes["GAGE_ID"] = attributes.GAGE_ID.str.zfill(8)
    frame = primary.merge(attributes, on="GAGE_ID", how="left", validate="one_to_one")
    frame["log10_area_km2"] = np.log10(frame.area_km2.clip(lower=0.01))
    if frame[[*ATTRIBUTE_COLUMNS.values(), "log10_area_km2"]].isna().any().any():
        missing = frame.loc[
            frame[[*ATTRIBUTE_COLUMNS.values(), "log10_area_km2"]].isna().any(axis=1), "GAGE_ID"
        ]
        raise RuntimeError(f"Hydroclimate attributes missing for {len(missing)} primary gauges")
    return frame


def robust_scaled(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    scaled = frame[columns].astype(float).copy()
    median = scaled.median()
    spread = scaled.quantile(0.75) - scaled.quantile(0.25)
    spread = spread.mask(spread <= 0, scaled.std()).replace(0, 1.0).fillna(1.0)
    return (scaled - median) / spread


def optimal_match(
    frame: pd.DataFrame,
    columns: list[str],
    design: str,
    feature_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Match every reference gauge to one unique control within eco-region and tier."""
    scaled = robust_scaled(frame, columns)
    feature_weights = feature_weights or {}
    weights = np.asarray([feature_weights.get(column, 1.0) for column in columns], dtype=float)
    records = []
    for (ecoregion, tier), stratum in frame.groupby(["ecoregion", "tier_short"], sort=True):
        reference = stratum[stratum.reference_status.eq("Reference")]
        control = stratum[stratum.reference_status.eq("Non-reference")]
        if reference.empty:
            continue
        if len(control) < len(reference):
            raise RuntimeError(f"Insufficient controls in {ecoregion}, {tier}")
        x = scaled.loc[reference.index, columns].to_numpy(float)
        y = scaled.loc[control.index, columns].to_numpy(float)
        cost = np.sqrt((((x[:, None, :] - y[None, :, :]) * weights) ** 2).sum(axis=2))
        row_index, column_index = linear_sum_assignment(cost)
        for row, column in zip(row_index, column_index, strict=True):
            ref_row = reference.iloc[row]
            control_row = control.iloc[column]
            records.append({
                "design": design,
                "ecoregion": ecoregion,
                "tier_short": tier,
                "reference_id": ref_row.GAGE_ID,
                "control_id": control_row.GAGE_ID,
                "distance": float(cost[row, column]),
            })
    matched = pd.DataFrame(records)
    expected = int(frame.reference_status.eq("Reference").sum())
    if len(matched) != expected or matched.control_id.nunique() != expected:
        raise RuntimeError("Matching failed to produce one unique control per reference gauge")
    return matched


def matched_metrics(frame: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict, dict]:
    indexed = frame.set_index("GAGE_ID", drop=False)
    reference = indexed.loc[pairs.reference_id].reset_index(drop=True)
    control = indexed.loc[pairs.control_id].reset_index(drop=True)
    return calculate_metrics(reference), calculate_metrics(control)


def bootstrap_matched_contrast(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    indexed = frame.set_index("GAGE_ID", drop=False)
    groups = [group.index.to_numpy() for _, group in pairs.groupby("ecoregion", sort=True)]
    records = []
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
        sample = pairs.loc[chosen]
        reference = indexed.loc[sample.reference_id].reset_index(drop=True)
        control = indexed.loc[sample.control_id].reset_index(drop=True)
        ref_metrics = calculate_metrics(reference)
        control_metrics = calculate_metrics(control)
        record = {"draw": draw, "comparison": f"{pairs.design.iloc[0]}_reference_minus_control"}
        for metric in METRIC_COLUMNS:
            record[metric] = ref_metrics[metric] - control_metrics[metric]
        records.append(record)
    return pd.DataFrame(records)


def contrast_summary(
    name: str,
    reference_metrics: dict,
    control_metrics: dict,
    bootstrap: pd.DataFrame,
) -> list[dict]:
    records = []
    for metric in METRIC_COLUMNS:
        difference = reference_metrics[metric] - control_metrics[metric]
        lo, hi = bootstrap[metric].quantile([0.025, 0.975])
        records.append({
            "comparison": name,
            "metric": metric,
            "reference_value": reference_metrics[metric],
            "control_value": control_metrics[metric],
            "reference_minus_control": difference,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "interval_excludes_zero": bool((lo > 0) or (hi < 0)),
        })
    return records


def standardized_difference(reference: pd.Series, control: pd.Series) -> float:
    reference = reference.astype(float)
    control = control.astype(float)
    pooled = np.sqrt((reference.var(ddof=1) + control.var(ddof=1)) / 2.0)
    return float((reference.mean() - control.mean()) / pooled) if pooled > 0 else 0.0


def balance_table(frame: pd.DataFrame, pairs: pd.DataFrame, columns: list[str], design: str) -> list[dict]:
    indexed = frame.set_index("GAGE_ID", drop=False)
    reference_all = frame[frame.reference_status.eq("Reference")]
    control_all = frame[frame.reference_status.eq("Non-reference")]
    reference_matched = indexed.loc[pairs.reference_id].reset_index(drop=True)
    control_matched = indexed.loc[pairs.control_id].reset_index(drop=True)
    records = []
    for column in columns:
        records.append({
            "design": design,
            "feature": column,
            "smd_before_matching": standardized_difference(reference_all[column], control_all[column]),
            "smd_after_matching": standardized_difference(reference_matched[column], control_matched[column]),
            "reference_mean": float(reference_matched[column].mean()),
            "control_mean": float(control_matched[column].mean()),
        })
    return records


def predictive_population_summary(frame: pd.DataFrame) -> pd.DataFrame:
    populations = {
        "All targets": np.ones(len(frame), dtype=bool),
        "Reference": frame.reference_status.eq("Reference"),
        "Non-reference": frame.reference_status.eq("Non-reference"),
        "Tier 1": frame.tier_short.eq("Tier 1"),
        "Tier 2": frame.tier_short.eq("Tier 2"),
        "Reference Tier 1": frame.reference_status.eq("Reference") & frame.tier_short.eq("Tier 1"),
        "Reference Tier 2": frame.reference_status.eq("Reference") & frame.tier_short.eq("Tier 2"),
    }
    records = []
    for population, mask in populations.items():
        subset = frame.loc[mask]
        joint = (
            subset.observed_usable
            & subset.dhbv_qet_kge.ge(0.30)
            & subset.fluxgate_qet_kge.ge(0.30)
        )
        records.append({
            "population": population,
            "target_gauges": len(subset),
            "observed_usable_gauges": int(subset.observed_usable.sum()),
            "primary_gauges": int(joint.sum()),
            "dhbv_median_kge": float(subset.dhbv_qet_kge.median()),
            "dhbv_fraction_kge_ge_030": float(subset.dhbv_qet_kge.ge(0.30).mean()),
            "fluxgate_median_kge": float(subset.fluxgate_qet_kge.median()),
            "fluxgate_fraction_kge_ge_030": float(subset.fluxgate_qet_kge.ge(0.30).mean()),
            "joint_fraction_kge_ge_030": float(
                (subset.dhbv_qet_kge.ge(0.30) & subset.fluxgate_qet_kge.ge(0.30)).mean()
            ),
        })
    return pd.DataFrame(records)


def skill_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for threshold in SKILL_THRESHOLDS:
        eligible = (
            frame.observed_usable
            & frame.dhbv_qet_kge.ge(threshold)
            & frame.fluxgate_qet_kge.ge(threshold)
        )
        for population, status in (("Reference", "Reference"), ("Non-reference", "Non-reference")):
            subset = frame[eligible & frame.reference_status.eq(status)]
            records.append({
                "skill_threshold": threshold,
                "population": population,
                **calculate_metrics(subset),
            })
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(PROJECT / "outputs/identification/gauge_level_identification.parquet")
    primary = pd.read_parquet(PROJECT / "outputs/identification/primary_identification_population.parquet")
    full["GAGE_ID"] = full.GAGE_ID.astype(str).str.zfill(8)
    primary["GAGE_ID"] = primary.GAGE_ID.astype(str).str.zfill(8)
    primary = load_hydroclimate(primary)

    group_frames = {
        "All primary": primary,
        "Reference": primary[primary.reference_status.eq("Reference")],
        "Non-reference": primary[primary.reference_status.eq("Non-reference")],
        "Tier 1": primary[primary.tier_short.eq("Tier 1")],
        "Tier 2": primary[primary.tier_short.eq("Tier 2")],
        "Reference Tier 1": primary[
            primary.reference_status.eq("Reference") & primary.tier_short.eq("Tier 1")
        ],
        "Reference Tier 2": primary[
            primary.reference_status.eq("Reference") & primary.tier_short.eq("Tier 2")
        ],
    }
    group_records, group_bootstraps = [], []
    for label, group in group_frames.items():
        record, bootstrap = group_summary(label, group.copy().reset_index(drop=True))
        group_records.append(record)
        group_bootstraps.append(bootstrap)
    group_summary_frame = pd.DataFrame(group_records)
    group_bootstrap_frame = pd.concat(group_bootstraps, ignore_index=True)

    reference = group_frames["Reference"].copy().reset_index(drop=True)
    non_reference = group_frames["Non-reference"].copy().reset_index(drop=True)
    unmatched_bootstrap = bootstrap_unmatched_contrast(reference, non_reference)
    unmatched_records = contrast_summary(
        "Unmatched reference minus non-reference",
        calculate_metrics(reference),
        calculate_metrics(non_reference),
        unmatched_bootstrap,
    )

    skill_columns = ["dhbv_qet_kge", "fluxgate_qet_kge"]
    hydroclimate_columns = [
        "log10_area_km2",
        "pptavg_basin",
        "tavg_basin",
        "pet",
        "snow_pct_precip",
        "precip_seasonality",
        *skill_columns,
    ]
    skill_pairs = optimal_match(primary, skill_columns, "skill_matched")
    hydro_match_weights = {"log10_area_km2": 4.0, "pptavg_basin": 2.0}
    hydro_pairs = optimal_match(
        primary,
        hydroclimate_columns,
        "hydroclimate_skill_matched",
        feature_weights=hydro_match_weights,
    )

    contrast_records = list(unmatched_records)
    contrast_bootstraps = [unmatched_bootstrap]
    balance_records = []
    for name, pairs, columns in (
        ("Skill-matched reference minus control", skill_pairs, skill_columns),
        ("Hydroclimate-and-skill-matched reference minus control", hydro_pairs, hydroclimate_columns),
    ):
        ref_metrics, control_metrics = matched_metrics(primary, pairs)
        bootstrap = bootstrap_matched_contrast(primary, pairs)
        contrast_records.extend(contrast_summary(name, ref_metrics, control_metrics, bootstrap))
        contrast_bootstraps.append(bootstrap)
        balance_records.extend(balance_table(primary, pairs, columns, pairs.design.iloc[0]))

    sensitivity = skill_sensitivity(full)
    predictive = predictive_population_summary(full)
    contrast_frame = pd.DataFrame(contrast_records)
    contrast_bootstrap_frame = pd.concat(contrast_bootstraps, ignore_index=True)
    pairs_frame = pd.concat([skill_pairs, hydro_pairs], ignore_index=True)
    balance_frame = pd.DataFrame(balance_records)

    paths = {
        "group_metrics": OUTPUT / "reference_group_metrics_with_ci.csv",
        "group_bootstrap": OUTPUT / "reference_group_bootstrap.csv.gz",
        "contrasts": OUTPUT / "reference_contrasts_with_ci.csv",
        "contrast_bootstrap": OUTPUT / "reference_contrast_bootstrap.csv.gz",
        "matched_pairs": OUTPUT / "reference_matched_pairs.csv",
        "balance": OUTPUT / "reference_match_balance.csv",
        "skill_sensitivity": OUTPUT / "reference_skill_sensitivity.csv",
        "predictive_population": OUTPUT / "reference_predictive_population.csv",
    }
    group_summary_frame.to_csv(paths["group_metrics"], index=False)
    group_bootstrap_frame.to_csv(paths["group_bootstrap"], index=False, compression="gzip")
    contrast_frame.to_csv(paths["contrasts"], index=False)
    contrast_bootstrap_frame.to_csv(paths["contrast_bootstrap"], index=False, compression="gzip")
    pairs_frame.to_csv(paths["matched_pairs"], index=False)
    balance_frame.to_csv(paths["balance"], index=False)
    sensitivity.to_csv(paths["skill_sensitivity"], index=False)
    predictive.to_csv(paths["predictive_population"], index=False)

    key_metrics = contrast_frame[
        contrast_frame.metric.isin(
            ["qet_cross_architecture_mad", "all_model_class_agreement_fraction"]
        )
    ].to_dict(orient="records")
    summary = {
        "analysis_status": "post_evaluation_expansion_of_prespecified_subgroups",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_strata": "aggregated ecoregion",
        "matching": {
            "exact_strata": ["aggregated ecoregion", "GAGES-II quality tier"],
            "replacement": False,
            "law_or_observed_recession_outcome_used": False,
            "reference_gauges_matched": int(len(skill_pairs)),
            "skill_features": skill_columns,
            "hydroclimate_and_skill_features": hydroclimate_columns,
            "hydroclimate_feature_weights": hydro_match_weights,
        },
        "key_contrasts": key_metrics,
        "files": {key: str(path.relative_to(PROJECT)) for key, path in paths.items()},
    }
    summary_path = OUTPUT / "reference_expansion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    paths["summary"] = summary_path

    receipt_path = PROJECT / "provenance/reference_expansion_checksums.json"
    receipt_path.write_text(
        json.dumps({path.name: sha256(path) for path in paths.values()}, indent=2) + "\n"
    )
    print(group_summary_frame.to_string(index=False), flush=True)
    print("\nReference-minus-control contrasts\n", flush=True)
    print(contrast_frame.to_string(index=False), flush=True)
    print("\nMatch balance\n", flush=True)
    print(balance_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
