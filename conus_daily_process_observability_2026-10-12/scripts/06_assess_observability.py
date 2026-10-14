#!/usr/bin/env python3
"""Apply the frozen process-observability decision rules without retuning."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
CURVE_SOURCE = SOURCE / "r7_curve_robustness_2026-10-09/outputs/natural_event_median_curves.npz"
EDGE_SOURCE = ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv"
ARCHITECTURES = ["dhbv", "fluxgate"]
RUNGS = ["q", "q_sm", "q_et", "q_sm_et"]
SEEDS = [11, 29, 47]
THRESHOLDS = [0.30, 0.40, 0.50, 0.60]
RESULTS = PROJECT / "results"
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_rho(x: np.ndarray, y: np.ndarray, minimum: int = 10) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum or np.nanstd(x[valid]) <= 0 or np.nanstd(y[valid]) <= 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def persistence_class(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, np.float32)
    valid = np.isfinite(values)
    out[valid] = np.digitize(values[valid], [0.50, np.sqrt(0.50)]).astype(np.float32)
    return out


def curve_gap(left: np.ndarray, right: np.ndarray, stop: int = 30) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(np.abs(left[:, 1:stop + 1] - right[:, 1:stop + 1]), axis=1)


def bootstrap_reduction(
    baseline: np.ndarray, candidate: np.ndarray, groups: np.ndarray, seed: int = 20261012,
) -> tuple[float, float]:
    valid = np.isfinite(baseline) & np.isfinite(candidate) & pd.notna(groups)
    baseline, candidate, groups = baseline[valid], candidate[valid], groups[valid]
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAPS, np.float64)
    for iteration in range(BOOTSTRAPS):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in members])
        denominator = np.nanmedian(baseline[sampled])
        estimates[iteration] = 1.0 - np.nanmedian(candidate[sampled]) / denominator if denominator > 0 else np.nan
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def load_runs(target_ids: np.ndarray):
    metrics, curves = {}, {}
    expected = set(target_ids.tolist())
    for architecture, rung, seed in itertools.product(ARCHITECTURES, RUNGS, SEEDS):
        run = f"{architecture}_{rung}_seed{seed}"
        output = PROJECT / "outputs" / run
        metric_path = output / "target_metrics.parquet"
        curve_path = output / "drainage_curves.npz"
        if not metric_path.exists() or not curve_path.exists():
            raise FileNotFoundError(f"Incomplete run {run}")
        metric = pd.read_parquet(metric_path)
        metric["GAGE_ID"] = metric.GAGE_ID.astype(str).str.zfill(8)
        metric = metric.set_index("GAGE_ID").reindex(target_ids)
        if set(metric.dropna(how="all").index) != expected:
            raise RuntimeError(f"Target metrics do not cover the frozen population: {run}")
        archive = np.load(curve_path)
        ids = np.char.zfill(archive["gage_ids"].astype(str), 8)
        if not np.array_equal(ids, target_ids):
            raise RuntimeError(f"Target curve order mismatch: {run}")
        metrics[(architecture, rung, seed)] = metric.reset_index()
        curves[(architecture, rung, seed)] = np.asarray(archive["median_normalized_q"], np.float32)
    return metrics, curves


def aggregate(metrics: dict, curves: dict):
    summary_metrics, summary_curves, seed_gap = {}, {}, {}
    for architecture, rung in itertools.product(ARCHITECTURES, RUNGS):
        frames = [metrics[(architecture, rung, seed)] for seed in SEEDS]
        numeric = [
            column for column in frames[0].columns
            if column != "GAGE_ID" and pd.api.types.is_numeric_dtype(frames[0][column])
        ]
        result = pd.DataFrame({"GAGE_ID": frames[0].GAGE_ID})
        for column in numeric:
            result[column] = np.nanmedian(
                np.stack([pd.to_numeric(frame[column], errors="coerce") for frame in frames]), axis=0
            )
        cube = np.stack([curves[(architecture, rung, seed)] for seed in SEEDS])
        summary_metrics[(architecture, rung)] = result
        summary_curves[(architecture, rung)] = np.nanmedian(cube, axis=0).astype(np.float32)
        pair_gaps = np.stack([
            curve_gap(cube[left], cube[right]) for left, right in itertools.combinations(range(3), 2)
        ])
        seed_gap[(architecture, rung)] = np.nanmedian(pair_gaps, axis=0)
    return summary_metrics, summary_curves, seed_gap


def process_learning(metrics: dict, summary_metrics: dict) -> pd.DataFrame:
    rows = []
    for architecture, rung in itertools.product(ARCHITECTURES, RUNGS):
        development = pd.concat([
            pd.read_parquet(PROJECT / "outputs" / f"{architecture}_{rung}_seed{seed}" / "development_process_metrics.parquet")
            for seed in SEEDS
        ], ignore_index=True)
        target = summary_metrics[(architecture, rung)]
        rows.append({
            "architecture": architecture, "rung": rung,
            "median_development_soil_spearman": float(development.soil_spearman.median()),
            "median_development_et_spearman": float(development.et_spearman.median()),
            "median_development_et_mae_month_mm": float(development.et_mae_month_mm.median()),
            "median_target_nldas_soil_spearman": float(target.soil_spearman.median()),
            "median_target_esa_cci_soil_spearman": float(target.esa_cci_soil_spearman.median()),
            "target_esa_cci_gauges": int(target.esa_cci_valid_weeks.ge(30).sum()),
        })
    frame = pd.DataFrame(rows)
    for metric in [
        "median_development_soil_spearman", "median_development_et_spearman",
        "median_target_nldas_soil_spearman", "median_target_esa_cci_soil_spearman",
    ]:
        frame[f"delta_vs_q__{metric}"] = frame.groupby("architecture")[metric].transform(
            lambda x: x - x.iloc[0]
        )
    metric = "median_development_et_mae_month_mm"
    frame[f"delta_vs_q__{metric}"] = frame.groupby("architecture")[metric].transform(
        lambda x: x - x.iloc[0]
    )
    return frame


def competence(summary_metrics: dict) -> pd.DataFrame:
    rows = []
    for architecture, rung in itertools.product(ARCHITECTURES, RUNGS):
        current = summary_metrics[(architecture, rung)].kge.to_numpy(float)
        baseline = summary_metrics[(architecture, "q")].kge.to_numpy(float)
        rows.append({
            "architecture": architecture, "rung": rung,
            "target_gauges": int(np.isfinite(current).sum()),
            "median_target_kge": float(np.nanmedian(current)),
            "fraction_target_kge_ge_0_30": float(np.nanmean(current >= 0.30)),
            "median_kge_change_vs_q": float(np.nanmedian(current) - np.nanmedian(baseline)),
            "within_0_05_of_q": bool(np.nanmedian(current) - np.nanmedian(baseline) >= -0.05),
        })
    return pd.DataFrame(rows)


def observed_arrays(target_ids: np.ndarray):
    archive = np.load(CURVE_SOURCE)
    source_ids = np.char.zfill(archive["gage_ids"].astype(str), 8)
    lookup = {gage: row for row, gage in enumerate(source_ids)}
    rows = np.asarray([lookup.get(gage, -1) for gage in target_ids], int)
    observed = np.full((len(target_ids), 8), np.nan, np.float32)
    found = rows >= 0
    observed[found] = archive["observed"][rows[found]]
    event_count = np.zeros(len(target_ids), np.int16)
    event_count[found] = archive["observed_events"][rows[found]]
    observed[event_count < 3] = np.nan
    return observed, event_count


def architecture_results(
    target: pd.DataFrame, summary_metrics: dict, summary_curves: dict,
    observed: np.ndarray, reference_only: bool = False,
) -> pd.DataFrame:
    rows = []
    subset = target.reference_status.eq("Reference").to_numpy() if reference_only else np.ones(len(target), bool)
    observed_r7 = observed[:, 7]
    observed_curve_valid = np.isfinite(observed[:, 0])
    for rung in RUNGS:
        left_metric = summary_metrics[("dhbv", rung)].kge.to_numpy(float)
        right_metric = summary_metrics[("fluxgate", rung)].kge.to_numpy(float)
        left_curve = summary_curves[("dhbv", rung)]
        right_curve = summary_curves[("fluxgate", rung)]
        gap = curve_gap(left_curve, right_curve)
        for threshold in THRESHOLDS:
            use = subset & (left_metric >= threshold) & (right_metric >= threshold) & np.isfinite(gap)
            l7, r7 = left_curve[:, 7], right_curve[:, 7]
            model_observed = use & observed_curve_valid
            model_obs_mae = np.nanmedian(np.concatenate([
                np.nanmean(np.abs(left_curve[model_observed, 1:8] - observed[model_observed, 1:8]), axis=1),
                np.nanmean(np.abs(right_curve[model_observed, 1:8] - observed[model_observed, 1:8]), axis=1),
            ])) if model_observed.any() else np.nan
            classes_left, classes_right = persistence_class(l7), persistence_class(r7)
            rows.append({
                "population": "reference" if reference_only else "all",
                "rung": rung, "skill_threshold": threshold,
                "mutually_qualified_gauges": int(use.sum()),
                "median_curve_gap_days_1_30": float(np.nanmedian(gap[use])) if use.any() else np.nan,
                "median_absolute_r7_gap": float(np.nanmedian(np.abs(l7[use] - r7[use]))) if use.any() else np.nan,
                "r7_cross_architecture_spearman": safe_rho(l7[use], r7[use]),
                "same_persistence_class_fraction": float(np.nanmean(classes_left[use] == classes_right[use])) if use.any() else np.nan,
                "observed_recession_gauges": int(model_observed.sum()),
                "median_model_observed_curve_mae_days_1_7": float(model_obs_mae),
                "dhbv_observed_r7_spearman": safe_rho(l7[model_observed], observed_r7[model_observed]),
                "fluxgate_observed_r7_spearman": safe_rho(r7[model_observed], observed_r7[model_observed]),
            })
    return pd.DataFrame(rows)


def paired_reductions(target: pd.DataFrame, summary_metrics: dict, summary_curves: dict) -> pd.DataFrame:
    q_gap = curve_gap(summary_curves[("dhbv", "q")], summary_curves[("fluxgate", "q")])
    q_left = summary_metrics[("dhbv", "q")].kge.to_numpy(float)
    q_right = summary_metrics[("fluxgate", "q")].kge.to_numpy(float)
    rows = []
    for rung in RUNGS[1:]:
        candidate_gap = curve_gap(summary_curves[("dhbv", rung)], summary_curves[("fluxgate", rung)])
        c_left = summary_metrics[("dhbv", rung)].kge.to_numpy(float)
        c_right = summary_metrics[("fluxgate", rung)].kge.to_numpy(float)
        for population, population_mask in [
            ("all", np.ones(len(target), bool)),
            ("reference", target.reference_status.eq("Reference").to_numpy()),
        ]:
            for threshold in THRESHOLDS:
                use = (
                    population_mask & (q_left >= threshold) & (q_right >= threshold)
                    & (c_left >= threshold) & (c_right >= threshold)
                    & np.isfinite(q_gap) & np.isfinite(candidate_gap)
                )
                q_median = float(np.nanmedian(q_gap[use])) if use.any() else np.nan
                candidate_median = float(np.nanmedian(candidate_gap[use])) if use.any() else np.nan
                reduction = 1.0 - candidate_median / q_median if q_median > 0 else np.nan
                low, high = bootstrap_reduction(
                    q_gap[use], candidate_gap[use], target.ecoregion.to_numpy()[use],
                    seed=20261012 + int(threshold * 100) + RUNGS.index(rung),
                ) if use.sum() >= 20 else (np.nan, np.nan)
                rows.append({
                    "population": population, "rung": rung, "skill_threshold": threshold,
                    "common_mutually_qualified_gauges": int(use.sum()),
                    "q_median_curve_gap": q_median,
                    "candidate_median_curve_gap": candidate_median,
                    "relative_reduction": reduction,
                    "reduction_bootstrap_low": low, "reduction_bootstrap_high": high,
                })
    return pd.DataFrame(rows)


def seed_results(target: pd.DataFrame, summary_metrics: dict, seed_gap: dict) -> pd.DataFrame:
    rows = []
    for architecture, rung in itertools.product(ARCHITECTURES, RUNGS):
        kge = summary_metrics[(architecture, rung)].kge.to_numpy(float)
        q_kge = summary_metrics[(architecture, "q")].kge.to_numpy(float)
        for population, population_mask in [
            ("all", np.ones(len(target), bool)),
            ("reference", target.reference_status.eq("Reference").to_numpy()),
        ]:
            for threshold in THRESHOLDS:
                use = (
                    population_mask & (kge >= threshold) & (q_kge >= threshold)
                    & np.isfinite(seed_gap[(architecture, rung)])
                )
                current = float(np.nanmedian(seed_gap[(architecture, rung)][use])) if use.any() else np.nan
                baseline_values = seed_gap[(architecture, "q")]
                common = use & np.isfinite(baseline_values)
                baseline = float(np.nanmedian(baseline_values[common])) if common.any() else np.nan
                reduction = 1.0 - current / baseline if baseline > 0 else np.nan
                rows.append({
                    "architecture": architecture, "population": population, "rung": rung,
                    "skill_threshold": threshold, "gauges": int(use.sum()),
                    "median_cross_seed_curve_gap": current,
                    "q_median_cross_seed_curve_gap_on_common_gauges": baseline,
                    "relative_reduction_vs_q": reduction,
                })
    return pd.DataFrame(rows)


def nested_results(target: pd.DataFrame, summary_metrics: dict, summary_curves: dict, observed: np.ndarray) -> pd.DataFrame:
    edges = pd.read_csv(EDGE_SOURCE, dtype=str)
    edges["upstream_gage"] = edges.upstream_gage.str.zfill(8)
    edges["downstream_gage"] = edges.downstream_gage.str.zfill(8)
    lookup = {gage: row for row, gage in enumerate(target.GAGE_ID)}
    edges = edges[edges.upstream_gage.isin(lookup) & edges.downstream_gage.isin(lookup)].copy()
    upstream = edges.upstream_gage.map(lookup).to_numpy(int)
    downstream = edges.downstream_gage.map(lookup).to_numpy(int)
    observed_r7 = observed[:, 7]
    observed_contrast = observed_r7[downstream] - observed_r7[upstream]
    rows = []
    for architecture, rung in itertools.product(ARCHITECTURES, RUNGS):
        r7 = summary_curves[(architecture, rung)][:, 7]
        kge = summary_metrics[(architecture, rung)].kge.to_numpy(float)
        model_contrast = r7[downstream] - r7[upstream]
        for threshold in THRESHOLDS:
            use = (
                (kge[upstream] >= threshold) & (kge[downstream] >= threshold)
                & np.isfinite(observed_contrast) & np.isfinite(model_contrast)
            )
            rows.append({
                "architecture": architecture, "rung": rung, "skill_threshold": threshold,
                "nested_pairs": int(use.sum()),
                "contrast_vs_observed_spearman": safe_rho(model_contrast[use], observed_contrast[use]),
                "contrast_vs_observed_mae": float(np.nanmedian(np.abs(model_contrast[use] - observed_contrast[use]))) if use.any() else np.nan,
            })
    return pd.DataFrame(rows)


def decide(
    rung: str, competence_frame: pd.DataFrame, process: pd.DataFrame,
    reductions: pd.DataFrame, architecture: pd.DataFrame, seeds: pd.DataFrame,
    nested: pd.DataFrame,
) -> dict:
    competence_rows = competence_frame[competence_frame.rung.eq(rung)]
    competent = bool(competence_rows.within_0_05_of_q.all() and competence_rows.median_target_kge.ge(0.30).all())
    process_rows = process[process.rung.eq(rung)]
    soil_target = "sm" in rung
    et_target = "et" in rung
    soil_learned = bool(
        (process_rows["delta_vs_q__median_development_soil_spearman"] > 0).all()
        and (process_rows["delta_vs_q__median_target_nldas_soil_spearman"] > 0).all()
    ) if soil_target else True
    soil_independent = bool(
        (process_rows["delta_vs_q__median_target_esa_cci_soil_spearman"] > 0).all()
    ) if soil_target else True
    et_learned = bool(
        (process_rows["delta_vs_q__median_development_et_spearman"] > 0).all()
        and (process_rows["delta_vs_q__median_development_et_mae_month_mm"] < 0).all()
    ) if et_target else True
    primary = reductions[
        reductions.population.eq("all") & reductions.rung.eq(rung) & reductions.skill_threshold.eq(0.30)
    ].iloc[0]
    curve_pass = bool(primary.relative_reduction >= 0.30 and primary.reduction_bootstrap_low > 0)
    threshold_direction = bool((reductions[
        reductions.population.eq("all") & reductions.rung.eq(rung)
    ].relative_reduction > 0).all())
    reference_direction = bool((reductions[
        reductions.population.eq("reference") & reductions.rung.eq(rung)
    ].relative_reduction > 0).all())
    seed_primary = seeds[
        seeds.population.eq("all") & seeds.rung.eq(rung) & seeds.skill_threshold.eq(0.30)
    ]
    seed_pass = bool(seed_primary.relative_reduction_vs_q.ge(0.30).all())
    response = architecture[
        architecture.population.eq("all") & architecture.rung.eq(rung) & architecture.skill_threshold.eq(0.30)
    ].iloc[0]
    rank_pass = bool(response.r7_cross_architecture_spearman >= 0.75)
    q_response = architecture[
        architecture.population.eq("all") & architecture.rung.eq("q") & architecture.skill_threshold.eq(0.30)
    ].iloc[0]
    observed_improves = bool(
        response.median_model_observed_curve_mae_days_1_7 < q_response.median_model_observed_curve_mae_days_1_7
    )
    current_nested = nested[nested.rung.eq(rung) & nested.skill_threshold.eq(0.30)]
    q_nested = nested[nested.rung.eq("q") & nested.skill_threshold.eq(0.30)].set_index("architecture")
    nested_improves = bool(all(
        row.contrast_vs_observed_spearman > q_nested.loc[row.architecture, "contrast_vs_observed_spearman"]
        and row.contrast_vs_observed_mae < q_nested.loc[row.architecture, "contrast_vs_observed_mae"]
        for row in current_nested.itertuples(index=False)
    ))
    every = all([
        competent, soil_learned, soil_independent, et_learned, curve_pass, seed_pass,
        rank_pass, observed_improves, nested_improves, threshold_direction, reference_direction,
    ])
    process_learned = soil_learned and et_learned
    if not competent:
        verdict = "inconclusive"
    elif every:
        verdict = "evidence for a remedy"
    elif process_learned and primary.relative_reduction > 0:
        verdict = "partial constraint"
    else:
        verdict = "no remedy in this design"
    return {
        "rung": rung, "verdict": verdict, "competence_pass": competent,
        "soil_product_learning_pass": soil_learned,
        "independent_esa_cci_pass": soil_independent,
        "et_learning_pass": et_learned,
        "curve_reduction_30_percent_and_ci_pass": curve_pass,
        "cross_seed_reduction_pass": seed_pass,
        "r7_rank_pass": rank_pass,
        "observed_curve_improvement_pass": observed_improves,
        "nested_contrast_improvement_pass": nested_improves,
        "all_skill_thresholds_direction_pass": threshold_direction,
        "reference_direction_pass": reference_direction,
        "primary_relative_curve_gap_reduction": float(primary.relative_reduction),
        "primary_reduction_95_interval": [
            float(primary.reduction_bootstrap_low), float(primary.reduction_bootstrap_high),
        ],
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(PROJECT / "data/frozen/station_index.csv", dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    target = stations[stations.role.eq("target")].reset_index(drop=True)
    target_ids = target.GAGE_ID.to_numpy(str)
    metrics, curves = load_runs(target_ids)
    summary_metrics, summary_curves, seed_gap = aggregate(metrics, curves)
    process = process_learning(metrics, summary_metrics)
    competence_frame = competence(summary_metrics)
    observed, observed_events = observed_arrays(target_ids)
    architecture = pd.concat([
        architecture_results(target, summary_metrics, summary_curves, observed, False),
        architecture_results(target, summary_metrics, summary_curves, observed, True),
    ], ignore_index=True)
    reductions = paired_reductions(target, summary_metrics, summary_curves)
    seeds = seed_results(target, summary_metrics, seed_gap)
    nested = nested_results(target, summary_metrics, summary_curves, observed)

    tables = {
        "model_competence.csv": competence_frame,
        "process_learning.csv": process,
        "architecture_gap_by_skill.csv": architecture,
        "rung_vs_q_paired_reduction.csv": reductions,
        "seed_stability.csv": seeds,
        "nested_contrast.csv": nested,
    }
    for name, table in tables.items():
        table.to_csv(RESULTS / name, index=False)
    decisions = [decide(rung, competence_frame, process, reductions, architecture, seeds, nested) for rung in RUNGS[1:]]
    receipt = {
        "protocol": "protocols/frozen_protocol.md",
        "runs_evaluated": len(ARCHITECTURES) * len(RUNGS) * len(SEEDS),
        "target_gauges": len(target),
        "observed_recession_gauges_with_three_events": int((observed_events >= 3).sum()),
        "decisions": decisions,
        "outputs": {name: sha256(RESULTS / name) for name in tables},
    }
    (RESULTS / "decision.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
