#!/usr/bin/env python3
"""Open held-out reference discharge after freezing, then evaluate discovery."""

from __future__ import annotations

import hashlib
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DAILY = ROOT / "continental_run/07_matched_daily/daily"
MODELS = ("dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.0, 0.30, 0.40, 0.50, 0.60)
CLASS_BREAKS = (0.50, np.sqrt(0.50))
START = pd.Timestamp("2019-01-01")
END = pd.Timestamp("2024-12-31")
DATES = pd.date_range(START, END, freq="D")
SCENARIOS = {
    "primary": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.001},
    "strict_rain": {"future_p": 0.1, "prior_p": 0.5, "declines": 6, "q0": 0.001},
    "relaxed_rain": {"future_p": 2.0, "prior_p": 4.0, "declines": 0, "q0": 0.001},
    "low_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.0001},
    "high_q0": {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.01},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_registry() -> dict:
    expected = {f"{model}_qet_seed{seed}" for model in MODELS for seed in SEEDS}
    runs = []
    for path in sorted((EXPERIMENT / "runs/frozen_refit").glob("*/receipt.json")):
        receipt = json.loads(path.read_text())
        if receipt["run"] in expected:
            if receipt.get("target_discharge_accessed") is not False:
                raise RuntimeError(f"Outcome firewall failed for {receipt['run']}")
            prediction = EXPERIMENT / receipt["prediction"]
            intervention_receipt = EXPERIMENT / f"outputs/interventions/{receipt['run']}_common_date_receipt.json"
            if not intervention_receipt.exists():
                raise RuntimeError(f"Missing intervention for {receipt['run']}")
            intervention = json.loads(intervention_receipt.read_text())
            if intervention.get("target_discharge_accessed") is not False:
                raise RuntimeError(f"Intervention firewall failed for {receipt['run']}")
            if sha256(prediction) != receipt["prediction_sha256"]:
                raise RuntimeError(f"Prediction checksum failed for {receipt['run']}")
            runs.append(
                {
                    "run": receipt["run"], "model": receipt["model"], "seed": receipt["seed"],
                    "prediction": receipt["prediction"], "prediction_sha256": receipt["prediction_sha256"],
                    "fingerprints": intervention["fingerprints"],
                    "fingerprints_sha256": intervention["fingerprints_sha256"],
                }
            )
    found = {run["run"] for run in runs}
    if found != expected:
        raise RuntimeError(f"Incomplete frozen registry; missing={sorted(expected - found)}")
    registry = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": "413 held-out GAGES-II reference gauges, Tier 1 plus Tier 2",
        "runs": runs,
        "run_count": len(runs),
        "target_discharge_accessed": False,
    }
    path = EXPERIMENT / "provenance/frozen_prediction_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n")
    (EXPERIMENT / "provenance/frozen_prediction_registry.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return registry


def read_target_q(job: tuple[int, str]) -> tuple[int, np.ndarray]:
    row, gage_id = job
    frame = pd.read_csv(
        DAILY / f"{gage_id}.csv.gz", usecols=["date", "q_mm_day"], parse_dates=["date"]
    ).drop_duplicates("date").set_index("date").sort_index().reindex(DATES)
    return row, pd.to_numeric(frame.q_mm_day, errors="coerce").to_numpy(np.float32)


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator) if denominator > 0 else np.nan


def hydro_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return {"valid_days": int(valid.sum()), "kge": np.nan, "nse": np.nan, "log_nse": np.nan, "correlation": np.nan, "percent_bias": np.nan}
    o, p = observed[valid].astype(float), predicted[valid].astype(float)
    if (
        np.mean(o) <= 0
        or np.std(o, ddof=1) <= 0
        or np.std(p, ddof=1) <= 0
    ):
        return {"valid_days": int(valid.sum()), "kge": np.nan, "nse": np.nan, "log_nse": np.nan, "correlation": np.nan, "percent_bias": np.nan}
    r = np.corrcoef(o, p)[0, 1]
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(p) / np.mean(o)
    return {
        "valid_days": int(valid.sum()),
        "kge": float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)),
        "nse": nse(o, p),
        "log_nse": nse(np.log1p(o), np.log1p(p)),
        "correlation": float(r),
        "percent_bias": float(100.0 * (p.sum() - o.sum()) / o.sum()),
    }


def rho(x, y, minimum: int = 10) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < minimum or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def mad(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.median(np.abs(x[valid] - y[valid]))) if valid.any() else np.nan


def retention_class(value) -> np.ndarray:
    """Assign persistence classes while preserving missing fingerprints."""
    value = np.asarray(value, dtype=float)
    classes = np.full(value.shape, np.nan, dtype=float)
    valid = np.isfinite(value)
    classes[valid] = np.digitize(value[valid], CLASS_BREAKS)
    return classes


def events_for_basin(q: np.ndarray, forcing: np.ndarray, dates: pd.DatetimeIndex, scenario: dict) -> list[dict]:
    q_window = np.lib.stride_tricks.sliding_window_view(q, 8)
    p_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 0], 8)
    t_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 2], 8)
    starts = np.arange(2, len(q) - 7)
    local_q = q_window[starts]
    decline_count = np.sum(
        np.diff(local_q, axis=1) <= np.maximum(1e-8, 0.01 * local_q[:, :-1]), axis=1
    )
    prior_p = forcing[starts - 2, 0] + forcing[starts - 1, 0]
    month = dates.month.to_numpy()[starts]
    mask = (
        np.isfinite(local_q).all(axis=1)
        & (local_q[:, 0] >= scenario["q0"])
        & (month >= 6) & (month <= 9)
        & (np.nanmin(t_window[starts], axis=1) > 5.0)
        & (prior_p <= scenario["prior_p"])
        & (np.nansum(p_window[starts], axis=1) <= scenario["future_p"])
        & (decline_count >= scenario["declines"])
        & (local_q[:, -1] < local_q[:, 0])
    )
    selected, next_allowed = [], 0
    for start, segment in zip(starts[mask], local_q[mask]):
        if start < next_allowed:
            continue
        selected.append({"start_index": int(start), "r7": float(segment[-1] / segment[0]), "q0": float(segment[0])})
        next_allowed = int(start) + 8
    return selected


def extract_observed(q, forcing, dates, target):
    summary_records, event_records = [], []
    for basin, station in enumerate(target.itertuples(index=False)):
        local_forcing = np.asarray(forcing[basin, :, :3])
        for name, scenario in SCENARIOS.items():
            events = events_for_basin(np.asarray(q[basin]), local_forcing, dates, scenario)
            r7 = np.asarray([event["r7"] for event in events])
            summary_records.append(
                {
                    "GAGE_ID": station.GAGE_ID, "scenario": name, "events": len(events),
                    "observed_median_r7": float(np.median(r7)) if len(r7) else np.nan,
                    "observed_iqr_r7": float(np.quantile(r7, 0.75) - np.quantile(r7, 0.25)) if len(r7) >= 2 else np.nan,
                    "usable": len(events) >= 3,
                }
            )
            event_records.extend(
                {"GAGE_ID": station.GAGE_ID, "scenario": name, **event} for event in events
            )
    return pd.DataFrame(summary_records), pd.DataFrame(event_records)


def bootstrap_pair(frame: pd.DataFrame, first: str, second: str, threshold: float, iterations: int = 2000) -> dict:
    mask = (frame[f"{first}_kge"] >= threshold) & (frame[f"{second}_kge"] >= threshold)
    local = frame.loc[mask & frame[f"{first}_r7"].notna() & frame[f"{second}_r7"].notna()].copy()
    rng = np.random.default_rng(90210)
    by_region = [group.index.to_numpy() for _, group in local.groupby("ecoregion")]
    rhos, mads = [], []
    for _ in range(iterations):
        index = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_region])
        sample = local.loc[index]
        rhos.append(rho(sample[f"{first}_r7"], sample[f"{second}_r7"]))
        mads.append(mad(sample[f"{first}_r7"], sample[f"{second}_r7"]))
    return {
        "first": first, "second": second, "skill_threshold": threshold, "n": len(local),
        "rho": rho(local[f"{first}_r7"], local[f"{second}_r7"]),
        "rho_ci95": [float(np.nanquantile(rhos, 0.025)), float(np.nanquantile(rhos, 0.975))],
        "mad": mad(local[f"{first}_r7"], local[f"{second}_r7"]),
        "mad_ci95": [float(np.nanquantile(mads, 0.025)), float(np.nanquantile(mads, 0.975))],
        "bootstrap_iterations": iterations,
    }


def main() -> None:
    registry = write_registry()
    stations = pd.read_csv(EXPERIMENT / "data/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    if not target.reference_status.eq("Reference").all():
        raise RuntimeError("Non-reference target entered the reference-only evaluation")
    eval_start = int(np.searchsorted(np.load(EXPERIMENT / "data/dates.npy"), np.datetime64(START)))
    forcing_all = np.load(EXPERIMENT / "data/forcing.npy", mmap_mode="r")
    forcing = np.asarray(forcing_all[target.row.to_numpy(int), eval_start:])

    q_path = EXPERIMENT / "data/q_target_2019_2024.npy"
    q = np.lib.format.open_memmap(q_path, mode="w+", dtype=np.float32, shape=(len(target), len(DATES)))
    q[:] = np.nan
    jobs = [(index, gage) for index, gage in enumerate(target.GAGE_ID)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(read_target_q, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            row, value = future.result()
            q[row] = value
            if count % 100 == 0 or count == len(futures):
                print(f"held-out reference discharge {count}/{len(futures)}", flush=True)
    q.flush()

    predictions: dict[str, np.ndarray] = {}
    by_model: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    for run in registry["runs"]:
        prediction = np.load(EXPERIMENT / run["prediction"], mmap_mode="r")
        predictions[run["run"]] = prediction
        by_model[run["model"]].append(prediction)
    ensemble = {model: np.mean(np.stack(values), axis=0) for model, values in by_model.items()}

    metric_records = []
    for run in registry["runs"]:
        prediction = predictions[run["run"]]
        for basin, station in enumerate(target.itertuples(index=False)):
            metric_records.append(
                {"GAGE_ID": station.GAGE_ID, "model": run["model"], "seed": run["seed"], "ensemble": False,
                 **hydro_metrics(q[basin], prediction[basin])}
            )
    for model, prediction in ensemble.items():
        for basin, station in enumerate(target.itertuples(index=False)):
            metric_records.append(
                {"GAGE_ID": station.GAGE_ID, "model": model, "seed": -1, "ensemble": True,
                 **hydro_metrics(q[basin], prediction[basin])}
            )
    metrics = pd.DataFrame(metric_records).merge(
        target[["GAGE_ID", "ecoregion", "tier_short", "latitude", "longitude", "area_km2"]],
        on="GAGE_ID", validate="many_to_one",
    )
    output = EXPERIMENT / "outputs/evaluation"
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "target_model_metrics.csv.gz", index=False, compression="gzip")
    ensemble_metrics = metrics.loc[metrics.ensemble].copy()
    skill_summary = (
        ensemble_metrics.groupby("model", as_index=False)
        .agg(
            gauges=("GAGE_ID", "size"), median_kge=("kge", "median"), q25_kge=("kge", lambda x: x.quantile(0.25)),
            fraction_kge_ge_03=("kge", lambda x: np.mean(x >= 0.30)),
            fraction_kge_ge_04=("kge", lambda x: np.mean(x >= 0.40)),
            fraction_kge_ge_05=("kge", lambda x: np.mean(x >= 0.50)),
            fraction_kge_ge_06=("kge", lambda x: np.mean(x >= 0.60)),
            median_nse=("nse", "median"), median_log_nse=("log_nse", "median"),
            median_correlation=("correlation", "median"), median_absolute_percent_bias=("percent_bias", lambda x: np.nanmedian(np.abs(x))),
        )
    )
    skill_summary["passes_global_competence"] = (skill_summary.median_kge >= 0.40) & (skill_summary.fraction_kge_ge_03 >= 0.60)
    skill_summary.to_csv(output / "skill_summary.csv", index=False)

    observed_summary, observed_events = extract_observed(q, forcing, DATES, target)
    observed_summary.to_csv(output / "observed_recession_summary.csv.gz", index=False, compression="gzip")
    observed_events.to_csv(output / "observed_recession_events.csv.gz", index=False, compression="gzip")
    primary_observed = observed_summary.loc[observed_summary.scenario.eq("primary")].set_index("GAGE_ID")

    proxy_rows = []
    observed_wide = observed_summary.pivot(index="GAGE_ID", columns="scenario", values="observed_median_r7")
    usable_wide = observed_summary.pivot(index="GAGE_ID", columns="scenario", values="usable")
    for first, second in itertools.combinations(SCENARIOS, 2):
        mask = usable_wide[first].fillna(False) & usable_wide[second].fillna(False)
        proxy_rows.append(
            {"first": first, "second": second, "n": int(mask.sum()),
             "spearman_rho": rho(observed_wide.loc[mask, first], observed_wide.loc[mask, second]),
             "median_absolute_difference": mad(observed_wide.loc[mask, first], observed_wide.loc[mask, second])}
        )
    proxy_reliability = pd.DataFrame(proxy_rows)
    proxy_reliability.to_csv(output / "observed_proxy_reliability.csv", index=False)

    frame = target[["GAGE_ID", "ecoregion", "tier_short"]].copy().set_index("GAGE_ID")
    for model in MODELS:
        local_skill = ensemble_metrics.loc[ensemble_metrics.model.eq(model)].set_index("GAGE_ID")
        frame[f"{model}_kge"] = local_skill.kge
        seed_values = []
        for seed in SEEDS:
            table = pd.read_csv(
                EXPERIMENT / f"outputs/interventions/{model}_qet_seed{seed}_common_date_fingerprints.csv.gz",
                dtype={"GAGE_ID": str}, usecols=["GAGE_ID", "median_r7_q0_001", "median_half_life_days", "median_release_7d_mm", "median_release_30d_mm"],
            ).set_index("GAGE_ID")
            seed_values.append(table)
        for value in ("median_r7_q0_001", "median_half_life_days", "median_release_7d_mm", "median_release_30d_mm"):
            frame[f"{model}_{value}"] = pd.concat([table[value] for table in seed_values], axis=1).median(axis=1)
        frame[f"{model}_r7"] = frame[f"{model}_median_r7_q0_001"]
    frame["observed_r7"] = primary_observed.observed_median_r7
    frame["observed_usable"] = primary_observed.usable
    frame = frame.reset_index()
    frame.to_csv(output / "gauge_level_reference_only.csv.gz", index=False, compression="gzip")

    pair_rows = []
    for threshold in THRESHOLDS:
        for first, second in itertools.combinations(MODELS, 2):
            mask = (
                (frame[f"{first}_kge"] >= threshold) & (frame[f"{second}_kge"] >= threshold)
                & frame[f"{first}_r7"].notna() & frame[f"{second}_r7"].notna()
            )
            x, y = frame.loc[mask, f"{first}_r7"], frame.loc[mask, f"{second}_r7"]
            observed_mask = mask & frame.observed_usable.fillna(False)
            hydrograph_r = []
            for basin in np.flatnonzero(mask.to_numpy()):
                valid = np.isfinite(ensemble[first][basin]) & np.isfinite(ensemble[second][basin])
                hydrograph_r.append(np.corrcoef(ensemble[first][basin, valid], ensemble[second][basin, valid])[0, 1])
            cross_rho, cross_mad = rho(x, y), mad(x, y)
            pair_rows.append(
                {
                    "skill_threshold": threshold, "first": first, "second": second, "n": int(mask.sum()),
                    "cross_architecture_rho": cross_rho,
                    "median_absolute_r7_difference": cross_mad,
                    "same_retention_class_fraction": float(np.mean(retention_class(x) == retention_class(y))) if mask.any() else np.nan,
                    "median_daily_hydrograph_correlation": float(np.nanmedian(hydrograph_r)) if hydrograph_r else np.nan,
                    "observed_subset_n": int(observed_mask.sum()),
                    "first_observed_rho": rho(frame.loc[observed_mask, f"{first}_r7"], frame.loc[observed_mask, "observed_r7"]),
                    "second_observed_rho": rho(frame.loc[observed_mask, f"{second}_r7"], frame.loc[observed_mask, "observed_r7"]),
                    "first_observed_mae": mad(frame.loc[observed_mask, f"{first}_r7"], frame.loc[observed_mask, "observed_r7"]),
                    "second_observed_mae": mad(frame.loc[observed_mask, f"{second}_r7"], frame.loc[observed_mask, "observed_r7"]),
                    "material_disagreement": bool(mask.sum() >= 10 and (cross_rho < 0.75 or cross_mad > 0.05)) if np.isfinite(cross_rho) else False,
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(output / "pairwise_response_summary.csv", index=False)

    architecture_rows = []
    for model in MODELS:
        observed_mask = frame.observed_usable.fillna(False) & (frame[f"{model}_kge"] >= 0.30)
        architecture_rows.append(
            {"model": model, "observed_subset_n": int(observed_mask.sum()),
             "observed_rho": rho(frame.loc[observed_mask, f"{model}_r7"], frame.loc[observed_mask, "observed_r7"]),
             "observed_mae": mad(frame.loc[observed_mask, f"{model}_r7"], frame.loc[observed_mask, "observed_r7"]),
             "median_r7": float(frame[f"{model}_r7"].median()),
             "median_half_life_days": float(frame[f"{model}_median_half_life_days"].median()),
             "median_release_7d_mm": float(frame[f"{model}_median_release_7d_mm"].median()),
             "median_release_30d_mm": float(frame[f"{model}_median_release_30d_mm"].median())}
        )
    architecture_response = pd.DataFrame(architecture_rows).merge(skill_summary, on="model", validate="one_to_one")
    architecture_response.to_csv(output / "architecture_summary.csv", index=False)

    key_bootstrap = bootstrap_pair(frame, "dhbv", "fluxgate", 0.30)
    high_skill = pairwise.loc[pairwise.skill_threshold.eq(0.60)].copy()
    high_skill_evaluable = high_skill.loc[high_skill.n >= 10]
    converged = high_skill_evaluable.loc[~high_skill_evaluable.material_disagreement]

    tier_rows = []
    for tier, group in frame.groupby("tier_short"):
        for model in MODELS:
            tier_rows.append(
                {"tier_short": tier, "model": model, "n": len(group),
                 "median_kge": float(group[f"{model}_kge"].median()),
                 "fraction_kge_ge_03": float(np.mean(group[f"{model}_kge"] >= 0.30))}
            )
    pd.DataFrame(tier_rows).to_csv(output / "tier_skill_summary.csv", index=False)

    nested_path = ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv"
    nested_rows = []
    if nested_path.exists():
        edges = pd.read_csv(nested_path, dtype=str)
        edges["upstream_gage"] = edges.upstream_gage.astype(str).str.zfill(8)
        edges["downstream_gage"] = edges.downstream_gage.astype(str).str.zfill(8)
        indexed = frame.set_index("GAGE_ID")
        available = set(frame.GAGE_ID)
        pairs = edges.loc[edges.upstream_gage.isin(available) & edges.downstream_gage.isin(available)]
        for model in MODELS:
            use = pairs.loc[
                pairs.upstream_gage.map(indexed[f"{model}_kge"]).ge(0.30)
                & pairs.downstream_gage.map(indexed[f"{model}_kge"]).ge(0.30)
                & pairs.upstream_gage.map(indexed.observed_usable).fillna(False)
                & pairs.downstream_gage.map(indexed.observed_usable).fillna(False)
            ]
            modeled = indexed.loc[use.downstream_gage, f"{model}_r7"].to_numpy() - indexed.loc[use.upstream_gage, f"{model}_r7"].to_numpy()
            observed = indexed.loc[use.downstream_gage, "observed_r7"].to_numpy() - indexed.loc[use.upstream_gage, "observed_r7"].to_numpy()
            nested_rows.append(
                {"model": model, "pairs": len(use), "contrast_vs_observed_rho": rho(modeled, observed), "contrast_vs_observed_mae": mad(modeled, observed)}
            )
    pd.DataFrame(nested_rows).to_csv(output / "nested_reference_pairs.csv", index=False)

    conclusion = {
        "experiment": "reference-only training, development, and held-out evaluation",
        "eligible_reference_gauges": 913, "reference_donors": 400, "reference_development": 100,
        "held_out_reference_targets": 413, "tier_1_targets": int((target.tier_short == "Tier 1").sum()),
        "tier_2_targets": int((target.tier_short == "Tier 2").sum()),
        "nonreference_gauges_used": 0,
        "skill_summary": skill_summary.to_dict(orient="records"),
        "primary_observed_usable_gauges": int(primary_observed.usable.sum()),
        "observed_proxy_primary_vs_scenarios": proxy_reliability.loc[(proxy_reliability['first'] == 'primary') | (proxy_reliability['second'] == 'primary')].to_dict(orient="records"),
        "dhbv_fluxgate_kge03_bootstrap": key_bootstrap,
        "high_skill_pairs_with_n_ge_10": int(len(high_skill_evaluable)),
        "high_skill_pairs_showing_material_disagreement": int(high_skill_evaluable.material_disagreement.sum()),
        "high_skill_pairs_converging": converged[["first", "second", "n", "cross_architecture_rho", "median_absolute_r7_difference"]].to_dict(orient="records"),
        "outcome_firewall": "15 predictions and 15 common-date intervention products checksummed before target discharge was opened",
        "target_opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_discharge_sha256": sha256(q_path),
    }
    (output / "reference_only_discovery.json").write_text(json.dumps(conclusion, indent=2) + "\n")
    print(skill_summary.to_string(index=False), flush=True)
    print(pairwise.loc[pairwise.skill_threshold.isin([0.30, 0.60])].to_string(index=False), flush=True)
    print(json.dumps(conclusion, indent=2), flush=True)


if __name__ == "__main__":
    main()
