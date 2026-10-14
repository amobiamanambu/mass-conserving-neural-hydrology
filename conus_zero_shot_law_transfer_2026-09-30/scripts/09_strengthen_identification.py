#!/usr/bin/env python3
"""Strengthen the two-architecture hydrologic identification test.

This post-evaluation analysis adds four reviewer-facing diagnostics without
changing model weights, target predictions, or the prespecified primary test:

1. agreement among five independently extracted observed-recession proxies;
2. hydrograph equivalence between the two frozen Q+ET model ensembles;
3. a common-date intervention sensitivity using the same antecedent dates in
   all six frozen Q+ET model members; and
4. an auditable flux trace and exact routing decomposition of seven-day
   retention.

The script can be run end to end when the processed forcing, target discharge,
frozen checkpoints, predictions, and original intervention archives are
available. Outputs are written under
``outputs/identification/strengthened_test`` and never overwrite the frozen
primary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import DifferentiableHBV, NeuralFluxGate, make_model  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs/identification/strengthened_test"
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")
RUNS = tuple(f"{architecture}_qet_seed{seed}" for architecture in ARCHITECTURES for seed in SEEDS)
Q0_THRESHOLD = 0.001
COMMON_QUANTILES = np.linspace(0.05, 0.95, 12)
SCENARIO_ORDER = ("primary", "strict_rain", "relaxed_rain", "low_q0", "high_q0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("quick", "extract", "intervene", "all"),
        default="all",
        help="Run quick diagnostics, common-state extraction, interventions, or all stages.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def rho(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def rowwise_hydrograph_metrics(reference: np.ndarray, comparison: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate model-to-model daily agreement at each gauge."""
    valid = np.isfinite(reference) & np.isfinite(comparison)
    count = valid.sum(axis=1)
    ref = np.where(valid, reference, 0.0).astype(np.float64)
    cmp = np.where(valid, comparison, 0.0).astype(np.float64)
    ref_mean = ref.sum(axis=1) / count
    cmp_mean = cmp.sum(axis=1) / count
    ref_centered = np.where(valid, reference - ref_mean[:, None], 0.0)
    cmp_centered = np.where(valid, comparison - cmp_mean[:, None], 0.0)
    covariance = (ref_centered * cmp_centered).sum(axis=1)
    ref_ss = np.square(ref_centered).sum(axis=1)
    cmp_ss = np.square(cmp_centered).sum(axis=1)
    correlation = covariance / np.sqrt(ref_ss * cmp_ss)
    ref_sd = np.sqrt(ref_ss / np.maximum(count - 1, 1))
    cmp_sd = np.sqrt(cmp_ss / np.maximum(count - 1, 1))
    alpha = cmp_sd / ref_sd
    beta = cmp_mean / ref_mean
    kge = 1.0 - np.sqrt(np.square(correlation - 1.0) + np.square(alpha - 1.0) + np.square(beta - 1.0))

    ref_log = np.log1p(np.maximum(reference, 0.0))
    cmp_log = np.log1p(np.maximum(comparison, 0.0))
    ref_log_mean = np.where(valid, ref_log, 0.0).sum(axis=1) / count
    cmp_log_mean = np.where(valid, cmp_log, 0.0).sum(axis=1) / count
    ref_log_centered = np.where(valid, ref_log - ref_log_mean[:, None], 0.0)
    cmp_log_centered = np.where(valid, cmp_log - cmp_log_mean[:, None], 0.0)
    log_correlation = (ref_log_centered * cmp_log_centered).sum(axis=1) / np.sqrt(
        np.square(ref_log_centered).sum(axis=1) * np.square(cmp_log_centered).sum(axis=1)
    )
    return {
        "hydrograph_pearson_r": correlation,
        "hydrograph_log_pearson_r": log_correlation,
        "hydrograph_dhbv_referenced_kge": kge,
        "hydrograph_mean_ratio": beta,
        "hydrograph_sd_ratio": alpha,
        "valid_days": count,
    }


def ensemble_prediction(architecture: str) -> np.ndarray:
    arrays = [
        np.load(PROJECT / f"outputs/predictions/{architecture}_qet_seed{seed}_target_q_2019_2024.npy", mmap_mode="r")
        for seed in SEEDS
    ]
    result = np.empty(arrays[0].shape, dtype=np.float32)
    for first in range(0, arrays[0].shape[0], 256):
        stop = min(first + 256, arrays[0].shape[0])
        result[first:stop] = np.mean([np.asarray(array[first:stop], dtype=np.float64) for array in arrays], axis=0)
    return result


def build_common_candidates(candidate_count: int) -> Path:
    """Select a broad, architecture-independent pool of common dates per gauge."""
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    q_obs = np.load(PROJECT / "data/evaluation/q_target_2019_2024.npy", mmap_mode="r")
    forcing = np.load(PROJECT / "data/processed/forcing.npy", mmap_mode="r")
    months = pd.DatetimeIndex(eval_dates.astype("datetime64[ns]")).month.to_numpy()
    warm_season = (months >= 6) & (months <= 9)
    quantiles = np.linspace(0.005, 0.995, candidate_count)
    index = np.full((len(target), candidate_count), -1, dtype=np.int16)
    value = np.full((len(target), candidate_count), np.nan, dtype=np.float32)
    for basin, station in enumerate(target.itertuples(index=False)):
        temperature = np.asarray(forcing[int(station.row), eval_start:eval_end, 2])
        discharge = np.asarray(q_obs[basin])
        eligible = np.flatnonzero(warm_season & (temperature > 5.0) & np.isfinite(discharge))
        if len(eligible) < len(COMMON_QUANTILES):
            eligible = np.flatnonzero(warm_season & np.isfinite(discharge))
        targets = np.quantile(discharge[eligible], quantiles)
        used: set[int] = set()
        selected: list[int] = []
        for target_value in targets:
            order = eligible[np.argsort(np.abs(discharge[eligible] - target_value), kind="stable")]
            pick = next((int(day) for day in order if int(day) not in used), None)
            if pick is None:
                break
            used.add(pick)
            selected.append(pick)
        index[basin, : len(selected)] = selected
        value[basin, : len(selected)] = discharge[selected]
    path = OUTPUT / "common_date_candidates.npz"
    np.savez_compressed(
        path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        eval_time_index=index,
        observed_q_mm_day=value,
        quantiles=quantiles.astype(np.float32),
    )
    return path


def quick_diagnostics(candidate_count: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    primary = pd.read_parquet(PROJECT / "outputs/identification/primary_identification_population.parquet")
    primary["GAGE_ID"] = primary.GAGE_ID.astype(str).str.zfill(8)

    dhbv = ensemble_prediction("dhbv")
    flux = ensemble_prediction("fluxgate")
    metrics = rowwise_hydrograph_metrics(dhbv, flux)
    hydro = target[["GAGE_ID", "ecoregion", "tier_short", "reference_status"]].copy()
    for name, values in metrics.items():
        hydro[name] = values
    hydro = hydro.merge(
        primary[["GAGE_ID", "dhbv_qet_r7", "fluxgate_qet_r7", "observed_median_r7"]],
        on="GAGE_ID",
        how="left",
    )
    hydro["primary"] = hydro.dhbv_qet_r7.notna()
    hydro["absolute_r7_difference"] = np.abs(hydro.fluxgate_qet_r7 - hydro.dhbv_qet_r7)
    hydro.to_csv(OUTPUT / "hydrograph_equivalence_gauge.csv.gz", index=False, compression="gzip")

    local = hydro.loc[hydro.primary].copy()
    local["agreement_decile"] = pd.qcut(
        local.hydrograph_pearson_r,
        q=10,
        labels=np.arange(1, 11),
        duplicates="drop",
    ).astype(int)
    bins = []
    for decile, group in local.groupby("agreement_decile", observed=True):
        bins.append(
            {
                "agreement_decile": int(decile),
                "n": int(len(group)),
                "pearson_r_min": float(group.hydrograph_pearson_r.min()),
                "pearson_r_median": float(group.hydrograph_pearson_r.median()),
                "pearson_r_max": float(group.hydrograph_pearson_r.max()),
                "r7_spearman_rho": rho(group.dhbv_qet_r7.to_numpy(), group.fluxgate_qet_r7.to_numpy()),
                "median_absolute_r7_difference": float(group.absolute_r7_difference.median()),
                "r7_difference_q25": float(group.absolute_r7_difference.quantile(0.25)),
                "r7_difference_q75": float(group.absolute_r7_difference.quantile(0.75)),
            }
        )
    pd.DataFrame(bins).to_csv(OUTPUT / "hydrograph_equivalence_deciles.csv", index=False)

    recession = pd.read_parquet(PROJECT / "outputs/observed_recessions/recession_summary.parquet")
    recession = recession.loc[recession.usable].copy()
    recession["GAGE_ID"] = recession.GAGE_ID.astype(str).str.zfill(8)
    pivot = recession.pivot(index="GAGE_ID", columns="scenario", values="observed_median_r7")
    pivot = pivot.reindex(columns=SCENARIO_ORDER)
    pairwise = []
    matrix = pd.DataFrame(np.nan, index=SCENARIO_ORDER, columns=SCENARIO_ORDER)
    for first in SCENARIO_ORDER:
        for second in SCENARIO_ORDER:
            common = pivot[[first, second]].dropna()
            value = rho(common[first].to_numpy(), common[second].to_numpy())
            matrix.loc[first, second] = value
            if SCENARIO_ORDER.index(first) < SCENARIO_ORDER.index(second):
                pairwise.append(
                    {
                        "scenario_1": first,
                        "scenario_2": second,
                        "n": int(len(common)),
                        "spearman_rho": value,
                        "median_absolute_difference": float(np.median(np.abs(common[first] - common[second]))),
                    }
                )
    pairwise_frame = pd.DataFrame(pairwise)
    pairwise_frame.to_csv(OUTPUT / "observed_proxy_pairwise_reliability.csv", index=False)
    matrix.to_csv(OUTPUT / "observed_proxy_spearman_matrix.csv", index_label="scenario")
    proxy_spread = pd.DataFrame(
        {
            "GAGE_ID": pivot.index,
            "scenario_count": pivot.notna().sum(axis=1),
            "scenario_range_r7": pivot.max(axis=1) - pivot.min(axis=1),
            "scenario_mad_r7": pivot.sub(pivot.median(axis=1), axis=0).abs().median(axis=1),
        }
    )
    proxy_spread.to_csv(OUTPUT / "observed_proxy_gauge_spread.csv.gz", index=False, compression="gzip")

    top_decile = local.loc[local.agreement_decile.eq(10)]
    high_r = local.loc[local.hydrograph_pearson_r.ge(0.95)]
    reference = local.loc[local.reference_status.eq("Reference")]
    summary = {
        "primary_gauges": int(len(local)),
        "median_model_to_model_daily_pearson_r": float(local.hydrograph_pearson_r.median()),
        "median_model_to_model_daily_log_pearson_r": float(local.hydrograph_log_pearson_r.median()),
        "median_dhbv_referenced_model_to_model_kge": float(local.hydrograph_dhbv_referenced_kge.median()),
        "spearman_hydrograph_r_vs_absolute_r7_difference": rho(
            local.hydrograph_pearson_r.to_numpy(), local.absolute_r7_difference.to_numpy()
        ),
        "top_hydrograph_agreement_decile": {
            "n": int(len(top_decile)),
            "minimum_daily_pearson_r": float(top_decile.hydrograph_pearson_r.min()),
            "r7_spearman_rho": rho(top_decile.dhbv_qet_r7.to_numpy(), top_decile.fluxgate_qet_r7.to_numpy()),
            "median_absolute_r7_difference": float(top_decile.absolute_r7_difference.median()),
        },
        "daily_pearson_r_at_least_0_95": {
            "n": int(len(high_r)),
            "r7_spearman_rho": rho(high_r.dhbv_qet_r7.to_numpy(), high_r.fluxgate_qet_r7.to_numpy()),
            "median_absolute_r7_difference": float(high_r.absolute_r7_difference.median()),
        },
        "reference_primary": {
            "n": int(len(reference)),
            "median_daily_pearson_r": float(reference.hydrograph_pearson_r.median()),
            "median_absolute_r7_difference": float(reference.absolute_r7_difference.median()),
        },
        "observed_proxy_reliability": {
            "pairwise_spearman_min": float(pairwise_frame.spearman_rho.min()),
            "pairwise_spearman_median": float(pairwise_frame.spearman_rho.median()),
            "pairwise_spearman_max": float(pairwise_frame.spearman_rho.max()),
            "median_within_gauge_scenario_range": float(proxy_spread.loc[proxy_spread.scenario_count.ge(2), "scenario_range_r7"].median()),
        },
    }
    write_json(OUTPUT / "quick_diagnostics_summary.json", summary)
    build_common_candidates(candidate_count)


@dataclass
class StepOutput:
    state: torch.Tensor
    q: torch.Tensor
    evap: torch.Tensor
    deep_loss: torch.Tensor
    recharge: torch.Tensor
    fast_to_slow: torch.Tensor
    q_fast: torch.Tensor
    q_slow: torch.Tensor
    route_pre: torch.Tensor
    route_k: torch.Tensor


class FrozenStepper:
    """Exact one-day replay of the two archived constrained architectures."""

    def __init__(self, model: torch.nn.Module, static: torch.Tensor):
        self.model = model
        self.architecture = model.architecture
        if isinstance(model, DifferentiableHBV):
            self.params = model.parameters_from_static(static)
            self.embedding = None
            self.state = static.new_zeros((len(static), 5))
        elif isinstance(model, NeuralFluxGate):
            self.embedding, self.params = model.static_terms(static)
            self.state = model.initial_state(self.params)
        else:
            raise TypeError(type(model))

    def set_state(self, state: torch.Tensor) -> None:
        self.state = state

    def step(self, forcing: torch.Tensor) -> StepOutput:
        snow, soil, fast, slow, route = self.state.unbind(dim=1)
        p = torch.clamp(forcing[:, 0], min=0.0)
        pet = torch.clamp(forcing[:, 1], min=0.0)
        temperature = forcing[:, 2]
        if self.architecture == "dhbv":
            snow_fraction = torch.sigmoid(-2.0 * (temperature - self.params["t0"]))
            snowfall = p * snow_fraction
            rainfall = p - snowfall
            melt = torch.minimum(
                snow + snowfall,
                self.params["degree_day"] * F.relu(temperature - self.params["t0"]),
            )
            snow = snow + snowfall - melt
            liquid = rainfall + melt
            saturation = torch.clamp(soil / self.params["capacity"], 0.0, 1.0)
            recharge = liquid * torch.pow(saturation + 1e-6, self.params["beta"])
            soil = soil + liquid - recharge
            overflow = F.relu(soil - self.params["capacity"])
            soil = torch.minimum(soil, self.params["capacity"])
            evap_fraction = torch.pow(
                torch.clamp(soil / self.params["capacity"], 0.0, 1.0) + 1e-6,
                self.params["evap_shape"],
            )
            evap = torch.minimum(soil, pet * evap_fraction)
            soil = soil - evap
            fast = fast + recharge + overflow
            fast_to_slow = fast * self.params["percolation"]
            fast = fast - fast_to_slow
            q_fast = fast * self.params["k_fast"]
            fast = fast - q_fast
            slow = slow + fast_to_slow
            deep_loss = slow * self.params["deep_fraction"]
            slow = slow - deep_loss
            q_slow = slow * self.params["k_slow"]
            slow = slow - q_slow
            route_pre = route + q_fast + q_slow
            route_k = self.params["routing"]
            q = route_pre * route_k
            route = route_pre - q
        else:
            snow_fraction = torch.sigmoid((self.params["t0"] - temperature) * 2.0)
            snowfall = p * snow_fraction
            rain = p - snowfall
            available_snow = snow + snowfall
            melt = torch.minimum(
                available_snow,
                F.relu(temperature - self.params["t0"]) * self.params["degree_day"],
            )
            snow = available_snow - melt
            liquid = rain + melt
            gate_input = torch.cat(
                [
                    (torch.log1p(p) / 4.0)[:, None],
                    (torch.log1p(pet) / 2.0)[:, None],
                    (temperature / 20.0)[:, None],
                    forcing[:, 3:5],
                    torch.clamp(soil / self.params["capacity"], 0.0, 2.0)[:, None],
                    (torch.log1p(fast) / 5.0)[:, None],
                    (torch.log1p(slow) / 5.0)[:, None],
                    (torch.log1p(route) / 5.0)[:, None],
                    self.embedding,
                ],
                dim=1,
            )
            gate = self.model.gates(gate_input)
            saturation = torch.clamp(soil / self.params["capacity"], 0.0, 1.0)
            direct = liquid * torch.sigmoid(gate[:, 0]) * torch.sqrt(saturation + 1e-6)
            soil = soil + liquid - direct
            evap_saturation = torch.clamp(soil / self.params["capacity"], 0.0, 1.0)
            evap = torch.minimum(
                soil,
                pet * (0.25 + 1.35 * torch.sigmoid(gate[:, 1])) * torch.sqrt(evap_saturation + 1e-6),
            )
            soil = soil - evap
            recharge = soil * (0.00005 + 0.08 * torch.sigmoid(gate[:, 2]))
            soil = soil - recharge
            deep_loss = soil * (0.000001 + 0.002 * torch.sigmoid(gate[:, 3]))
            soil = soil - deep_loss
            overflow = F.relu(soil - self.params["capacity"])
            soil = torch.minimum(soil, self.params["capacity"])
            fast_share = torch.sigmoid(gate[:, 4])
            fast = fast + direct + overflow + recharge * fast_share
            slow = slow + recharge * (1.0 - fast_share)
            q_fast = fast * (0.005 + 0.895 * torch.sigmoid(gate[:, 5]))
            q_slow = slow * (0.0001 + 0.12 * torch.sigmoid(gate[:, 6]))
            fast = fast - q_fast
            slow = slow - q_slow
            route_pre = route + q_fast + q_slow
            route_k = 0.01 + 0.89 * torch.sigmoid(gate[:, 7])
            q = route_pre * route_k
            route = route_pre - q
            fast_to_slow = torch.zeros_like(q)
        self.state = torch.stack([snow, soil, fast, slow, route], dim=1)
        return StepOutput(
            state=self.state,
            q=q,
            evap=evap,
            deep_loss=deep_loss,
            recharge=recharge,
            fast_to_slow=fast_to_slow,
            q_fast=q_fast,
            q_slow=q_slow,
            route_pre=route_pre,
            route_k=route_k,
        )


def load_run(run: str, n_static: int) -> tuple[torch.nn.Module, dict]:
    receipt = json.loads((PROJECT / f"runs/frozen_refit/{run}/receipt.json").read_text())
    checkpoint = torch.load(PROJECT / receipt["checkpoint"], map_location="cpu", weights_only=False)
    model = make_model(receipt["architecture"], n_static)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), receipt


def day_capture_map(indices: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for day in np.unique(indices[indices >= 0]):
        rows, columns = np.nonzero(indices == day)
        result[int(day)] = (rows, columns)
    return result


def extract_candidate_states(run: str, batch_size: int) -> dict:
    candidates = np.load(OUTPUT / "common_date_candidates.npz")
    candidate_index = candidates["eval_time_index"]
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    if not np.array_equal(candidates["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Candidate station order mismatch")
    forcing = np.load(PROJECT / "data/processed/forcing.npy", mmap_mode="r")
    static = np.load(PROJECT / "data/processed/static.npy", mmap_mode="r")
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    model, receipt = load_run(run, static.shape[1])
    archived_anchor = np.load(PROJECT / receipt["antecedent_states"])
    archived_prediction = np.load(PROJECT / receipt["prediction"], mmap_mode="r")
    captured = np.full((len(target), candidate_index.shape[1], 5), np.nan, dtype=np.float32)
    state_max_abs = 0.0
    prediction_max_abs = 0.0

    with torch.no_grad():
        for first in range(0, len(target), batch_size):
            stop = min(first + batch_size, len(target))
            rows = target.row.to_numpy(int)[first:stop]
            x = torch.from_numpy(np.asarray(forcing[rows, :eval_end])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            stepper = FrozenStepper(model, s)
            candidate_map = day_capture_map(candidate_index[first:stop])
            verification_map = day_capture_map(archived_anchor["eval_time_index"][first:stop])
            for day in range(eval_end):
                output = stepper.step(x[:, day])
                eval_day = day - eval_start
                if eval_day in candidate_map:
                    local_rows, columns = candidate_map[eval_day]
                    captured[first + local_rows, columns] = output.state[local_rows].cpu().numpy()
                if eval_day in verification_map:
                    local_rows, columns = verification_map[eval_day]
                    expected = archived_anchor["state"][first + local_rows, columns]
                    actual = output.state[local_rows].cpu().numpy()
                    state_max_abs = max(state_max_abs, float(np.max(np.abs(expected - actual))))
                    expected_q = np.asarray(archived_prediction[first + local_rows, eval_day])
                    actual_q = output.q[local_rows].cpu().numpy()
                    prediction_max_abs = max(prediction_max_abs, float(np.max(np.abs(expected_q - actual_q))))
            print(f"{run}: captured common-date candidates {stop:,}/{len(target):,}", flush=True)

    path = OUTPUT / f"{run}_common_candidate_states.npz"
    np.savez_compressed(
        path,
        gage_ids=target.GAGE_ID.to_numpy(str),
        eval_time_index=candidate_index,
        state=captured,
    )
    result = {
        "run": run,
        "candidate_states": str(path.relative_to(PROJECT)),
        "candidate_states_sha256": sha256(path),
        "maximum_abs_state_replay_error": state_max_abs,
        "maximum_abs_prediction_replay_error_mm_day": prediction_max_abs,
    }
    write_json(OUTPUT / f"{run}_common_candidate_receipt.json", result)
    return result


def extract_all_candidate_states(batch_size: int) -> None:
    receipts = [extract_candidate_states(run, batch_size) for run in RUNS]
    write_json(OUTPUT / "common_candidate_extraction_receipt.json", {"runs": receipts})


def select_common_anchors() -> dict:
    candidates = np.load(OUTPUT / "common_date_candidates.npz")
    candidate_index = candidates["eval_time_index"]
    observed_q = candidates["observed_q_mm_day"]
    states = {run: np.load(OUTPUT / f"{run}_common_candidate_states.npz")["state"] for run in RUNS}
    maximum_snow = np.maximum.reduce([state[..., 0] for state in states.values()])
    common_position = np.full((len(candidate_index), len(COMMON_QUANTILES)), -1, dtype=np.int16)
    relaxation = np.full(len(candidate_index), -1, dtype=np.int8)
    thresholds = (1.0, 2.0, 5.0)
    for basin in range(len(candidate_index)):
        selected = None
        used_threshold = -1
        for code, threshold in enumerate(thresholds):
            valid = np.flatnonzero(
                (candidate_index[basin] >= 0)
                & np.isfinite(observed_q[basin])
                & (maximum_snow[basin] < threshold)
            )
            if len(valid) < len(COMMON_QUANTILES):
                continue
            target_values = np.quantile(observed_q[basin, valid], COMMON_QUANTILES)
            used: set[int] = set()
            picks: list[int] = []
            for target_value in target_values:
                order = valid[np.argsort(np.abs(observed_q[basin, valid] - target_value), kind="stable")]
                pick = next((int(value) for value in order if int(value) not in used), None)
                if pick is None:
                    break
                used.add(pick)
                picks.append(pick)
            if len(picks) == len(COMMON_QUANTILES):
                selected = picks
                used_threshold = code
                break
        if selected is not None:
            common_position[basin] = selected
            relaxation[basin] = used_threshold

    valid_gauge = np.all(common_position >= 0, axis=1)
    common_time = np.full(common_position.shape, -1, dtype=np.int16)
    for basin in np.flatnonzero(valid_gauge):
        common_time[basin] = candidate_index[basin, common_position[basin]]
    path = OUTPUT / "common_anchors.npz"
    np.savez_compressed(
        path,
        gage_ids=candidates["gage_ids"],
        candidate_position=common_position,
        eval_time_index=common_time,
        snow_threshold_relaxation_code=relaxation,
        quantiles=COMMON_QUANTILES.astype(np.float32),
    )
    unique, counts = np.unique(relaxation, return_counts=True)
    summary = {
        "gauges": int(len(common_position)),
        "gauges_with_12_common_anchors": int(valid_gauge.sum()),
        "snow_threshold_relaxation_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "threshold_code_mm": {"0": 1.0, "1": 2.0, "2": 5.0, "-1": None},
        "common_anchors": str(path.relative_to(PROJECT)),
        "common_anchors_sha256": sha256(path),
    }
    write_json(OUTPUT / "common_anchor_selection_summary.json", summary)
    return summary


def zero_input_forcing(eval_dates: np.ndarray, time_index: np.ndarray) -> np.ndarray:
    base_date = eval_dates[time_index].astype("datetime64[D]")
    experiment_date = base_date[:, None] + np.arange(1, 92).astype("timedelta64[D]")[None, :]
    day_of_year = pd.DatetimeIndex(experiment_date.reshape(-1).astype("datetime64[ns]")).dayofyear.to_numpy()
    day_of_year = day_of_year.reshape(len(time_index), 91)
    forcing = np.zeros((len(time_index), 91, 5), dtype=np.float32)
    forcing[:, :, 2] = 10.0
    forcing[:, :, 3] = np.sin(2 * np.pi * day_of_year / 365.25)
    forcing[:, :, 4] = np.cos(2 * np.pi * day_of_year / 365.25)
    return forcing


TRACE_COLUMNS = (
    "evap",
    "deep_loss",
    "soil_to_channel",
    "fast_to_slow",
    "fast_release",
    "slow_release",
    "outlet_q",
)


def trace_intervention_batch(
    model: torch.nn.Module,
    static: torch.Tensor,
    initial_state: torch.Tensor,
    forcing: torch.Tensor,
) -> dict[str, np.ndarray]:
    stepper = FrozenStepper(model, static)
    stepper.set_state(initial_state)
    initial_state_numpy = initial_state.cpu().numpy()
    initial_total = initial_state_numpy.sum(axis=1)
    q0 = q7 = route0 = route7 = k0 = k7 = None
    cumulative = {name: torch.zeros(len(initial_state), dtype=initial_state.dtype) for name in TRACE_COLUMNS}
    with torch.no_grad():
        for day in range(91):
            output = stepper.step(forcing[:, day])
            if day == 0:
                q0 = output.q.clone()
                route0 = output.route_pre.clone()
                k0 = output.route_k.clone()
            if day == 7:
                q7 = output.q.clone()
                route7 = output.route_pre.clone()
                k7 = output.route_k.clone()
            if day <= 7:
                cumulative["evap"] += output.evap
                cumulative["deep_loss"] += output.deep_loss
                cumulative["soil_to_channel"] += output.recharge
                cumulative["fast_to_slow"] += output.fast_to_slow
                cumulative["fast_release"] += output.q_fast
                cumulative["slow_release"] += output.q_slow
                cumulative["outlet_q"] += output.q
    assert q0 is not None and q7 is not None and route0 is not None and route7 is not None and k0 is not None and k7 is not None
    valid = (q0 >= Q0_THRESHOLD) & (q7 > 0) & (route0 > 0) & (route7 > 0) & (k0 > 0) & (k7 > 0)
    r7 = torch.full_like(q0, torch.nan)
    storage_term = torch.full_like(q0, torch.nan)
    gate_term = torch.full_like(q0, torch.nan)
    residual = torch.full_like(q0, torch.nan)
    r7[valid] = q7[valid] / q0[valid]
    storage_term[valid] = torch.log(route7[valid] / route0[valid])
    gate_term[valid] = torch.log(k7[valid] / k0[valid])
    residual[valid] = torch.log(r7[valid]) - storage_term[valid] - gate_term[valid]
    result = {
        "q0_mm_day": q0.cpu().numpy(),
        "r7": r7.cpu().numpy(),
        "routing_storage_log_change": storage_term.cpu().numpy(),
        "routing_gate_log_change": gate_term.cpu().numpy(),
        "decomposition_residual": residual.cpu().numpy(),
        "initial_total_storage_mm": initial_total,
    }
    denominator = np.maximum(initial_total, 1e-6)
    for position, store in enumerate(("snow", "soil", "fast", "slow", "route")):
        result[f"initial_{store}_storage_mm"] = initial_state_numpy[:, position]
        result[f"initial_{store}_fraction_total_storage"] = initial_state_numpy[:, position] / denominator
    for name, value in cumulative.items():
        raw = value.cpu().numpy()
        result[f"cumulative_{name}_mm"] = raw
        result[f"cumulative_{name}_fraction_initial_storage"] = raw / denominator
    return result


def run_interventions_for_states(
    run: str,
    label: str,
    state: np.ndarray,
    time_index: np.ndarray,
    valid_gauge: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    stations = pd.read_csv(PROJECT / "data/processed/station_index.csv", dtype={"GAGE_ID": str})
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    static_all = np.load(PROJECT / "data/processed/static.npy", mmap_mode="r")
    dates = np.load(PROJECT / "data/processed/dates.npy", mmap_mode="r")
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    model, receipt = load_run(run, static_all.shape[1])
    basin_index = np.repeat(np.flatnonzero(valid_gauge), state.shape[1])
    anchor_index = np.tile(np.arange(state.shape[1]), int(valid_gauge.sum()))
    state_flat = state[valid_gauge].reshape(-1, 5)
    time_flat = time_index[valid_gauge].reshape(-1)
    records = []
    for first in range(0, len(state_flat), batch_size):
        stop = min(first + batch_size, len(state_flat))
        basin = basin_index[first:stop]
        rows = target.row.to_numpy(int)[basin]
        forcing = zero_input_forcing(eval_dates, time_flat[first:stop])
        traced = trace_intervention_batch(
            model,
            torch.from_numpy(np.asarray(static_all[rows])).float(),
            torch.from_numpy(state_flat[first:stop]).float(),
            torch.from_numpy(forcing).float(),
        )
        frame = pd.DataFrame(traced)
        frame.insert(0, "antecedent_time_index", time_flat[first:stop])
        frame.insert(0, "anchor", anchor_index[first:stop])
        frame.insert(0, "GAGE_ID", target.GAGE_ID.to_numpy(str)[basin])
        frame.insert(0, "seed", int(receipt["seed"]))
        frame.insert(0, "architecture", receipt["architecture"])
        frame.insert(0, "anchor_design", label)
        records.append(frame)
        if stop == len(state_flat) or stop % (20 * batch_size) == 0:
            print(f"{run}: {label} interventions {stop:,}/{len(state_flat):,}", flush=True)
    return pd.concat(records, ignore_index=True)


def aggregate_trace(trace: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in trace.columns if column not in {"anchor_design", "architecture", "GAGE_ID", "seed", "anchor", "antecedent_time_index"}]
    seed = trace.groupby(["anchor_design", "architecture", "seed", "GAGE_ID"], as_index=False)[numeric].median()
    return seed.groupby(["anchor_design", "architecture", "GAGE_ID"], as_index=False)[numeric].median()


def summarize_common_and_mechanism(original: pd.DataFrame, common: pd.DataFrame) -> None:
    primary = pd.read_parquet(PROJECT / "outputs/identification/primary_identification_population.parquet")
    primary["GAGE_ID"] = primary.GAGE_ID.astype(str).str.zfill(8)
    aggregated = pd.concat([aggregate_trace(original), aggregate_trace(common)], ignore_index=True)
    wide = aggregated.pivot(index=["anchor_design", "GAGE_ID"], columns="architecture", values="r7").reset_index()
    wide = wide.merge(
        primary[["GAGE_ID", "ecoregion", "reference_status", "observed_median_r7"]],
        on="GAGE_ID",
        how="inner",
    )
    wide["absolute_r7_difference"] = np.abs(wide.fluxgate - wide.dhbv)
    wide.to_csv(OUTPUT / "anchor_design_gauge_comparison.csv", index=False)

    summaries = []
    for design, group in wide.groupby("anchor_design"):
        for population, local in (
            ("all_primary", group),
            ("reference_primary", group.loc[group.reference_status.eq("Reference")]),
        ):
            summaries.append(
                {
                    "anchor_design": design,
                    "population": population,
                    "n": int(len(local)),
                    "cross_architecture_rho": rho(local.dhbv.to_numpy(), local.fluxgate.to_numpy()),
                    "median_absolute_r7_difference": float(local.absolute_r7_difference.median()),
                    "dhbv_observed_rho": rho(local.dhbv.to_numpy(), local.observed_median_r7.to_numpy()),
                    "fluxgate_observed_rho": rho(local.fluxgate.to_numpy(), local.observed_median_r7.to_numpy()),
                    "dhbv_observed_mae": float(np.median(np.abs(local.dhbv - local.observed_median_r7))),
                    "fluxgate_observed_mae": float(np.median(np.abs(local.fluxgate - local.observed_median_r7))),
                }
            )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUTPUT / "anchor_design_summary.csv", index=False)

    mechanism = aggregated.merge(primary[["GAGE_ID", "reference_status"]], on="GAGE_ID", how="inner")
    mechanism_rows = []
    mechanism_columns = [
        "routing_storage_log_change",
        "routing_gate_log_change",
        "decomposition_residual",
        *[f"initial_{store}_fraction_total_storage" for store in ("snow", "soil", "fast", "slow", "route")],
        *[f"cumulative_{name}_fraction_initial_storage" for name in TRACE_COLUMNS],
    ]
    for (design, architecture), group in mechanism.groupby(["anchor_design", "architecture"]):
        for metric in mechanism_columns:
            values = group[metric].dropna()
            mechanism_rows.append(
                {
                    "anchor_design": design,
                    "architecture": architecture,
                    "metric": metric,
                    "n": int(len(values)),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    mechanism_frame = pd.DataFrame(mechanism_rows)
    mechanism_frame.to_csv(OUTPUT / "mechanism_summary.csv", index=False)

    pairing_keys = ["GAGE_ID", "seed", "anchor", "antecedent_time_index"]
    trace_columns = ["r7", "routing_storage_log_change", "routing_gate_log_change"]
    dhbv_common = common.loc[common.architecture.eq("dhbv"), pairing_keys + trace_columns]
    flux_common = common.loc[common.architecture.eq("fluxgate"), pairing_keys + trace_columns]
    gap = dhbv_common.merge(flux_common, on=pairing_keys, suffixes=("_dhbv", "_fluxgate"), how="inner").dropna()
    gap["log_r7_gap_fluxgate_minus_dhbv"] = np.log(gap.r7_fluxgate) - np.log(gap.r7_dhbv)
    gap["routing_storage_gap"] = (
        gap.routing_storage_log_change_fluxgate - gap.routing_storage_log_change_dhbv
    )
    gap["routing_gate_gap"] = gap.routing_gate_log_change_fluxgate - gap.routing_gate_log_change_dhbv
    gap["exact_decomposition_residual"] = (
        gap.log_r7_gap_fluxgate_minus_dhbv - gap.routing_storage_gap - gap.routing_gate_gap
    )
    gap_path = OUTPUT / "cross_architecture_gap_decomposition.parquet"
    gap.to_parquet(gap_path, index=False, compression="zstd")
    gap_summary = {}
    for metric in (
        "log_r7_gap_fluxgate_minus_dhbv",
        "routing_storage_gap",
        "routing_gate_gap",
        "exact_decomposition_residual",
    ):
        gap_summary[metric] = {
            "q25": float(gap[metric].quantile(0.25)),
            "median": float(gap[metric].median()),
            "q75": float(gap[metric].quantile(0.75)),
        }
    gap_summary["n_paired_trajectories"] = int(len(gap))
    gap_summary["storage_term_same_sign_as_total_fraction"] = float(
        np.mean(np.sign(gap.log_r7_gap_fluxgate_minus_dhbv) == np.sign(gap.routing_storage_gap))
    )
    gap_summary["maximum_absolute_exact_residual"] = float(gap.exact_decomposition_residual.abs().max())

    summary = {
        "anchor_designs": summaries,
        "common_date_cross_architecture_gap_decomposition": gap_summary,
        "maximum_absolute_exact_decomposition_residual": float(
            pd.concat([original.decomposition_residual.abs(), common.decomposition_residual.abs()]).max()
        ),
        "interpretation": (
            "The routing identity is exact for every valid trajectory. The storage term measures change in "
            "pre-release routing water; the gate term measures change in the routing release coefficient. "
            "Cumulative fluxes diagnose how recharge, fast release, slow release, deep loss, and routing "
            "produce the storage term but are not treated as independent causal effects."
        ),
    }
    write_json(OUTPUT / "strengthened_identification_summary.json", summary)


def run_original_and_common_interventions(batch_size: int) -> None:
    select_common_anchors()
    common = np.load(OUTPUT / "common_anchors.npz")
    valid_gauge = np.all(common["candidate_position"] >= 0, axis=1)
    common_records = []
    original_records = []
    for run in RUNS:
        candidate_state = np.load(OUTPUT / f"{run}_common_candidate_states.npz")["state"]
        common_state = np.full((len(valid_gauge), len(COMMON_QUANTILES), 5), np.nan, dtype=np.float32)
        valid_positions = common["candidate_position"][valid_gauge]
        common_state[valid_gauge] = np.take_along_axis(
            candidate_state[valid_gauge],
            valid_positions[..., None],
            axis=1,
        )
        common_records.append(
            run_interventions_for_states(
                run,
                "common_date",
                common_state,
                common["eval_time_index"],
                valid_gauge,
                batch_size,
            )
        )

        original_anchor = np.load(PROJECT / f"outputs/antecedent_states/{run}_antecedent_states.npz")
        all_valid = np.ones(len(valid_gauge), dtype=bool)
        original_trace = run_interventions_for_states(
            run,
            "model_specific_storage_quantile",
            original_anchor["state"],
            original_anchor["eval_time_index"],
            all_valid,
            batch_size,
        )
        archived = pd.read_parquet(PROJECT / f"outputs/law_interventions/{run}_fingerprints.parquet")
        replay = original_trace.groupby("GAGE_ID", as_index=False).r7.median()
        check = archived[["GAGE_ID", "median_r7"]].merge(replay, on="GAGE_ID", how="inner")
        max_error = float(np.nanmax(np.abs(check.median_r7 - check.r7)))
        if max_error > 2e-5:
            raise RuntimeError(f"{run} intervention replay differs from archive by {max_error}")
        original_trace["archived_fingerprint_max_abs_error"] = max_error
        original_records.append(original_trace)

    original = pd.concat(original_records, ignore_index=True)
    common_frame = pd.concat(common_records, ignore_index=True)
    original_path = OUTPUT / "original_anchor_flux_trace.parquet"
    common_path = OUTPUT / "common_date_flux_trace.parquet"
    original.to_parquet(original_path, index=False, compression="zstd")
    common_frame.to_parquet(common_path, index=False, compression="zstd")
    summarize_common_and_mechanism(original, common_frame)
    write_json(
        OUTPUT / "intervention_trace_receipt.json",
        {
            "original_trace": str(original_path.relative_to(PROJECT)),
            "original_trace_sha256": sha256(original_path),
            "common_trace": str(common_path.relative_to(PROJECT)),
            "common_trace_sha256": sha256(common_path),
            "target_discharge_used_for_common_date_selection": True,
            "status": "post-evaluation sensitivity; primary frozen analysis unchanged",
        },
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    np.random.seed(20261003)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"quick", "all"}:
        quick_diagnostics(args.candidate_count)
    if args.stage in {"extract", "all"}:
        if not (OUTPUT / "common_date_candidates.npz").exists():
            build_common_candidates(args.candidate_count)
        extract_all_candidate_states(args.batch_size)
    if args.stage in {"intervene", "all"}:
        run_original_and_common_interventions(args.batch_size)


if __name__ == "__main__":
    main()
