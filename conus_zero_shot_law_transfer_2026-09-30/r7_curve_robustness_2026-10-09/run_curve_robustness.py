#!/usr/bin/env python3
"""Post-evaluation robustness tests for the R7 drainage-response result.

This script does not train or refit a model. It uses the frozen five-model
predictions, archived 90-day zero-input curves, and previously extracted USGS
recession events to answer three questions:

1. Does cross-architecture disagreement persist beyond the day-7 endpoint?
2. Do complete model response curves agree with complete observed recessions?
3. Can six representative, skill-qualified Tier-1 gauges illustrate the
   national result without selecting gauges by model error?

All results are descriptive post-evaluation robustness checks. R7 remains the
prespecified primary fingerprint in the source experiment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MULTI = PROJECT / "multimodel_neural_conservation_stress_test_2026-10-03"
OUTPUT = HERE / "outputs"
FIGURES = HERE / "figures"
PROVENANCE = HERE / "provenance"
for directory in (OUTPUT, FIGURES, PROVENANCE):
    directory.mkdir(parents=True, exist_ok=True)

# Model order and labels are defined locally so the analysis is self-contained.
PRIMARY = ["dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr"]
LABEL = {
    "dhbv": "dHBV",
    "fluxgate": "Flux gate",
    "mcp_ma3": "MCP-MA3",
    "mc_node": "Neural ODE",
    "mclstm_dr": "MC-LSTM-DR",
}


# Restrained journal styling for optional diagnostic graphics.
INK = "#000000"
TEXT = "#1A1A1A"
MUTED = "#555555"
GRID = "#E0E0E0"
BLUE = "#1F4E79"
ORANGE = "#D0803A"
GRAY = "#888888"
LIGHT = "#D2D2D2"
RED = "#9E3D32"

MODEL_STYLE = {
    "dhbv": dict(color=BLUE, ls="-"),
    "fluxgate": dict(color=ORANGE, ls="-"),
    "mcp_ma3": dict(color=INK, ls="--"),
    "mc_node": dict(color="#6E6E6E", ls="-."),
    "mclstm_dr": dict(color=RED, ls=(0, (1, 1.2))),
}

MM = 1 / 25.4
FIG_WIDTH_MM = 180


def set_che_style() -> None:
    """Apply a restrained journal plotting style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": TEXT,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, letter: str, title: str, y: float = 1.025) -> None:
    label = ax.text(
        0.0,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
    )
    ax.annotate(
        f" {title}",
        xy=(1, 0),
        xycoords=label,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK,
    )


def che_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.4, ls="-")


def reference_line(ax: plt.Axes, axis: str, value: float, color: str = MUTED) -> None:
    options = dict(color=color, lw=0.65, ls=(0, (1.2, 1.6)), zorder=1)
    (ax.axhline if axis == "y" else ax.axvline)(value, **options)


SEEDS = (11, 29, 47)
Q0_THRESHOLD = 0.001
KGE_THRESHOLD = 0.30
BOOTSTRAPS = 2000
RNG_SEED = 20261009
HORIZONS = (3, 7, 14, 30)
SKILL_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)

PAIR_LABEL = {
    tuple(sorted(("dhbv", "fluxgate"))): "dHBV – Flux gate",
    tuple(sorted(("dhbv", "mcp_ma3"))): "dHBV – MCP-MA3",
    tuple(sorted(("dhbv", "mc_node"))): "dHBV – Neural ODE",
    tuple(sorted(("dhbv", "mclstm_dr"))): "dHBV – MC-LSTM-DR",
    tuple(sorted(("fluxgate", "mcp_ma3"))): "Flux gate – MCP-MA3",
    tuple(sorted(("fluxgate", "mc_node"))): "Flux gate – Neural ODE",
    tuple(sorted(("fluxgate", "mclstm_dr"))): "Flux gate – MC-LSTM-DR",
    tuple(sorted(("mcp_ma3", "mc_node"))): "MCP-MA3 – Neural ODE",
    tuple(sorted(("mcp_ma3", "mclstm_dr"))): "MCP-MA3 – MC-LSTM-DR",
    tuple(sorted(("mc_node", "mclstm_dr"))): "Neural ODE – MC-LSTM-DR",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def curve_path(model: str, seed: int) -> Path:
    if model in {"dhbv", "fluxgate"}:
        return PROJECT / "outputs" / "law_interventions" / f"{model}_qet_seed{seed}_curves.npz"
    return (
        MULTI
        / "outputs"
        / "interventions"
        / f"{model}_qet_seed{seed}_model_specific_curves.npz"
    )


def prediction_path(model: str, seed: int) -> Path:
    if model in {"dhbv", "fluxgate"}:
        return PROJECT / "outputs" / "predictions" / f"{model}_qet_seed{seed}_target_q_2019_2024.npy"
    return MULTI / "outputs" / "predictions" / f"{model}_qet_seed{seed}_target_q_2019_2024.npy"


def load_seed_median_curve(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    archive = np.load(path, mmap_mode="r")
    ids = np.asarray(archive["gage_ids"]).astype(str)
    if not np.array_equal(ids, expected_ids):
        raise RuntimeError(f"Gauge order mismatch in {path}")
    q = archive["q_mm_day"]
    output = np.full((len(ids), q.shape[2]), np.nan, dtype=np.float32)
    for lo in range(0, len(ids), 192):
        hi = min(lo + 192, len(ids))
        block = np.asarray(q[lo:hi], dtype=np.float32)
        q0 = block[:, :, 0]
        valid = np.isfinite(q0) & (q0 >= Q0_THRESHOLD)
        ratio = np.full(block.shape, np.nan, dtype=np.float32)
        np.divide(block, q0[:, :, None], out=ratio, where=valid[:, :, None])
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            output[lo:hi] = np.nanmedian(ratio, axis=1)
    return output


def build_zero_input_curves(gage_ids: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    curves: dict[str, np.ndarray] = {}
    input_hashes: dict[str, str] = {}
    for model in PRIMARY:
        seed_curves = []
        for seed in SEEDS:
            path = curve_path(model, seed)
            input_hashes[str(path.relative_to(PROJECT))] = sha256(path)
            seed_curves.append(load_seed_median_curve(path, gage_ids))
        with np.errstate(all="ignore"):
            curves[model] = np.nanmedian(np.stack(seed_curves), axis=0).astype(np.float32)
        print(f"zero-input curves: {LABEL[model]}", flush=True)
    np.savez_compressed(
        OUTPUT / "zero_input_seed_median_curves.npz",
        gage_ids=gage_ids,
        days=np.arange(91, dtype=np.int16),
        **curves,
    )
    return curves, input_hashes


def ensemble_prediction(model: str, input_hashes: dict[str, str]) -> np.ndarray:
    total = None
    for seed in SEEDS:
        path = prediction_path(model, seed)
        input_hashes[str(path.relative_to(PROJECT))] = sha256(path)
        values = np.load(path, mmap_mode="r")
        if total is None:
            total = np.asarray(values, dtype=np.float64)
        else:
            total += np.asarray(values, dtype=np.float64)
    return (total / len(SEEDS)).astype(np.float32)


def observed_event_curves(
    events: pd.DataFrame,
    target_ids: np.ndarray,
    observed_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(target_ids)
    median = np.full((n, 8), np.nan, dtype=np.float32)
    q25 = np.full((n, 8), np.nan, dtype=np.float32)
    q75 = np.full((n, 8), np.nan, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int16)
    grouped = {key: frame.start_index.to_numpy(np.int64) for key, frame in events.groupby("GAGE_ID")}
    offsets = np.arange(8, dtype=np.int64)
    for i, gage in enumerate(target_ids):
        starts = grouped.get(gage)
        if starts is None or len(starts) == 0:
            continue
        block = np.asarray(observed_q[i, starts[:, None] + offsets[None, :]], dtype=np.float64)
        valid = np.isfinite(block).all(axis=1) & (block[:, 0] >= Q0_THRESHOLD)
        block = block[valid]
        if len(block) == 0:
            continue
        ratio = block / block[:, [0]]
        median[i] = np.median(ratio, axis=0)
        q25[i] = np.quantile(ratio, 0.25, axis=0)
        q75[i] = np.quantile(ratio, 0.75, axis=0)
        counts[i] = len(block)
    return median, q25, q75, counts


def predicted_event_curves(
    events: pd.DataFrame,
    target_ids: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(target_ids)
    median = np.full((n, 8), np.nan, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int16)
    grouped = {key: frame.start_index.to_numpy(np.int64) for key, frame in events.groupby("GAGE_ID")}
    offsets = np.arange(8, dtype=np.int64)
    for i, gage in enumerate(target_ids):
        starts = grouped.get(gage)
        if starts is None or len(starts) == 0:
            continue
        block = np.asarray(prediction[i, starts[:, None] + offsets[None, :]], dtype=np.float64)
        valid = np.isfinite(block).all(axis=1) & (block[:, 0] >= Q0_THRESHOLD)
        block = block[valid]
        if len(block) == 0:
            continue
        ratio = block / block[:, [0]]
        median[i] = np.median(ratio, axis=0)
        counts[i] = len(block)
    return median, counts


def build_natural_event_curves(
    events: pd.DataFrame,
    target_ids: np.ndarray,
    input_hashes: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    observed_path = PROJECT / "data" / "evaluation" / "q_target_2019_2024.npy"
    input_hashes[str(observed_path.relative_to(PROJECT))] = sha256(observed_path)
    observed_q = np.load(observed_path, mmap_mode="r")
    obs_med, obs_q25, obs_q75, obs_count = observed_event_curves(events, target_ids, observed_q)

    natural: dict[str, np.ndarray] = {}
    natural_count: dict[str, np.ndarray] = {}
    for model in PRIMARY:
        prediction = ensemble_prediction(model, input_hashes)
        natural[model], natural_count[model] = predicted_event_curves(events, target_ids, prediction)
        print(f"natural recession curves: {LABEL[model]}", flush=True)
        del prediction

    np.savez_compressed(
        OUTPUT / "natural_event_median_curves.npz",
        gage_ids=target_ids,
        days=np.arange(8, dtype=np.int16),
        observed=obs_med,
        observed_q25=obs_q25,
        observed_q75=obs_q75,
        observed_events=obs_count,
        **{f"{model}_natural": curve for model, curve in natural.items()},
        **{f"{model}_events": count for model, count in natural_count.items()},
    )
    return obs_med, obs_q25, obs_q75, obs_count, natural, natural_count


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def stratified_median_interval(values: np.ndarray, regions: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(values) & pd.notna(regions)
    values = np.asarray(values[valid], dtype=float)
    regions = np.asarray(regions[valid], dtype=object)
    if len(values) == 0:
        return np.nan, np.nan
    groups = [np.flatnonzero(regions == region) for region in np.unique(regions)]
    rng = np.random.default_rng(RNG_SEED)
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        estimates[b] = np.median(values[sampled])
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def normalized_auc(curve: np.ndarray, horizon: int) -> np.ndarray:
    return np.trapezoid(curve[:, : horizon + 1], x=np.arange(horizon + 1), axis=1) / horizon


def crossing_day(curves: np.ndarray) -> np.ndarray:
    result = np.full(len(curves), 91.0, dtype=float)
    for i, curve in enumerate(curves):
        if not np.isfinite(curve).all():
            result[i] = np.nan
            continue
        crossing = np.flatnonzero(curve <= 0.5)
        if len(crossing):
            result[i] = float(crossing[0])
    return result


def pairwise_curve_analysis(
    gauges: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    time_rows = []
    for first, second in itertools.combinations(PRIMARY, 2):
        mask = (
            gauges["observed_usable"].to_numpy(bool)
            & (gauges[f"{first}_kge"].to_numpy(float) >= KGE_THRESHOLD)
            & (gauges[f"{second}_kge"].to_numpy(float) >= KGE_THRESHOLD)
            & np.isfinite(curves[first][:, :31]).all(axis=1)
            & np.isfinite(curves[second][:, :31]).all(axis=1)
        )
        a = curves[first][mask]
        b = curves[second][mask]
        regions = gauges.loc[mask, "ecoregion"].to_numpy(object)
        gap = np.abs(a[:, 1:31] - b[:, 1:31])
        d30 = np.mean(gap, axis=1)
        ci_low, ci_high = stratified_median_interval(d30, regions)
        auc_a, auc_b = normalized_auc(a, 30), normalized_auc(b, 30)
        half_a, half_b = crossing_day(a), crossing_day(b)
        key = tuple(sorted((first, second)))
        row = {
            "first": first,
            "second": second,
            "pair": PAIR_LABEL[key],
            "gauges": int(mask.sum()),
            "median_curve_distance_d1_d30": float(np.median(d30)),
            "curve_distance_ci_low": ci_low,
            "curve_distance_ci_high": ci_high,
            "auc30_spearman_rho": rank_correlation(auc_a, auc_b),
            "median_absolute_auc30_difference": float(np.median(np.abs(auc_a - auc_b))),
            "half_life_spearman_rho": rank_correlation(half_a, half_b),
            "median_absolute_half_life_difference_days": float(np.nanmedian(np.abs(half_a - half_b))),
        }
        for horizon in HORIZONS:
            row[f"r{horizon}_spearman_rho"] = rank_correlation(a[:, horizon], b[:, horizon])
            row[f"median_absolute_r{horizon}_difference"] = float(
                np.median(np.abs(a[:, horizon] - b[:, horizon]))
            )
        summary_rows.append(row)
        for day in range(1, 31):
            time_rows.append(
                {
                    "first": first,
                    "second": second,
                    "pair": PAIR_LABEL[key],
                    "gauges": int(mask.sum()),
                    "day": day,
                    "median_absolute_retention_difference": float(np.median(gap[:, day - 1])),
                    "spearman_rho": rank_correlation(a[:, day], b[:, day]),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values("median_curve_distance_d1_d30").reset_index(drop=True)
    time = pd.DataFrame(time_rows)
    summary.to_csv(OUTPUT / "pairwise_full_curve_summary.csv", index=False)
    time.to_csv(OUTPUT / "pairwise_daily_curve_agreement.csv", index=False)
    return summary, time


def pairwise_skill_grid(gauges: pd.DataFrame, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for first, second in itertools.combinations(PRIMARY, 2):
        key = tuple(sorted((first, second)))
        for threshold in SKILL_THRESHOLDS:
            mask = (
                gauges["observed_usable"].to_numpy(bool)
                & (gauges[f"{first}_kge"].to_numpy(float) >= threshold)
                & (gauges[f"{second}_kge"].to_numpy(float) >= threshold)
                & np.isfinite(curves[first][:, :31]).all(axis=1)
                & np.isfinite(curves[second][:, :31]).all(axis=1)
            )
            a = curves[first][mask]
            b = curves[second][mask]
            row = {
                "first": first,
                "second": second,
                "pair": PAIR_LABEL[key],
                "kge_threshold": threshold,
                "gauges": int(mask.sum()),
                "median_curve_distance_d1_d30": float(np.median(np.mean(np.abs(a[:, 1:31] - b[:, 1:31]), axis=1)))
                if len(a)
                else np.nan,
            }
            for horizon in HORIZONS:
                row[f"median_absolute_r{horizon}_difference"] = (
                    float(np.median(np.abs(a[:, horizon] - b[:, horizon]))) if len(a) else np.nan
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "pairwise_full_curve_skill_grid.csv", index=False)
    return result


def decay_rates(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    early = np.full(len(curve), np.nan, dtype=float)
    late = np.full(len(curve), np.nan, dtype=float)
    valid_early = np.isfinite(curve[:, 3]) & (curve[:, 3] > 0)
    early[valid_early] = -np.log(curve[valid_early, 3]) / 3.0
    valid_late = valid_early & np.isfinite(curve[:, 7]) & (curve[:, 7] > 0)
    late[valid_late] = -np.log(curve[valid_late, 7] / curve[valid_late, 3]) / 4.0
    return early, late


def observational_curve_analysis(
    gauges: pd.DataFrame,
    zero_curves: dict[str, np.ndarray],
    observed: np.ndarray,
    observed_counts: np.ndarray,
    natural: dict[str, np.ndarray],
    natural_counts: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_rows = []
    summary_rows = []
    obs_auc = normalized_auc(observed, 7)
    obs_early, obs_late = decay_rates(observed)

    for model in PRIMARY:
        for source, source_curve, source_counts in (
            ("zero-input response", zero_curves[model][:, :8], np.full(len(gauges), 12)),
            ("natural-event prediction", natural[model], natural_counts[model]),
        ):
            mask = (
                gauges["observed_usable"].to_numpy(bool)
                & (gauges[f"{model}_kge"].to_numpy(float) >= KGE_THRESHOLD)
                & (observed_counts >= 3)
                & (source_counts >= 3)
                & np.isfinite(observed).all(axis=1)
                & np.isfinite(source_curve).all(axis=1)
            )
            obs = observed[mask]
            mod = source_curve[mask]
            regions = gauges.loc[mask, "ecoregion"].to_numpy(object)
            for day in range(1, 8):
                day_rows.append(
                    {
                        "model": model,
                        "source": source,
                        "day": day,
                        "gauges": int(mask.sum()),
                        "spearman_rho": rank_correlation(mod[:, day], obs[:, day]),
                        "median_absolute_difference": float(np.median(np.abs(mod[:, day] - obs[:, day]))),
                    }
                )

            curve_mae = np.mean(np.abs(mod[:, 1:8] - obs[:, 1:8]), axis=1)
            ci_low, ci_high = stratified_median_interval(curve_mae, regions)
            mod_auc = normalized_auc(mod, 7)
            mod_early, mod_late = decay_rates(mod)
            local_obs_auc = obs_auc[mask]
            local_obs_early = obs_early[mask]
            local_obs_late = obs_late[mask]
            summary_rows.append(
                {
                    "model": model,
                    "source": source,
                    "gauges": int(mask.sum()),
                    "median_curve_mae_days1_7": float(np.median(curve_mae)),
                    "curve_mae_ci_low": ci_low,
                    "curve_mae_ci_high": ci_high,
                    "auc7_spearman_rho": rank_correlation(mod_auc, local_obs_auc),
                    "median_absolute_auc7_difference": float(np.median(np.abs(mod_auc - local_obs_auc))),
                    "early_decay_spearman_rho": rank_correlation(mod_early, local_obs_early),
                    "median_absolute_early_decay_difference_per_day": float(
                        np.nanmedian(np.abs(mod_early - local_obs_early))
                    ),
                    "late_decay_spearman_rho": rank_correlation(mod_late, local_obs_late),
                    "median_absolute_late_decay_difference_per_day": float(
                        np.nanmedian(np.abs(mod_late - local_obs_late))
                    ),
                }
            )
    daily = pd.DataFrame(day_rows)
    summary = pd.DataFrame(summary_rows)
    daily.to_csv(OUTPUT / "observed_curve_daily_agreement.csv", index=False)
    summary.to_csv(OUTPUT / "observed_curve_summary.csv", index=False)
    return daily, summary


def validate_r7(gauges: pd.DataFrame, curves: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model in PRIMARY:
        archived = gauges[f"{model}_r7"].to_numpy(float)
        rebuilt = curves[model][:, 7]
        valid = np.isfinite(archived) & np.isfinite(rebuilt)
        difference = np.abs(archived[valid] - rebuilt[valid])
        rows.append(
            {
                "model": model,
                "gauges_compared": int(valid.sum()),
                "maximum_absolute_r7_reconciliation_error": float(np.max(difference)),
                "median_absolute_r7_reconciliation_error": float(np.median(difference)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "r7_reconciliation.csv", index=False)
    if result.maximum_absolute_r7_reconciliation_error.max() > 2e-5:
        raise RuntimeError("Rebuilt day-7 curves do not reconcile with archived fingerprints")
    return result


def validate_observed_r7(gauges: pd.DataFrame, observed: np.ndarray) -> pd.DataFrame:
    archived = gauges["observed_median_r7"].to_numpy(float)
    rebuilt = observed[:, 7]
    valid = np.isfinite(archived) & np.isfinite(rebuilt)
    difference = np.abs(archived[valid] - rebuilt[valid])
    result = pd.DataFrame(
        [
            {
                "gauges_compared": int(valid.sum()),
                "maximum_absolute_r7_reconciliation_error": float(np.max(difference)),
                "median_absolute_r7_reconciliation_error": float(np.median(difference)),
            }
        ]
    )
    result.to_csv(OUTPUT / "observed_r7_reconciliation.csv", index=False)
    if result.maximum_absolute_r7_reconciliation_error.iloc[0] > 2e-5:
        raise RuntimeError("Rebuilt observed day-7 curves do not reconcile with archived observed R7")
    return result


def select_six_gauges(
    gauges: pd.DataFrame,
    zero_curves: dict[str, np.ndarray],
    observed: np.ndarray,
    observed_counts: np.ndarray,
) -> pd.DataFrame:
    all_skill = np.column_stack(
        [gauges[f"{model}_kge"].to_numpy(float) >= KGE_THRESHOLD for model in PRIMARY]
    ).all(axis=1)
    all_curves = np.column_stack(
        [np.isfinite(zero_curves[model][:, :8]).all(axis=1) for model in PRIMARY]
    ).all(axis=1)
    eligible = (
        gauges["observed_usable"].to_numpy(bool)
        & gauges["tier_short"].eq("Tier 1").to_numpy(bool)
        & all_skill
        & all_curves
        & np.isfinite(observed).all(axis=1)
        & (observed_counts >= 5)
    )
    candidates = gauges.loc[eligible].copy()
    candidates["row_index"] = np.flatnonzero(eligible)
    centers = (np.arange(6) + 0.5) / 6.0
    targets = np.quantile(candidates["observed_median_r7"], centers)

    # Choose one ecoregion per sextile without using model error. The global
    # assignment minimizes total distance from the six observed-R7 centers.
    regions = sorted(candidates.ecoregion.unique())
    cost = np.full((6, len(regions)), 1e6, dtype=float)
    nearest: dict[tuple[int, int], int] = {}
    scale = max(float(candidates.observed_median_r7.quantile(0.75) - candidates.observed_median_r7.quantile(0.25)), 1e-6)
    for i, target in enumerate(targets):
        for j, region in enumerate(regions):
            local = candidates.loc[candidates.ecoregion.eq(region)]
            if local.empty:
                continue
            index = (local["observed_median_r7"] - target).abs().idxmin()
            nearest[(i, j)] = index
            cost[i, j] = abs(float(local.loc[index, "observed_median_r7"]) - float(target)) / scale
    row_assignment, region_assignment = linear_sum_assignment(cost)
    if len(row_assignment) != 6 or np.any(cost[row_assignment, region_assignment] >= 1e5):
        raise RuntimeError("Could not select six geographically distinct R7 representatives")

    selected_rows = []
    for i, j in sorted(zip(row_assignment, region_assignment)):
        number = i + 1
        center = centers[i]
        target = targets[i]
        row = candidates.loc[nearest[(i, j)]].copy()
        row["selection_sextile"] = number
        row["selection_quantile_center"] = center
        row["selection_target_observed_r7"] = target
        row["selection_rule"] = "global minimum-distance assignment with unique ecoregions"
        row["observed_curve_events"] = int(observed_counts[int(row.row_index)])
        row["minimum_five_model_kge"] = float(min(row[f"{model}_kge"] for model in PRIMARY))
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    metadata = pd.read_csv(PROJECT / "data" / "frozen" / "station_split.csv", dtype={"GAGE_ID": str})
    metadata["GAGE_ID"] = metadata.GAGE_ID.str.zfill(8)
    selected = selected.merge(
        metadata[["GAGE_ID", "station_name", "state", "FORESTNLCD06", "SLOPE_PCT", "ELEV_MEAN_M_BASIN"]],
        on="GAGE_ID",
        how="left",
        validate="one_to_one",
    )
    selected.to_csv(OUTPUT / "six_gauge_selection.csv", index=False)
    return selected


def export_six_gauge_curves(
    selected: pd.DataFrame,
    zero_curves: dict[str, np.ndarray],
    observed: np.ndarray,
    observed_q25: np.ndarray,
    observed_q75: np.ndarray,
    natural: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for station in selected.itertuples(index=False):
        i = int(station.row_index)
        for day in range(8):
            row = {
                "selection_sextile": station.selection_sextile,
                "GAGE_ID": station.GAGE_ID,
                "day": day,
                "observed_median": float(observed[i, day]),
                "observed_q25": float(observed_q25[i, day]),
                "observed_q75": float(observed_q75[i, day]),
            }
            for model in PRIMARY:
                row[f"{model}_zero_input"] = float(zero_curves[model][i, day])
                row[f"{model}_natural_prediction"] = float(natural[model][i, day])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "six_gauge_curves.csv", index=False)
    return frame


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        options = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": "white",
            "metadata": {"Title": stem},
        }
        if suffix == "png":
            options["dpi"] = 600
        fig.savefig(FIGURES / f"{stem}.{suffix}", **options)
    plt.close(fig)


def draw_national_figure(
    pair_time: pd.DataFrame,
    daily: pd.DataFrame,
    skill_grid: pd.DataFrame,
) -> None:
    set_che_style()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_MM * MM, 5.55), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.955, wspace=0.33, hspace=0.31)

    ax = axes[0, 0]
    original = "dHBV – Flux gate"
    convergent = "MCP-MA3 – Neural ODE"
    for pair, frame in pair_time.groupby("pair", sort=False):
        if pair == original:
            ax.plot(frame.day, frame.median_absolute_retention_difference, color=BLUE, lw=1.45, zorder=5)
        elif pair == convergent:
            ax.plot(frame.day, frame.median_absolute_retention_difference, color=ORANGE, lw=1.25, zorder=4)
        else:
            ax.plot(frame.day, frame.median_absolute_retention_difference, color="#A0A0A0", lw=0.65, alpha=0.78)
    reference_line(ax, "x", 7)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 0.26)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Elapsed dry-down time (days)")
    ax.set_ylabel("Median absolute retention difference")
    che_grid(ax)
    panel_label(ax, "a", "Disagreement is not confined to day 7")
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.45, label="dHBV vs. Flux gate"),
            Line2D([], [], color=ORANGE, lw=1.25, label="MCP-MA3 vs. Neural ODE"),
            Line2D([], [], color="#A0A0A0", lw=0.75, label="Other eight pairs"),
        ],
        loc="upper right",
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0.4,
        fontsize=6.7,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
    )

    ax = axes[0, 1]
    for label, color, marker in (
        ("dHBV – Flux gate", BLUE, "o"),
        (convergent, ORANGE, "s"),
    ):
        local = skill_grid.loc[skill_grid.pair.eq(label)].sort_values("kge_threshold")
        ax.plot(
            local.kge_threshold,
            local.median_absolute_r7_difference,
            color=color,
            lw=1.25,
            marker=marker,
            ms=3.8,
            zorder=4,
        )
        ax.plot(
            local.kge_threshold,
            local.median_absolute_r30_difference,
            color=color,
            lw=1.05,
            ls="--",
            marker=marker,
            ms=3.4,
            mfc="white",
            zorder=4,
        )
    reference_line(ax, "y", 0.05, color=INK)
    pair_legend = ax.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=1.25, label="dHBV–Flux"),
            Line2D([], [], color=ORANGE, lw=1.25, label="MCP-MA3–Neural ODE"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        fontsize=5.9,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=1.0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
    )
    ax.add_artist(pair_legend)
    ax.legend(
        handles=[
            Line2D([], [], color=INK, lw=1.1, marker="o", ms=3.2, label="Day 7"),
            Line2D([], [], color=INK, lw=1.0, ls="--", marker="o", ms=3.0,
                   mfc="white", label="Day 30"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=2,
        fontsize=5.9,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=1.2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
    )
    ax.text(0.595, 0.166, "n = 446", ha="right", va="bottom", fontsize=5.7, color=BLUE)
    ax.text(0.595, 0.098, "n = 86", ha="right", va="bottom", fontsize=5.7, color=ORANGE)
    ax.set_xlim(0.285, 0.615)
    ax.set_ylim(0, 0.22)
    ax.set_xticks(SKILL_THRESHOLDS)
    ax.set_xticklabels([f"≥ {v:.2f}" for v in SKILL_THRESHOLDS])
    ax.set_xlabel("KGE required for both models")
    ax.set_ylabel("Median absolute retention difference")
    che_grid(ax)
    panel_label(ax, "b", "Day-7 convergence does not ensure day-30 convergence")

    for column, source, title, letter in (
        (axes[1, 0], "zero-input response", "Zero-input response versus observed streamflow recessions", "c"),
        (axes[1, 1], "natural-event prediction", "Predicted versus observed streamflow recessions", "d"),
    ):
        frame = daily.loc[daily.source.eq(source)]
        for model in PRIMARY:
            local = frame.loc[frame.model.eq(model)]
            column.plot(
                local.day,
                local.spearman_rho,
                lw=1.2,
                marker="o",
                ms=2.6,
                mec=MODEL_STYLE[model]["color"],
                mew=0.35,
                label=LABEL[model],
                **MODEL_STYLE[model],
            )
        reference_line(column, "y", 0.60)
        column.set_xlim(1, 7)
        column.set_ylim(0.20, 0.62)
        column.set_xticks(range(1, 8))
        column.set_yticks(np.arange(0.2, 0.61, 0.1))
        column.set_xlabel("Recession day")
        che_grid(column)
        panel_label(column, letter, title)
    axes[1, 0].set_ylabel(r"Spearman $\rho$ with observed streamflow recessions")
    axes[1, 1].tick_params(labelleft=False)
    handles = [Line2D([], [], lw=1.2, label=LABEL[m], **MODEL_STYLE[m]) for m in PRIMARY]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=5,
        columnspacing=1.3,
        handlelength=2.2,
        handletextpad=0.45,
    )
    save_figure(fig, "candidate_main_full_curve_robustness")


def draw_high_skill_figure(
    skill_grid: pd.DataFrame,
    gauges: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> None:
    set_che_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_MM * MM, 3.05))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.275, top=0.90, wspace=0.27)
    pair = "MCP-MA3 – Neural ODE"
    ax = axes[0]
    first, second = "mcp_ma3", "mc_node"
    for threshold, color, width in zip(
        SKILL_THRESHOLDS,
        ["#C9C9C9", "#858585", ORANGE, BLUE],
        [0.75, 0.85, 1.05, 1.40],
    ):
        mask = (
            gauges["observed_usable"].to_numpy(bool)
            & (gauges[f"{first}_kge"].to_numpy(float) >= threshold)
            & (gauges[f"{second}_kge"].to_numpy(float) >= threshold)
        )
        a = curves[first][mask]
        b = curves[second][mask]
        values = np.median(np.abs(a[:, 1:31] - b[:, 1:31]), axis=0)
        ax.plot(
            np.arange(1, 31),
            values,
            color=color,
            lw=width,
            label=f"KGE ≥ {threshold:.2f} (n = {mask.sum():,})",
        )
    reference_line(ax, "x", 7)
    reference_line(ax, "y", 0.05, color=INK)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 0.10)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Elapsed dry-down time (days)")
    ax.set_ylabel("Median absolute retention difference")
    ax.legend(
        loc="upper left",
        fontsize=6.6,
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0.4,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
    )
    che_grid(ax)
    panel_label(ax, "a", "High-skill agreement depends on the horizon")

    ax = axes[1]
    styles = [
        ("dHBV – Flux gate", BLUE, "o"),
        (pair, ORANGE, "s"),
    ]
    for label, color, marker in styles:
        local = skill_grid.loc[skill_grid.pair.eq(label)].sort_values("kge_threshold")
        ax.plot(
            local.kge_threshold,
            local.median_absolute_r7_difference,
            color=color,
            lw=1.25,
            marker=marker,
            ms=4.0,
            label=("dHBV–Flux: day 7" if label.startswith("dHBV") else "MCP-MA3–NODE: day 7"),
        )
        ax.plot(
            local.kge_threshold,
            local.median_absolute_r30_difference,
            color=color,
            lw=1.05,
            ls="--",
            marker=marker,
            ms=3.5,
            mfc="white",
            label=("dHBV–Flux: day 30" if label.startswith("dHBV") else "MCP-MA3–NODE: day 30"),
        )
        for row in local.itertuples(index=False):
            if row.kge_threshold in (0.30, 0.60):
                left_edge = row.kge_threshold == 0.30
                ax.text(
                    row.kge_threshold + (0.005 if left_edge else -0.005),
                    row.median_absolute_r30_difference + 0.006,
                    f"n = {int(row.gauges):,}",
                    color=color,
                    fontsize=6.1,
                    ha="left" if left_edge else "right",
                )
    reference_line(ax, "y", 0.05, color=INK)
    ax.set_xlim(0.285, 0.615)
    ax.set_ylim(0, 0.17)
    ax.set_xticks(SKILL_THRESHOLDS)
    ax.set_xticklabels([f"≥ {v:.2f}" for v in SKILL_THRESHOLDS])
    ax.set_xlabel("KGE required for both models")
    ax.set_ylabel("Median absolute retention difference")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.015),
        ncol=2,
        fontsize=6.2,
        handlelength=2.2,
        handletextpad=0.5,
        columnspacing=1.2,
        labelspacing=0.25,
    )
    che_grid(ax)
    panel_label(ax, "b", "Day-7 convergence does not ensure day-30 convergence")
    save_figure(fig, "candidate_high_skill_horizon_sensitivity")


def draw_six_gauge_figure(selected: pd.DataFrame, source: pd.DataFrame) -> None:
    set_che_style()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH_MM * MM, 4.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.072, right=0.99, bottom=0.145, top=0.95, wspace=0.12, hspace=0.23)

    for ax, station in zip(axes.flat, selected.itertuples(index=False)):
        frame = source.loc[source.GAGE_ID.eq(station.GAGE_ID)].sort_values("day")
        day = frame.day.to_numpy(float)
        ax.fill_between(day, frame.observed_q25, frame.observed_q75, color=LIGHT, alpha=0.58, lw=0, zorder=1)
        ax.plot(day, frame.observed_median, color=INK, lw=1.35, ls="-", zorder=5)
        for model in PRIMARY:
            values = frame[f"{model}_zero_input"].to_numpy(float)
            ax.plot(day, values, lw=1.05, zorder=4, **MODEL_STYLE[model])
        reference_line(ax, "x", 7)
        ax.set_xlim(0, 7.12)
        ax.set_ylim(0, 1.06)
        ax.set_xticks(range(0, 8))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        letter = chr(97 + int(station.selection_sextile) - 1)
        panel_label(ax, letter, f"USGS {station.GAGE_ID}; {station.ecoregion}")
        station_name = str(station.station_name).title()
        ax.text(
            0.02,
            0.035,
            f"{station_name}\n"
            f"Observed $R_7$ = {station.observed_median_r7:.3f};  "
            f"{int(station.observed_curve_events)} events;  minimum KGE = {station.minimum_five_model_kge:.2f}",
            transform=ax.transAxes,
            va="bottom",
            fontsize=5.65,
            linespacing=1.20,
            color=TEXT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        )
        che_grid(ax)

    for ax in axes[-1, :]:
        ax.set_xlabel("Drying time (days)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized discharge, $Q(t)/Q(0)$")

    handles = [
        Line2D([], [], color=INK, lw=1.35, ls="-", label="Observed median"),
        Patch(facecolor=LIGHT, edgecolor="none", alpha=0.58, label="Observed interquartile range"),
    ] + [Line2D([], [], lw=1.15, label=LABEL[m], **MODEL_STYLE[m]) for m in PRIMARY]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        columnspacing=1.25,
        handlelength=2.3,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    save_figure(fig, "candidate_main_six_representative_gauges")


def write_results_summary(
    pair_summary: pd.DataFrame,
    skill_grid: pd.DataFrame,
    observed_summary: pd.DataFrame,
    selected: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> None:
    original = pair_summary.loc[pair_summary.pair.eq("dHBV – Flux gate")].iloc[0]
    high_skill = skill_grid.loc[
        skill_grid.pair.eq("MCP-MA3 – Neural ODE") & skill_grid.kge_threshold.eq(0.60)
    ].iloc[0]
    lines = [
        "# Full-curve and natural-recession robustness results",
        "",
        "These are post-evaluation checks using frozen outputs; no model was trained or refitted.",
        "",
        "## Original dHBV–flux-gate pair",
        "",
        f"- Mutually skill-qualified gauges with observed recessions: {int(original.gauges):,}.",
        f"- Median mean absolute curve distance over days 1–30: {original.median_curve_distance_d1_d30:.3f} "
        f"(ecoregion-stratified 95% interval {original.curve_distance_ci_low:.3f}–{original.curve_distance_ci_high:.3f}).",
        f"- Median absolute differences were R3={original.median_absolute_r3_difference:.3f}, "
        f"R7={original.median_absolute_r7_difference:.3f}, R14={original.median_absolute_r14_difference:.3f}, "
        f"and R30={original.median_absolute_r30_difference:.3f}.",
        f"- Median absolute 30-day curve-area difference: {original.median_absolute_auc30_difference:.3f}.",
        f"- Median absolute curve-derived half-life difference: {original.median_absolute_half_life_difference_days:.1f} days.",
        f"- At KGE ≥ 0.60, MCP-MA3 and the neural ODE converged at day 7 "
        f"(|ΔR7|={high_skill.median_absolute_r7_difference:.3f}, n={int(high_skill.gauges):,}) but not at day 30 "
        f"(|ΔR30|={high_skill.median_absolute_r30_difference:.3f}); their mean days-1–30 curve distance was "
        f"{high_skill.median_curve_distance_d1_d30:.3f}.",
        "",
        "## Observed recession-curve checks",
        "",
    ]
    for model in PRIMARY:
        for source in ("zero-input response", "natural-event prediction"):
            row = observed_summary.loc[
                observed_summary.model.eq(model) & observed_summary.source.eq(source)
            ].iloc[0]
            lines.append(
                f"- {LABEL[model]}, {source}: n={int(row.gauges):,}, median seven-day curve MAE="
                f"{row.median_curve_mae_days1_7:.3f}, AUC7 rank correlation={row.auc7_spearman_rho:.3f}, "
                f"early/late decay rank correlations={row.early_decay_spearman_rho:.3f}/{row.late_decay_spearman_rho:.3f}."
            )
    lines.extend(
        [
            "",
            "## Six-gauge illustration",
            "",
            "Six Tier-1 gauges were selected near the centers of six equal-frequency bins of observed R7, "
            "with a global minimum-distance assignment requiring six different ecoregions. Eligibility required "
            "at least five observed events and KGE ≥ 0.30 for all five models. No model error or response "
            "agreement entered the selection rule.",
            "",
        ]
    )
    for row in selected.itertuples(index=False):
        lines.append(
            f"- Sextile {int(row.selection_sextile)}: USGS {row.GAGE_ID}, {str(row.station_name).title()} "
            f"({row.ecoregion}); observed R7={row.observed_median_r7:.3f}, events={int(row.observed_curve_events)}."
        )
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            f"The largest difference between rebuilt day-7 curves and archived R7 fingerprints was "
            f"{reconciliation.maximum_absolute_r7_reconciliation_error.max():.2e}.",
            "",
        ]
    )
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gauges_path = MULTI / "outputs" / "comparison" / "gauge_level_multimodel.parquet"
    events_path = PROJECT / "outputs" / "observed_recessions" / "recession_events.parquet"
    gauges = pd.read_parquet(gauges_path)
    gauges["GAGE_ID"] = gauges.GAGE_ID.astype(str).str.zfill(8)
    gage_ids = gauges.GAGE_ID.to_numpy(str)
    events = pd.read_parquet(events_path)
    events["GAGE_ID"] = events.GAGE_ID.astype(str).str.zfill(8)
    events = events.loc[events.scenario.eq("primary")].copy()

    input_hashes = {
        str(gauges_path.relative_to(PROJECT)): sha256(gauges_path),
        str(events_path.relative_to(PROJECT)): sha256(events_path),
    }
    zero_curves, curve_hashes = build_zero_input_curves(gage_ids)
    input_hashes.update(curve_hashes)
    reconciliation = validate_r7(gauges, zero_curves)
    observed, obs_q25, obs_q75, obs_counts, natural, natural_counts = build_natural_event_curves(
        events, gage_ids, input_hashes
    )
    observed_reconciliation = validate_observed_r7(gauges, observed)
    pair_summary, pair_time = pairwise_curve_analysis(gauges, zero_curves)
    skill_grid = pairwise_skill_grid(gauges, zero_curves)
    daily, observed_summary = observational_curve_analysis(
        gauges, zero_curves, observed, obs_counts, natural, natural_counts
    )
    selected = select_six_gauges(gauges, zero_curves, observed, obs_counts)
    six_source = export_six_gauge_curves(selected, zero_curves, observed, obs_q25, obs_q75, natural)

    draw_national_figure(pair_time, daily, skill_grid)
    draw_high_skill_figure(skill_grid, gauges, zero_curves)
    draw_six_gauge_figure(selected, six_source)
    write_results_summary(pair_summary, skill_grid, observed_summary, selected, reconciliation)

    receipt = {
        "analysis": "post-evaluation full-curve and natural-recession robustness",
        "models_retrained": False,
        "primary_fingerprint_changed": False,
        "primary_fingerprint": "R7 remains prespecified",
        "kge_threshold": KGE_THRESHOLD,
        "initial_flow_threshold_mm_day": Q0_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": RNG_SEED,
        "input_sha256": input_hashes,
        "observed_r7_reconciliation_maximum_absolute_error": float(
            observed_reconciliation.maximum_absolute_r7_reconciliation_error.iloc[0]
        ),
        "outputs": {},
    }
    for path in sorted(list(OUTPUT.iterdir()) + list(FIGURES.iterdir()) + [HERE / "RESULTS.md"]):
        if path.is_file():
            receipt["outputs"][str(path.relative_to(HERE))] = sha256(path)
    receipt_path = PROVENANCE / "curve_robustness_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print((HERE / "RESULTS.md").read_text(), flush=True)
    print(f"Wrote {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
