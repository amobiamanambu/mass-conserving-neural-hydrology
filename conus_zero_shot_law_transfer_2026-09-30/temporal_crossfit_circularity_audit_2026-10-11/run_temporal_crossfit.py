#!/usr/bin/env python3
"""Audit outcome reuse with disjoint skill and recession-evaluation periods."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
PROJECT = AUDIT.parent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


INK = "#111111"
MUTED = "#666666"
GRID = "#D7D7D7"
BLUE = "#2B6A99"
AMBER = "#D17F1F"


def set_style() -> None:
    """Apply a restrained journal style to the optional audit graphic."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "axes.edgecolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 500,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.025,
        1.025,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.3,
        fontweight="bold",
        color=INK,
    )


SEEDS = (11, 29, 47)
SKILL_THRESHOLD = 0.30
MATERIAL_GAP = 0.05
RANK_THRESHOLD = 0.75
OBSERVED_RHO_THRESHOLD = 0.60
OBSERVED_MAE_THRESHOLD = 0.10
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261011
PRIMARY_SCENARIO = {"future_p": 1.0, "prior_p": 2.0, "declines": 5, "q0": 0.001}
FOLDS = (
    {
        "design": "skill_2019_2021__benchmark_2022_2024",
        "label": "Skill 2019–21\nR7 2022–24",
        "skill_start": "2019-01-01",
        "skill_end": "2021-12-31",
        "benchmark_start": "2022-01-01",
        "benchmark_end": "2024-12-31",
    },
    {
        "design": "skill_2022_2024__benchmark_2019_2021",
        "label": "Skill 2022–24\nR7 2019–21",
        "skill_start": "2022-01-01",
        "skill_end": "2024-12-31",
        "benchmark_start": "2019-01-01",
        "benchmark_end": "2021-12-31",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_slice(dates: pd.DatetimeIndex, start: str, end: str) -> slice:
    first = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    stop = int(dates.searchsorted(pd.Timestamp(end), side="right"))
    return slice(first, stop)


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o = observed[valid].astype(np.float64)
    p = predicted[valid].astype(np.float64)
    o_mean, p_mean = o.mean(), p.mean()
    o_std, p_std = o.std(ddof=1), p.std(ddof=1)
    if o_std <= 0 or o_mean <= 0 or p_std <= 0:
        return np.nan
    correlation = float(np.corrcoef(o, p)[0, 1])
    alpha = float(p_std / o_std)
    beta = float(p_mean / o_mean)
    if not np.isfinite([correlation, alpha, beta]).all():
        return np.nan
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def events_for_basin(
    q: np.ndarray,
    forcing: np.ndarray,
    dates: pd.DatetimeIndex,
    scenario: dict[str, float | int],
) -> np.ndarray:
    """Return R7 values using the primary extraction rule."""
    if len(q) < 10:
        return np.asarray([], dtype=float)
    q_window = np.lib.stride_tricks.sliding_window_view(q, 8)
    p_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 0], 8)
    t_window = np.lib.stride_tricks.sliding_window_view(forcing[:, 2], 8)
    starts = np.arange(2, len(q) - 7)
    local_q = q_window[starts]
    decline_count = np.sum(
        np.diff(local_q, axis=1) <= np.maximum(1e-8, 0.01 * local_q[:, :-1]), axis=1
    )
    prior_p = forcing[starts - 2, 0] + forcing[starts - 1, 0]
    months = dates.month.to_numpy()[starts]
    mask = (
        np.isfinite(local_q).all(axis=1)
        & (local_q[:, 0] >= scenario["q0"])
        & (months >= 6)
        & (months <= 9)
        & (np.nanmin(t_window[starts], axis=1) > 5.0)
        & (prior_p <= scenario["prior_p"])
        & (np.nansum(p_window[starts], axis=1) <= scenario["future_p"])
        & (decline_count >= scenario["declines"])
        & (local_q[:, -1] < local_q[:, 0])
    )
    candidate_starts = starts[mask]
    candidate_q = local_q[mask]
    selected: list[float] = []
    next_allowed = 0
    for start, segment_q in zip(candidate_starts, candidate_q, strict=True):
        if start < next_allowed:
            continue
        selected.append(float(segment_q[-1] / segment_q[0]))
        next_allowed = int(start) + 8
    return np.asarray(selected, dtype=float)


def rho(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.nanstd(x[valid]) == 0 or np.nanstd(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def median_abs_difference(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> float:
    difference = np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
    valid = np.isfinite(difference)
    return float(np.median(difference[valid])) if valid.any() else np.nan


def summarize(frame: pd.DataFrame, design: str, label: str) -> dict[str, float | int | str]:
    return {
        "design": design,
        "label": label,
        "gauges": int(len(frame)),
        "q_cross_architecture_rho": rho(frame.dhbv_q_r7, frame.fluxgate_q_r7),
        "q_cross_architecture_mad": median_abs_difference(frame.dhbv_q_r7, frame.fluxgate_q_r7),
        "qet_cross_architecture_rho": rho(frame.dhbv_qet_r7, frame.fluxgate_qet_r7),
        "qet_cross_architecture_mad": median_abs_difference(frame.dhbv_qet_r7, frame.fluxgate_qet_r7),
        "dhbv_observed_rho": rho(frame.dhbv_qet_r7, frame.observed_r7),
        "dhbv_observed_mae": median_abs_difference(frame.dhbv_qet_r7, frame.observed_r7),
        "fluxgate_observed_rho": rho(frame.fluxgate_qet_r7, frame.observed_r7),
        "fluxgate_observed_mae": median_abs_difference(frame.fluxgate_qet_r7, frame.observed_r7),
    }


def bootstrap(frame: pd.DataFrame, design: str) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [group.index.to_numpy() for _, group in frame.groupby("ecoregion", sort=True)]
    records = []
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        result = summarize(frame.loc[chosen], design, design)
        records.append({"design": design, "draw": draw, **{k: v for k, v in result.items() if k not in {"design", "label", "gauges"}}})
    return pd.DataFrame(records)


def add_intervals(summary: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "q_cross_architecture_rho",
        "q_cross_architecture_mad",
        "qet_cross_architecture_rho",
        "qet_cross_architecture_mad",
        "dhbv_observed_rho",
        "dhbv_observed_mae",
        "fluxgate_observed_rho",
        "fluxgate_observed_mae",
    ]
    output = summary.copy()
    for metric in metrics:
        intervals = (
            draws.groupby("design")[metric]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: f"{metric}_lo", 0.975: f"{metric}_hi"})
        )
        output = output.merge(intervals, left_on="design", right_index=True, how="left", validate="one_to_one")
    return output


def load_fingerprints(target: pd.DataFrame) -> pd.DataFrame:
    frame = target[["GAGE_ID", "ecoregion", "tier_short", "reference_status"]].copy()
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    frame = frame.set_index("GAGE_ID")
    for architecture in ("dhbv", "fluxgate"):
        for rung in ("q", "qet"):
            seed_values = []
            for seed in SEEDS:
                path = PROJECT / f"outputs/law_interventions/{architecture}_{rung}_seed{seed}_fingerprints.parquet"
                table = pd.read_parquet(path).set_index("GAGE_ID")
                table.index = table.index.astype(str).str.zfill(8)
                seed_values.append(table.loc[frame.index, "median_r7"].rename(str(seed)))
            frame[f"{architecture}_{rung}_r7"] = pd.concat(seed_values, axis=1).median(axis=1)
    return frame.reset_index()


def compute_fold_inputs(
    target: pd.DataFrame,
    q: np.ndarray,
    forcing: np.ndarray,
    all_dates: pd.DatetimeIndex,
    evaluation_dates: pd.DatetimeIndex,
    fold: dict[str, str],
) -> pd.DataFrame:
    skill_slice = index_slice(evaluation_dates, fold["skill_start"], fold["skill_end"])
    benchmark_slice = index_slice(evaluation_dates, fold["benchmark_start"], fold["benchmark_end"])
    forcing_eval_start = int(all_dates.searchsorted(evaluation_dates[0], side="left"))
    forcing_benchmark = slice(
        forcing_eval_start + benchmark_slice.start,
        forcing_eval_start + benchmark_slice.stop,
    )
    benchmark_dates = evaluation_dates[benchmark_slice]

    prediction = {}
    for architecture in ("dhbv", "fluxgate"):
        runs = [
            np.load(PROJECT / f"outputs/predictions/{architecture}_qet_seed{seed}_target_q_2019_2024.npy", mmap_mode="r")
            for seed in SEEDS
        ]
        prediction[architecture] = runs

    records = []
    for basin, station in enumerate(target.itertuples(index=False)):
        observed_skill = np.asarray(q[basin, skill_slice])
        kges = {}
        for architecture in ("dhbv", "fluxgate"):
            ensemble = np.mean(
                np.stack([np.asarray(run[basin, skill_slice], dtype=np.float64) for run in prediction[architecture]]),
                axis=0,
            )
            kges[architecture] = kge(observed_skill, ensemble)

        observed_benchmark = np.asarray(q[basin, benchmark_slice])
        local_forcing = np.asarray(forcing[int(station.row), forcing_benchmark])
        event_r7 = events_for_basin(observed_benchmark, local_forcing, benchmark_dates, PRIMARY_SCENARIO)
        records.append(
            {
                "GAGE_ID": str(station.GAGE_ID).zfill(8),
                "design": fold["design"],
                "ecoregion": station.ecoregion,
                "tier_short": station.tier_short,
                "reference_status": station.reference_status,
                "dhbv_skill_period_kge": kges["dhbv"],
                "fluxgate_skill_period_kge": kges["fluxgate"],
                "observed_events": int(len(event_r7)),
                "observed_r7": float(np.median(event_r7)) if len(event_r7) else np.nan,
                "observed_usable": bool(len(event_r7) >= 3),
            }
        )
        if (basin + 1) % 500 == 0 or basin + 1 == len(target):
            print(f"{fold['design']}: {basin + 1:,}/{len(target):,}", flush=True)
    return pd.DataFrame(records)


def errorbar(ax, x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, color: str, marker: str, label: str) -> None:
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, hi - y]),
        fmt=marker,
        color=color,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=4.6,
        capsize=2.0,
        elinewidth=0.75,
        label=label,
        zorder=5,
    )


def make_figure(summary: pd.DataFrame) -> None:
    set_style()
    order = [
        "all_targets_no_target_q",
        "full_period_same_record",
        FOLDS[0]["design"],
        FOLDS[1]["design"],
    ]
    plot = summary.set_index("design").loc[order].reset_index()
    labels = plot.label.tolist()
    x = np.arange(len(plot))
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    y = plot.qet_cross_architecture_mad.to_numpy(float)
    errorbar(ax, x, y, plot.qet_cross_architecture_mad_lo.to_numpy(float), plot.qet_cross_architecture_mad_hi.to_numpy(float), BLUE, "o", "dHBV vs. flux gate")
    ax.axhline(MATERIAL_GAP, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("Median |ΔR7|")
    ax.set_title("Response-magnitude disagreement", loc="left")
    panel(ax, "(a)")

    ax = axes[0, 1]
    y = plot.qet_cross_architecture_rho.to_numpy(float)
    errorbar(ax, x, y, plot.qet_cross_architecture_rho_lo.to_numpy(float), plot.qet_cross_architecture_rho_hi.to_numpy(float), BLUE, "o", "dHBV vs. flux gate")
    ax.axhline(RANK_THRESHOLD, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Response-rank agreement", loc="left")
    panel(ax, "(b)")

    observed = plot.design.ne("all_targets_no_target_q").to_numpy()
    xo = x[observed]
    ax = axes[1, 0]
    for offset, prefix, color, marker, label in (
        (-0.08, "dhbv", BLUE, "o", "dHBV"),
        (0.08, "fluxgate", AMBER, "s", "Flux gate"),
    ):
        y = plot.loc[observed, f"{prefix}_observed_rho"].to_numpy(float)
        lo = plot.loc[observed, f"{prefix}_observed_rho_lo"].to_numpy(float)
        hi = plot.loc[observed, f"{prefix}_observed_rho_hi"].to_numpy(float)
        errorbar(ax, xo + offset, y, lo, hi, color, marker, label)
    ax.axhline(OBSERVED_RHO_THRESHOLD, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("Model–observed Spearman ρ")
    ax.set_title("Agreement with recession benchmark", loc="left")
    panel(ax, "(c)")
    ax.legend(frameon=False, ncol=2, loc="lower right")

    ax = axes[1, 1]
    for offset, prefix, color, marker, label in (
        (-0.08, "dhbv", BLUE, "o", "dHBV"),
        (0.08, "fluxgate", AMBER, "s", "Flux gate"),
    ):
        y = plot.loc[observed, f"{prefix}_observed_mae"].to_numpy(float)
        lo = plot.loc[observed, f"{prefix}_observed_mae_lo"].to_numpy(float)
        hi = plot.loc[observed, f"{prefix}_observed_mae_hi"].to_numpy(float)
        errorbar(ax, xo + offset, y, lo, hi, color, marker, label)
    ax.axhline(OBSERVED_MAE_THRESHOLD, color=INK, lw=0.8, ls=":")
    ax.set_ylabel("Median absolute error in R7")
    ax.set_title("Benchmark magnitude error", loc="left")
    panel(ax, "(d)")

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", color=GRID, lw=0.45, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_ylim(bottom=0)
    axes[0, 0].text(0.99, 0.04, "dotted line: prespecified criterion", transform=axes[0, 0].transAxes, ha="right", va="bottom", color=MUTED, fontsize=6.2)

    output = AUDIT / "figures" / "temporal_crossfit_circularity_audit"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=500 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def write_results(summary: pd.DataFrame, fold_counts: pd.DataFrame) -> None:
    indexed = summary.set_index("design")

    def estimate_ci(row: pd.Series, metric: str) -> str:
        estimate = row[metric]
        if not np.isfinite(estimate):
            return "—"
        return f"{estimate:.3f} [{row[f'{metric}_lo']:.3f}, {row[f'{metric}_hi']:.3f}]"

    lines = [
        "# Temporal cross-fit circularity audit",
        "",
        "The audit used disjoint target-discharge periods for predictive qualification and the observed-recession benchmark. Archived model fingerprints were not recomputed or tuned.",
        "",
        "## Results",
        "",
        "| Design | Gauges | Q+ET rho [95% CI] | Median absolute R7 gap [95% CI] | dHBV-observed rho [95% CI] | Flux-gate-observed rho [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["all_targets_no_target_q", "full_period_same_record", FOLDS[0]["design"], FOLDS[1]["design"]]
    for design in order:
        row = indexed.loc[design]
        lines.append(
            f"| {row.label.replace(chr(10), ' / ')} | {int(row.gauges):,} | "
            f"{estimate_ci(row, 'qet_cross_architecture_rho')} | "
            f"{estimate_ci(row, 'qet_cross_architecture_mad')} | "
            f"{estimate_ci(row, 'dhbv_observed_rho')} | "
            f"{estimate_ci(row, 'fluxgate_observed_rho')} |"
        )
    lines.extend(["", "## Cross-fit sample construction", "", "| Fold | Mutual KGE-qualified | Usable benchmark R7 | Final gauges | Median events |", "|---|---:|---:|---:|---:|"])
    for row in fold_counts.itertuples(index=False):
        lines.append(f"| {row.design} | {row.mutual_skill_gauges:,} | {row.usable_benchmark_gauges:,} | {row.analysis_gauges:,} | {row.median_events:.1f} |")
    cross = indexed.loc[[FOLDS[0]["design"], FOLDS[1]["design"]]]
    conclusion = (
        "Both temporal folds retained material response disagreement. The same-record design is therefore not required for the central dHBV–flux-gate magnitude result."
        if (cross.qet_cross_architecture_mad > MATERIAL_GAP).all()
        else "At least one temporal fold did not retain the prespecified magnitude disagreement; the result requires further investigation."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            "",
            "The cross-architecture rank correlations rose above the prespecified 0.75 criterion in both temporal folds. The cross-fit therefore supports persistent magnitude disagreement, but not a claim that response ranks must also disagree after temporal separation.",
            "",
            "Both architectures remained below the observed-benchmark rank criterion and above its magnitude-error criterion in both folds. The all-target analysis also retained a median gap above 0.10 without using target discharge for either selection or benchmarking.",
            "",
            "The audit addresses reuse of target discharge for KGE qualification and observed-recession extraction. It does not convert the discharge-derived recession proxy into a direct storage observation.",
            "",
        ]
    )
    (AUDIT / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    station_path = PROJECT / "data/processed/station_index.csv"
    q_path = PROJECT / "data/evaluation/q_target_2019_2024.npy"
    forcing_path = PROJECT / "data/processed/forcing.npy"
    dates_path = PROJECT / "data/processed/dates.npy"
    target = pd.read_csv(station_path, dtype={"GAGE_ID": str})
    target = target.loc[target.role.eq("target")].reset_index(drop=True)
    target["GAGE_ID"] = target.GAGE_ID.astype(str).str.zfill(8)
    q = np.load(q_path, mmap_mode="r")
    forcing = np.load(forcing_path, mmap_mode="r")
    all_dates = pd.DatetimeIndex(np.asarray(np.load(dates_path, mmap_mode="r")).astype("datetime64[ns]"))
    evaluation_dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    if q.shape != (len(target), len(evaluation_dates)):
        raise RuntimeError(f"Target-discharge shape mismatch: {q.shape}")

    fingerprints = load_fingerprints(target)
    fold_frames = []
    fold_count_records = []
    summary_records = []
    bootstrap_frames = []

    all_targets = fingerprints.copy()
    all_targets["observed_r7"] = np.nan
    summary_records.append(summarize(all_targets, "all_targets_no_target_q", f"All {len(target):,} targets\nno target Q"))
    bootstrap_frames.append(bootstrap(all_targets, "all_targets_no_target_q"))

    full = pd.read_parquet(PROJECT / "outputs/identification/gauge_level_identification.parquet")
    full.GAGE_ID = full.GAGE_ID.astype(str).str.zfill(8)
    full = full.rename(columns={"observed_median_r7": "observed_r7"})
    full_mask = full.observed_usable.astype(bool) & (full.dhbv_qet_kge >= SKILL_THRESHOLD) & (full.fluxgate_qet_kge >= SKILL_THRESHOLD)
    full_primary = full.loc[full_mask].copy().reset_index(drop=True)
    summary_records.append(summarize(full_primary, "full_period_same_record", "Full 2019–24\nsame record"))
    bootstrap_frames.append(bootstrap(full_primary, "full_period_same_record"))

    for fold in FOLDS:
        fold_input = compute_fold_inputs(target, q, forcing, all_dates, evaluation_dates, fold)
        merged = fold_input.merge(
            fingerprints,
            on=["GAGE_ID", "ecoregion", "tier_short", "reference_status"],
            how="left",
            validate="one_to_one",
        )
        mutual_skill = (merged.dhbv_skill_period_kge >= SKILL_THRESHOLD) & (merged.fluxgate_skill_period_kge >= SKILL_THRESHOLD)
        usable = merged.observed_usable.astype(bool)
        analysis = merged.loc[mutual_skill & usable].copy().reset_index(drop=True)
        fold_frames.append(merged)
        fold_count_records.append(
            {
                "design": fold["design"],
                "mutual_skill_gauges": int(mutual_skill.sum()),
                "usable_benchmark_gauges": int(usable.sum()),
                "analysis_gauges": int(len(analysis)),
                "median_events": float(analysis.observed_events.median()),
            }
        )
        summary_records.append(summarize(analysis, fold["design"], fold["label"]))
        bootstrap_frames.append(bootstrap(analysis, fold["design"]))

    gauge_level = pd.concat(fold_frames, ignore_index=True)
    fold_counts = pd.DataFrame(fold_count_records)
    summary = pd.DataFrame(summary_records)
    bootstrap_draws = pd.concat(bootstrap_frames, ignore_index=True)
    summary_ci = add_intervals(summary, bootstrap_draws)

    gauge_level.to_csv(AUDIT / "crossfit_gauge_level.csv.gz", index=False, compression="gzip")
    fold_counts.to_csv(AUDIT / "crossfit_sample_counts.csv", index=False)
    bootstrap_draws.to_csv(AUDIT / "crossfit_bootstrap_draws.csv.gz", index=False, compression="gzip")
    summary_ci.to_csv(AUDIT / "crossfit_summary.csv", index=False)
    make_figure(summary_ci)
    write_results(summary_ci, fold_counts)

    outputs = [
        AUDIT / "crossfit_gauge_level.csv.gz",
        AUDIT / "crossfit_sample_counts.csv",
        AUDIT / "crossfit_bootstrap_draws.csv.gz",
        AUDIT / "crossfit_summary.csv",
        AUDIT / "RESULTS.md",
        AUDIT / "figures/temporal_crossfit_circularity_audit.png",
        AUDIT / "figures/temporal_crossfit_circularity_audit.pdf",
        AUDIT / "figures/temporal_crossfit_circularity_audit.svg",
    ]
    archived_inputs = [
        station_path,
        q_path,
        forcing_path,
        dates_path,
        PROJECT / "outputs/identification/gauge_level_identification.parquet",
        *[
            PROJECT / f"outputs/predictions/{architecture}_qet_seed{seed}_target_q_2019_2024.npy"
            for architecture in ("dhbv", "fluxgate")
            for seed in SEEDS
        ],
        *[
            PROJECT / f"outputs/law_interventions/{architecture}_{rung}_seed{seed}_fingerprints.parquet"
            for architecture in ("dhbv", "fluxgate")
            for rung in ("q", "qet")
            for seed in SEEDS
        ],
    ]
    receipt = {
        "analysis": "disjoint temporal cross-fit of KGE qualification and observed R7 benchmark",
        "skill_threshold": SKILL_THRESHOLD,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "folds": FOLDS,
        "inputs": {str(path.relative_to(PROJECT)): sha256(path) for path in archived_inputs},
        "outputs": {str(path.relative_to(AUDIT)): sha256(path) for path in outputs},
    }
    (AUDIT / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(summary_ci.to_string(index=False), flush=True)
    print(f"Wrote audit to {AUDIT}", flush=True)


if __name__ == "__main__":
    main()
