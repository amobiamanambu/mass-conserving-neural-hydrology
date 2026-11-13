#!/usr/bin/env python3
"""Post-opening amended analysis (2026-11-08).

Extends the prespecified analysis (script 09) without modifying it:

1. closure eligibility under the prespecified rule AND under the float64 replay audit
   (script 08b) that removes float32 round-off from the author dHBV accounting;
2. all ten architecture pairs (five lineages), with signed differences, one-sidedness,
   and the within-architecture seed null evaluated at the same mutually qualified gauges;
3. per-model national R7 statistics, R7 > 1 frequencies and rising-discharge frequencies;
4. reference / non-reference / low-snow subgroups for all pairs;
5. training-length sensitivity from the epoch-10/20 checkpoint interventions (scripts
   06/07/08 --epoch), when outputs/sensitivity exists;
6. the MC-LSTM batch-composition sensitivity (script 15), when present;
7. dHBV internal state retention at the common high-skill gauges from the float64 states.

Everything is written to outputs/evaluation_amended/.  Target discharge is already open;
this script uses the archived national_gauge_metrics.csv.gz for KGE.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/evaluation"
FROZEN = ROOT / "outputs/frozen"
SENS = ROOT / "outputs/sensitivity"
OUT = ROOT / "outputs/evaluation_amended"
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
RANK_THRESHOLD = 0.75
MAGNITUDE_THRESHOLD = 0.05
BOOTSTRAP = 2000
MIN_PAIR = 10
MIN_HIGH = 30
LOW_SNOW_PCT = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(a[valid], b[valid]).statistic)


def nanmedian(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def persistence_class(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid] = np.digitize(values[valid], [0.50, np.sqrt(0.50)]).astype(np.int8)
    return result


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / denom


def stratified_ci(first, second, regions, mask, key):
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        return dict(rho_lo=np.nan, rho_hi=np.nan, gap_lo=np.nan, gap_hi=np.nan)
    groups = [indices[regions[indices] == item] for item in np.unique(regions[indices])]
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    rhos = np.full(BOOTSTRAP, np.nan)
    gaps = np.full(BOOTSTRAP, np.nan)
    diff_all = first - second
    for i in range(BOOTSTRAP):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        rhos[i] = rho(first[sample], second[sample])
        gaps[i] = np.nanmedian(np.abs(diff_all[sample]))
    return dict(rho_lo=float(np.nanquantile(rhos, 0.025)), rho_hi=float(np.nanquantile(rhos, 0.975)),
                gap_lo=float(np.nanquantile(gaps, 0.025)), gap_hi=float(np.nanquantile(gaps, 0.975)))


def load_model(model: str, gauges: list[str]):
    preds, r7_seed, r30_seed, curves = [], [], [], []
    rising = []
    for seed in SEEDS:
        archive = np.load(FROZEN / f"{model}_seed{seed}_stock_predictions.npz")
        assert list(archive["gauge_ids"].astype(str)) == gauges
        preds.append(archive["q_sim_mm_day"].astype(np.float64))
        curve_path = FROZEN / f"{model}_seed{seed}_intervention_curves.npz"
        if model == "author_dhbv":
            replay = FROZEN / f"{model}_seed{seed}_float64_replay_curves.npz"
            if replay.exists():
                curve_path = replay
        curve_archive = np.load(curve_path)
        q = curve_archive["q_mm_day"].astype(np.float64)
        q0 = q[:, :, 0]
        valid = np.isfinite(q0) & (q0 >= 0.001)
        normalized = np.divide(q, q0[:, :, None], out=np.full_like(q, np.nan), where=valid[:, :, None])
        with np.errstate(all="ignore"):
            r7_seed.append(np.nanmedian(np.where(valid, normalized[:, :, 7], np.nan), axis=1))
            r30_seed.append(np.nanmedian(np.where(valid, normalized[:, :, 30], np.nan), axis=1))
            curves.append(np.nanmedian(normalized, axis=1))
        rises = (np.diff(q[:, :, :8], axis=2) > 1e-9).any(axis=2) & valid
        rising.append(rises.sum() / max(valid.sum(), 1))
    with np.errstate(all="ignore"):
        return dict(
            prediction=np.mean(np.stack(preds), axis=0),
            r7_seed=np.stack(r7_seed),
            r30_seed=np.stack(r30_seed),
            r7=np.nanmedian(np.stack(r7_seed), axis=0),
            r30=np.nanmedian(np.stack(r30_seed), axis=0),
            curve=np.nanmedian(np.stack(curves), axis=0),
            rising_anchor_fraction=float(np.mean(rising)),
        )


def eligibility() -> dict:
    registry = json.loads((ROOT / "provenance/preopening_freeze_registry.json").read_text())
    prespecified = registry["closure_eligibility_all_three_seeds"]
    result = {}
    for model in MODELS:
        entry = {"prespecified_three_seed_closure": bool(prespecified[model])}
        receipts = [FROZEN / f"{model}_seed{s}_float64_replay_receipt.json" for s in SEEDS]
        if all(p.exists() for p in receipts):
            loaded = [json.loads(p.read_text()) for p in receipts]
            entry["float64_replay_available"] = True
            entry["float64_replay_passes_prespecified_tolerance_all_seeds"] = bool(
                all(r["passes_prespecified_tolerance_in_float64"] for r in loaded))
            entry["float64_replay_max_historical_cumulative_residual_mm"] = max(
                r["maximum_historical_cumulative_residual_mm"] for r in loaded)
            entry["float64_replay_max_intervention_cumulative_drift_mm"] = max(
                r["maximum_intervention_total_cumulative_drift_mm"] for r in loaded)
            entry["float64_replay_max_q_difference_vs_float32_mm_day"] = max(
                r["maximum_q_difference_float64_vs_frozen_float32_mm_day"] for r in loaded)
            entry["float64_replay_max_abs_r7_difference_vs_float32"] = max(
                r["maximum_absolute_r7_difference_float64_vs_float32"] for r in loaded)
        else:
            entry["float64_replay_available"] = False
        entry["admitted_under_float64_audit"] = bool(
            entry["prespecified_three_seed_closure"]
            or entry.get("float64_replay_passes_prespecified_tolerance_all_seeds", False))
        result[model] = entry
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    gauge = pd.read_csv(EVAL / "national_gauge_metrics.csv.gz", dtype={"GAGE_ID": str})
    gauge["GAGE_ID"] = gauge.GAGE_ID.str.zfill(8)
    assert gauge.GAGE_ID.tolist() == gauges
    attributes = pd.read_csv(ROOT / "data/generic_blind/attributes/attributes.csv", dtype={"basin": str})
    attributes["basin"] = attributes.basin.str.zfill(8)
    snow = attributes.set_index("basin").reindex(gauges)["SNOW_PCT_PRECIP"].to_numpy(float)
    regions = gauge.ecoregion.fillna("Unknown").astype(str).to_numpy()
    reference = gauge.reference_status.eq("Reference").to_numpy()
    observed = gauge.observed_r7.to_numpy(float)
    observed_usable = gauge.observed_usable.to_numpy(bool) & np.isfinite(observed)

    elig = eligibility()
    (OUT / "amended_eligibility.json").write_text(json.dumps(elig, indent=2) + "\n")
    admitted = tuple(m for m in MODELS if elig[m]["admitted_under_float64_audit"])
    prespecified = tuple(m for m in MODELS if elig[m]["prespecified_three_seed_closure"])

    data = {m: load_model(m, gauges) for m in MODELS}
    K = {m: gauge[f"{m}_kge"].to_numpy(float) for m in MODELS}
    float64_r7_change = {}
    for m in MODELS:
        difference = float(np.nanmax(np.abs(gauge[f"{m}_r7"].to_numpy(float) - data[m]["r7"])))
        if m != "author_dhbv":
            assert difference < 1e-9
        float64_r7_change[m] = difference

    # Canonical post-audit gauge table: prediction skill is unchanged, while
    # dHBV R7/R30 now comes from the same float64 replay used for closure.
    amended_gauge = gauge.copy()
    for m in MODELS:
        amended_gauge[f"{m}_r7"] = data[m]["r7"]
        amended_gauge[f"{m}_r30"] = data[m]["r30"]
    amended_gauge.to_csv(OUT / "amended_gauge_metrics.csv.gz", index=False)

    # 1. per-model summary -------------------------------------------------------
    arch_rows = []
    for m in MODELS:
        r7 = data[m]["r7"]
        seed_pairs = [(i, j) for i, j in itertools.combinations(range(3), 2)]
        sp_gap = [nanmedian(np.abs(data[m]["r7_seed"][i] - data[m]["r7_seed"][j])) for i, j in seed_pairs]
        sp_rho = [rho(data[m]["r7_seed"][i], data[m]["r7_seed"][j]) for i, j in seed_pairs]
        arch_rows.append({
            "model": m,
            "prespecified_closure_eligible": elig[m]["prespecified_three_seed_closure"],
            "admitted_under_float64_audit": elig[m]["admitted_under_float64_audit"],
            "median_kge": nanmedian(K[m]),
            "median_r7": nanmedian(r7),
            "q25_r7": float(np.nanquantile(r7, 0.25)),
            "q75_r7": float(np.nanquantile(r7, 0.75)),
            "fraction_gauges_r7_above_1": float(np.nanmean(r7 > 1.0)),
            "fraction_gauges_r7_above_1_low_snow": float(np.nanmean(r7[snow < LOW_SNOW_PCT] > 1.0)),
            "fraction_anchors_with_rising_q_days_0_7": data[m]["rising_anchor_fraction"],
            "median_r7_day14": nanmedian(data[m]["curve"][:, 14]),
            "median_r7_day30": nanmedian(data[m]["curve"][:, 30]),
            "median_r7_day60": nanmedian(data[m]["curve"][:, 60]),
            "seed_pair_median_abs_r7_difference_mean": float(np.mean(sp_gap)),
            "seed_pair_r7_spearman_mean": float(np.mean(sp_rho)),
            "observed_r7_spearman_kge_ge_0_60": rho(r7[observed_usable & (K[m] >= 0.6)], observed[observed_usable & (K[m] >= 0.6)]),
            "observed_r7_signed_median_error_kge_ge_0_60": nanmedian((r7 - observed)[observed_usable & (K[m] >= 0.6)]),
        })
    pd.DataFrame(arch_rows).to_csv(OUT / "amended_architecture_summary.csv", index=False)

    # 2. all pairs -------------------------------------------------------------------
    hydro_corr = {}
    for a, b in itertools.combinations(MODELS, 2):
        hydro_corr[(a, b)] = rowwise_pearson(data[a]["prediction"], data[b]["prediction"])
    rows, daily_rows, subgroup_rows = [], [], []
    group_masks = {
        "all_targets": np.ones(len(gauges), bool),
        "reference_targets": reference,
        "nonreference_targets": ~reference,
        f"low_snow_targets_lt_{int(LOW_SNOW_PCT)}pct": snow < LOW_SNOW_PCT,
        f"snow_targets_ge_{int(LOW_SNOW_PCT)}pct": snow >= LOW_SNOW_PCT,
    }
    for a, b in itertools.combinations(MODELS, 2):
        r7a, r7b = data[a]["r7"], data[b]["r7"]
        class_a, class_b = persistence_class(r7a), persistence_class(r7b)
        for thr in THRESHOLDS:
            mask = (np.isfinite(K[a]) & np.isfinite(K[b]) & (K[a] >= thr) & (K[b] >= thr)
                    & np.isfinite(r7a) & np.isfinite(r7b))
            n = int(mask.sum())
            diff = r7a[mask] - r7b[mask]
            gap_curve = np.abs(data[a]["curve"][mask, :31] - data[b]["curve"][mask, :31])
            within = {}
            for label, m in (("first", a), ("second", b)):
                seeds = data[m]["r7_seed"]
                sp = [nanmedian(np.abs(seeds[i][mask] - seeds[j][mask])) for i, j in itertools.combinations(range(3), 2)]
                sr = [rho(seeds[i][mask], seeds[j][mask]) for i, j in itertools.combinations(range(3), 2)]
                loo = [nanmedian(np.abs(seeds[i][mask] - np.nanmedian(np.delete(seeds, i, axis=0), axis=0)[mask])) for i in range(3)]
                onesided = [float(np.nanmean(seeds[i][mask] > seeds[j][mask])) for i, j in itertools.combinations(range(3), 2)]
                within[f"{label}_seed_pair_median_abs_r7_gap_mean"] = float(np.mean(sp))
                within[f"{label}_seed_pair_median_abs_r7_gap_max"] = float(np.max(sp))
                within[f"{label}_seed_pair_r7_spearman_mean"] = float(np.mean(sr))
                within[f"{label}_loo_seed_vs_median_gap_mean"] = float(np.mean(loo))
                within[f"{label}_seed_pair_fraction_first_seed_higher_max_deviation_from_half"] = float(max(abs(x - 0.5) for x in onesided))
            ci = stratified_ci(r7a, r7b, regions, mask, f"{a}|{b}|{thr:.2f}") if n >= 3 else dict(rho_lo=np.nan, rho_hi=np.nan, gap_lo=np.nan, gap_hi=np.nan)
            cross_gap = nanmedian(np.abs(diff)) if n else np.nan
            # conservative within-architecture reference: the larger of the two models' largest single-seed-pair gaps
            # at the same gauges (the same quantity that the CI test and Table 4 use)
            within_ref = max(within["first_seed_pair_median_abs_r7_gap_max"], within["second_seed_pair_median_abs_r7_gap_max"]) if n else np.nan
            rows.append({
                "first": a, "second": b, "skill_threshold": thr, "mutually_qualified_gauges": n,
                "first_prespecified_eligible": a in prespecified, "second_prespecified_eligible": b in prespecified,
                "first_admitted_float64": a in admitted, "second_admitted_float64": b in admitted,
                "pair_in_prespecified_analysis": (a in prespecified and b in prespecified),
                "pair_in_amended_analysis": (a in admitted and b in admitted),
                "median_hydrograph_correlation": nanmedian(hydro_corr[(a, b)][mask]) if n else np.nan,
                "r7_spearman": rho(r7a[mask], r7b[mask]) if n else np.nan,
                "r7_spearman_ci95_low": ci["rho_lo"], "r7_spearman_ci95_high": ci["rho_hi"],
                "median_absolute_r7_gap": cross_gap,
                "median_absolute_r7_gap_ci95_low": ci["gap_lo"], "median_absolute_r7_gap_ci95_high": ci["gap_hi"],
                "median_signed_r7_difference_first_minus_second": nanmedian(diff) if n else np.nan,
                "fraction_first_r7_higher": float(np.nanmean(diff > 0)) if n else np.nan,
                "median_normalized_curve_gap_days_0_30": nanmedian(np.nanmean(gap_curve, axis=1)) if n else np.nan,
                "median_signed_curve_difference_day_30": nanmedian(data[a]["curve"][mask, 30] - data[b]["curve"][mask, 30]) if n else np.nan,
                "same_persistence_class_fraction": float(np.mean(class_a[mask] == class_b[mask])) if n else np.nan,
                "cross_over_within_seed_gap_ratio": (cross_gap / within_ref) if (n and within_ref and np.isfinite(within_ref) and within_ref > 0) else np.nan,
                **within,
            })
            if n:
                for day in range(31):
                    daily_rows.append({"first": a, "second": b, "skill_threshold": thr, "day": day,
                                       "median_absolute_gap": float(np.nanmedian(gap_curve[:, day])),
                                       "median_signed_difference": float(np.nanmedian(data[a]["curve"][mask, day] - data[b]["curve"][mask, day]))})
            if thr in (0.30, 0.60):
                for gname, gmask in group_masks.items():
                    sub = mask & gmask
                    nn = int(sub.sum())
                    d = r7a[sub] - r7b[sub]
                    subgroup_rows.append({
                        "group": gname, "first": a, "second": b, "skill_threshold": thr, "gauges": nn,
                        "r7_spearman": rho(r7a[sub], r7b[sub]) if nn else np.nan,
                        "median_absolute_r7_difference": nanmedian(np.abs(d)) if nn else np.nan,
                        "median_signed_r7_difference": nanmedian(d) if nn else np.nan,
                        "same_persistence_class_fraction": float(np.mean(class_a[sub] == class_b[sub])) if nn else np.nan,
                    })
    pairwise = pd.DataFrame(rows)
    pairwise["evaluable_pair"] = pairwise.mutually_qualified_gauges.ge(MIN_PAIR) & np.isfinite(pairwise.r7_spearman)
    pairwise["material_disagreement"] = pairwise.evaluable_pair & (
        pairwise.r7_spearman.lt(RANK_THRESHOLD) | pairwise.median_absolute_r7_gap.gt(MAGNITUDE_THRESHOLD))
    pairwise["high_skill_convergence"] = (pairwise.skill_threshold.eq(0.60) & pairwise.mutually_qualified_gauges.ge(MIN_HIGH)
                                           & pairwise.r7_spearman.ge(RANK_THRESHOLD) & pairwise.median_absolute_r7_gap.le(MAGNITUDE_THRESHOLD))
    pairwise["cross_gap_exceeds_within_seed_gap_ci"] = pairwise.median_absolute_r7_gap_ci95_low.gt(
        pairwise[["first_seed_pair_median_abs_r7_gap_max", "second_seed_pair_median_abs_r7_gap_max"]].max(axis=1))
    pairwise.to_csv(OUT / "amended_pairwise_summary.csv", index=False)
    pd.DataFrame(daily_rows).to_csv(OUT / "amended_daily_gap_curves.csv", index=False)
    subgroup = pd.DataFrame(subgroup_rows)
    subgroup["material_disagreement"] = subgroup.gauges.ge(MIN_PAIR) & (
        subgroup.r7_spearman.lt(RANK_THRESHOLD) | subgroup.median_absolute_r7_difference.gt(MAGNITUDE_THRESHOLD))
    subgroup.to_csv(OUT / "amended_subgroup_pairwise_summary.csv", index=False)

    # 3. common high-skill sets ---------------------------------------------------
    common = {}
    for label, models in (("prespecified_three", prespecified), ("admitted", admitted), ("all_five", MODELS)):
        mask = np.ones(len(gauges), bool)
        for m in models:
            mask &= np.isfinite(K[m]) & (K[m] >= 0.60)
        common[label] = {"models": list(models), "gauges_kge_ge_0_60_in_all": int(mask.sum())}
    (OUT / "amended_common_high_skill_sets.json").write_text(json.dumps(common, indent=2) + "\n")

    # 4. development-plateau histories ---------------------------------------------
    # This branch replaces the original epoch-10/20/30 sensitivity with the
    # development-only convergence experiment. R7 is computed only at each
    # architecture's selected plateau checkpoint, so it cannot affect stopping.
    histories = []
    for m in admitted:
        path = ROOT / f"development_history_{m}_seed11.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "model", m)
            histories.append(frame)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    history.to_csv(OUT / "amended_development_plateau_history.csv", index=False)

    # 5. MC-LSTM batch sensitivity ---------------------------------------------------
    batch_summary = SENS / "mclstm_seed11_batch_sensitivity.json"
    if batch_summary.exists():
        (OUT / "amended_mclstm_batch_sensitivity.json").write_text(batch_summary.read_text())
        batch = pd.read_csv(
            SENS / "mclstm_seed11_batch_sensitivity.csv",
            dtype={"GAGE_ID": "string"},
        )
        batch["GAGE_ID"] = batch["GAGE_ID"].str.zfill(8)
        batch.to_csv(OUT / "amended_mclstm_batch_sensitivity.csv", index=False)

    # 6. dHBV float64 state retention at common high-skill gauges ---------------------
    state_paths = [FROZEN / f"author_dhbv_seed{s}_float64_replay_curves.npz" for s in SEEDS]
    if all(p.exists() for p in state_paths):
        mask = np.ones(len(gauges), bool)
        for m in admitted:
            mask &= np.isfinite(K[m]) & (K[m] >= 0.60)
        idx = np.flatnonzero(mask)
        names = ["SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ", "ROUTING"]
        retention = {n: [] for n in names}
        share = {n: [] for n in names}
        q_ratio, s_ratio = [], []
        for p in state_paths:
            arch = np.load(p)
            states = arch["states_mm"][idx].astype(np.float64)  # [g, 12, 92, 5]
            routing = arch["routing_buffer_mm"][idx].astype(np.float64)  # [g, 12, 92]
            total0 = states[:, :, 0, :].sum(-1) + routing[:, :, 0]
            for k, n in enumerate(names[:5]):
                s0 = states[:, :, 0, k]
                with np.errstate(all="ignore"):
                    retention[n].append(np.where(s0 > 1e-6, states[:, :, 30, k] / s0, np.nan).ravel())
                    share[n].append((s0 / total0).ravel())
            with np.errstate(all="ignore"):
                r0 = routing[:, :, 0]
                retention["ROUTING"].append(np.where(r0 > 1e-6, routing[:, :, 30] / r0, np.nan).ravel())
                share["ROUTING"].append((r0 / total0).ravel())
                q = arch["q_mm_day"][idx]
                q0 = q[:, :, 0]
                q_ratio.append(np.where(q0 >= 0.001, q[:, :, 30] / q0, np.nan).ravel())
                total30 = states[:, :, 30, :].sum(-1) + routing[:, :, 30]
                s_ratio.append((total30 / total0).ravel())
        rows6 = []
        for n in names:
            vals = np.concatenate(retention[n]); sh = np.concatenate(share[n])
            rows6.append({"model": "author_dhbv", "state": n, "q25": float(np.nanquantile(vals, 0.25)), "median": float(np.nanmedian(vals)),
                          "q75": float(np.nanquantile(vals, 0.75)), "day0_fraction_q25": float(np.nanquantile(sh, 0.25)),
                          "day0_fraction_median": float(np.nanmedian(sh)), "day0_fraction_q75": float(np.nanquantile(sh, 0.75)),
                          "n": int(np.isfinite(vals).sum())})
        pd.DataFrame(rows6).to_csv(OUT / "amended_dhbv_state_retention.csv", index=False)
        qq = np.concatenate(q_ratio); ss = np.concatenate(s_ratio)
        (OUT / "amended_dhbv_day30_summary.json").write_text(json.dumps({
            "common_high_skill_gauges_all_admitted_models": int(mask.sum()),
            "median_q_ratio_day30": float(np.nanmedian(qq)), "q25_q_ratio_day30": float(np.nanquantile(qq, 0.25)), "q75_q_ratio_day30": float(np.nanquantile(qq, 0.75)),
            "median_total_storage_ratio_day30": float(np.nanmedian(ss)), "q25_total_storage_ratio_day30": float(np.nanquantile(ss, 0.25)), "q75_total_storage_ratio_day30": float(np.nanquantile(ss, 0.75)),
        }, indent=2) + "\n")

        # 6b. the same common gauge set for every admitted model: dHBV day-by-day
        #     trajectories from the float64 replay, and the three archived models
        #     (script 13 per-gauge products) re-summarised at the common-4 gauges,
        #     so Fig. 9 compares all admitted models at one population.
        q_curves, s_curves = [], []
        for p in state_paths:
            arch = np.load(p)
            q = arch["q_mm_day"][idx]; q0 = q[:, :, 0]
            with np.errstate(all="ignore"):
                q_curves.append(np.where((q0 >= 0.001)[:, :, None], q[:, :, :31] / q0[:, :, None], np.nan).reshape(-1, 31))
            states = arch["states_mm"][idx].astype(np.float64); routing = arch["routing_buffer_mm"][idx].astype(np.float64)
            total = states.sum(-1) + routing
            s_curves.append((total[:, :, :31] / total[:, :, :1]).reshape(-1, 31))
        q_curves = np.concatenate(q_curves); s_curves = np.concatenate(s_curves)
        rows6b = []
        for day in range(31):
            for metric_name, arr in (("q_ratio", q_curves), ("storage_ratio", s_curves)):
                rows6b.append({"model": "author_dhbv", "day": day, "metric": metric_name,
                               "q25": float(np.nanquantile(arr[:, day], 0.25)), "median": float(np.nanmedian(arr[:, day])),
                               "q75": float(np.nanquantile(arr[:, day], 0.75)), "n": int(np.isfinite(arr[:, day]).sum())})
        pd.DataFrame(rows6b).to_csv(OUT / "amended_dhbv_trajectory_summary.csv", index=False)
        common_ids = set(int(g) for g in np.asarray(gauges)[idx])
        traj = pd.read_csv(EVAL / "internal_process_trajectory_gauge.csv.gz")
        traj = traj[traj.GAGE_ID.astype(int).isin(common_ids) & traj.day.le(30)]
        rows = []
        for (model, day), local in traj.groupby(["model", "day"]):
            for metric_name in ("q_ratio", "storage_ratio", "cumulative_q_fraction", "cumulative_loss_fraction"):
                vals = local[metric_name].to_numpy(float)
                rows.append({"model": model, "day": int(day), "metric": metric_name,
                             "q25": float(np.nanquantile(vals, 0.25)), "median": float(np.nanmedian(vals)),
                             "q75": float(np.nanquantile(vals, 0.75)), "n": int(np.isfinite(vals).sum())})
        pd.DataFrame(rows).to_csv(OUT / "amended_internal_process_trajectory_summary.csv", index=False)
        st = pd.read_csv(EVAL / "internal_state_retention_gauge.csv.gz")
        st = st[st.GAGE_ID.astype(int).isin(common_ids)]
        rows = []
        for (model, state), local in st.groupby(["model", "state"], sort=False):
            vals = local.day30_retention.to_numpy(float); sh = local.day0_fraction_total.to_numpy(float)
            rows.append({"model": model, "state": state, "q25": float(np.nanquantile(vals, 0.25)), "median": float(np.nanmedian(vals)),
                         "q75": float(np.nanquantile(vals, 0.75)), "day0_fraction_q25": float(np.nanquantile(sh, 0.25)),
                         "day0_fraction_median": float(np.nanmedian(sh)), "day0_fraction_q75": float(np.nanquantile(sh, 0.75)),
                         "n": int(np.isfinite(vals).sum())})
        pd.DataFrame(rows).to_csv(OUT / "amended_internal_state_retention_summary.csv", index=False)
        fl = pd.read_csv(EVAL / "internal_process_flux_gauge.csv.gz")
        fl = fl[fl.GAGE_ID.astype(int).isin(common_ids)]
        rows = []
        for (model, process), local in fl.groupby(["model", "process"], sort=False):
            vals = local.day1_30_fraction_initial_storage.to_numpy(float)
            rows.append({"model": model, "process": process, "q25": float(np.nanquantile(vals, 0.25)), "median": float(np.nanmedian(vals)),
                         "q75": float(np.nanquantile(vals, 0.75)), "n": int(np.isfinite(vals).sum())})
        pd.DataFrame(rows).to_csv(OUT / "amended_internal_process_flux_summary.csv", index=False)

    # 7. discovery summary -----------------------------------------------------------
    t60 = pairwise.loc[pairwise.skill_threshold.eq(0.60)]
    t30 = pairwise.loc[pairwise.skill_threshold.eq(0.30)]
    discovery = {
        "prespecified_closure_eligible_models": list(prespecified),
        "admitted_under_float64_audit": list(admitted),
        "float64_replay_available_for_dhbv": elig["author_dhbv"]["float64_replay_available"],
        "maximum_absolute_r7_change_from_float32_replay": float64_r7_change["author_dhbv"],
        "amended_pairs": int(t60.pair_in_amended_analysis.sum()),
        "amended_materially_disagreeing_pairs_kge_0_30": int((t30.pair_in_amended_analysis & t30.material_disagreement).sum()),
        "amended_materially_disagreeing_pairs_kge_0_60": int((t60.pair_in_amended_analysis & t60.material_disagreement).sum()),
        "amended_convergent_pairs_kge_0_60": t60.loc[t60.pair_in_amended_analysis & t60.high_skill_convergence, ["first", "second"]].to_dict("records"),
        "amended_pairs_whose_cross_gap_exceeds_within_seed_gap_kge_0_60": t60.loc[t60.pair_in_amended_analysis & t60.cross_gap_exceeds_within_seed_gap_ci, ["first", "second"]].to_dict("records"),
        "central_non_guarantee_supported_amended": bool((t60.pair_in_amended_analysis & t60.material_disagreement).any()),
        "common_high_skill_sets": common,
        "inputs_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [EVAL / "national_gauge_metrics.csv.gz", ROOT / "provenance/preopening_freeze_registry.json"]},
    }
    (OUT / "amended_discovery.json").write_text(json.dumps(discovery, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(arch_rows).round(3).to_string(index=False))
    cols = ["first", "second", "skill_threshold", "mutually_qualified_gauges", "median_hydrograph_correlation", "r7_spearman",
            "median_absolute_r7_gap", "median_signed_r7_difference_first_minus_second", "fraction_first_r7_higher",
            "median_normalized_curve_gap_days_0_30", "same_persistence_class_fraction", "cross_over_within_seed_gap_ratio", "material_disagreement", "high_skill_convergence"]
    print(t60[cols].round(3).to_string(index=False))
    print(json.dumps(discovery, indent=2))


if __name__ == "__main__":
    main()
