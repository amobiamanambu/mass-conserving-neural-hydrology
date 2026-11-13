#!/usr/bin/env python3
"""Compare zero-PET and climatological-PET drainage interventions."""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/frozen"
PET = ROOT / "outputs/sensitivity_pet_climatology"
EVAL = ROOT / "outputs/evaluation"
OUT = ROOT / "outputs/evaluation_pet_climatology"
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
RANK_THRESHOLD = 0.75
MAGNITUDE_THRESHOLD = 0.05
BOOTSTRAP = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    return (
        float(spearmanr(first[valid], second[valid]).statistic)
        if valid.sum() >= 3 else np.nan
    )


def load_response(model: str, directory: Path, suffix: str) -> dict[str, np.ndarray]:
    seed_curves = []
    gauge_ids = None
    for seed in SEEDS:
        if model == "author_dhbv":
            name = (
                f"{model}_seed{seed}_pet_climatology_float64_replay_curves.npz"
                if suffix else f"{model}_seed{seed}_float64_replay_curves.npz"
            )
            path = directory / name
        else:
            path = directory / f"{model}_seed{seed}{suffix}_intervention_curves.npz"
        archive = np.load(path)
        current_ids = np.asarray([str(item).zfill(8) for item in archive["gauge_ids"]])
        if gauge_ids is None:
            gauge_ids = current_ids
        elif not np.array_equal(gauge_ids, current_ids):
            raise RuntimeError(f"Gauge-order mismatch in {path}")
        q = archive["q_mm_day"].astype(np.float64)
        q0 = q[:, :, 0]
        valid = np.isfinite(q0) & (q0 >= 0.001)
        normalized = np.divide(
            q, q0[:, :, None], out=np.full_like(q, np.nan), where=valid[:, :, None]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            seed_curves.append(np.nanmedian(normalized, axis=1))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        curve = np.nanmedian(np.stack(seed_curves), axis=0)
    seed_r7 = np.stack([c[:, 7] for c in seed_curves])
    return {"gauge_ids": gauge_ids, "curve": curve, "r7": curve[:, 7], "r30": curve[:, 30], "seed_r7": seed_r7}


def stratified_gap_ci(difference: np.ndarray, regions: np.ndarray, key: str) -> tuple[float, float]:
    """Ecoregion-stratified bootstrap of the median absolute R7 gap (same design as script 16)."""
    groups = [np.flatnonzero(regions == item) for item in np.unique(regions)]
    if len(difference) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    gaps = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        gaps[i] = np.nanmedian(np.abs(difference[sample]))
    return float(np.nanquantile(gaps, 0.025)), float(np.nanquantile(gaps, 0.975))


def seed_reference(seed_r7: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Within-architecture reproducibility at the same gauges."""
    pairs = [np.nanmedian(np.abs(seed_r7[i][mask] - seed_r7[j][mask])) for i, j in itertools.combinations(range(seed_r7.shape[0]), 2)]
    loo = [np.nanmedian(np.abs(seed_r7[i][mask] - np.nanmedian(np.delete(seed_r7, i, axis=0), axis=0)[mask])) for i in range(seed_r7.shape[0])]
    onesided = [float(np.nanmean(seed_r7[i][mask] > seed_r7[j][mask])) for i, j in itertools.combinations(range(seed_r7.shape[0]), 2)]
    return {"max": float(np.max(pairs)), "mean": float(np.mean(pairs)), "loo": float(np.mean(loo)),
            "max_deviation_from_half": float(max(abs(x - 0.5) for x in onesided))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gauge = pd.read_csv(EVAL / "national_gauge_metrics.csv.gz", dtype={"GAGE_ID": str})
    gauge["GAGE_ID"] = gauge["GAGE_ID"].str.zfill(8)
    regions_all = gauge["ecoregion"].fillna("Unknown").astype(str).to_numpy() if "ecoregion" in gauge else np.full(len(gauge), "All")
    responses = {}
    for model in MODELS:
        zero = load_response(model, FROZEN, "")
        pet = load_response(model, PET, "_pet_climatology")
        if not np.array_equal(zero["gauge_ids"], pet["gauge_ids"]):
            raise RuntimeError(f"Zero-PET/PET gauge mismatch for {model}")
        if not np.array_equal(zero["gauge_ids"], gauge["GAGE_ID"].to_numpy(str)):
            raise RuntimeError(f"Evaluation/intervention gauge mismatch for {model}")
        responses[(model, "zero")] = zero
        responses[(model, "climatology")] = pet

    model_rows = []
    for model in MODELS:
        zero = responses[(model, "zero")]
        pet = responses[(model, "climatology")]
        for threshold in THRESHOLDS:
            skill = gauge[f"{model}_kge"].to_numpy(float)
            mask = (
                np.isfinite(skill) & (skill >= threshold)
                & np.isfinite(zero["r7"]) & np.isfinite(pet["r7"])
            )
            model_rows.append({
                "model": model,
                "skill_threshold": threshold,
                "qualified_gauges": int(mask.sum()),
                "zero_pet_median_r7": float(np.nanmedian(zero["r7"][mask])),
                "climatology_pet_median_r7": float(np.nanmedian(pet["r7"][mask])),
                "zero_vs_climatology_r7_spearman": rho(zero["r7"][mask], pet["r7"][mask]),
                "median_absolute_r7_change": float(np.nanmedian(np.abs(pet["r7"][mask] - zero["r7"][mask]))),
                "median_signed_r7_change_climatology_minus_zero": float(np.nanmedian(pet["r7"][mask] - zero["r7"][mask])),
                "median_absolute_r30_change": float(np.nanmedian(np.abs(pet["r30"][mask] - zero["r30"][mask]))),
            })
    model_frame = pd.DataFrame(model_rows)
    model_frame.to_csv(OUT / "pet_climatology_model_summary.csv", index=False)

    pair_rows = []
    for first, second in itertools.combinations(MODELS, 2):
        first_skill = gauge[f"{first}_kge"].to_numpy(float)
        second_skill = gauge[f"{second}_kge"].to_numpy(float)
        for threshold in THRESHOLDS:
            base_mask = (
                np.isfinite(first_skill) & np.isfinite(second_skill)
                & (first_skill >= threshold) & (second_skill >= threshold)
            )
            for pet_mode in ("zero", "climatology"):
                a = responses[(first, pet_mode)]
                b = responses[(second, pet_mode)]
                mask = base_mask & np.isfinite(a["r7"]) & np.isfinite(b["r7"])
                difference = a["r7"][mask] - b["r7"][mask]
                curve_gap = np.abs(a["curve"][mask, :31] - b["curve"][mask, :31])
                r7_rho = rho(a["r7"][mask], b["r7"][mask])
                r7_gap = float(np.nanmedian(np.abs(difference))) if mask.any() else np.nan
                ref_a = seed_reference(a["seed_r7"], mask) if mask.sum() >= 3 else None
                ref_b = seed_reference(b["seed_r7"], mask) if mask.sum() >= 3 else None
                seed_max = max(ref_a["max"], ref_b["max"]) if ref_a else np.nan
                seed_loo = max(ref_a["loo"], ref_b["loo"]) if ref_a else np.nan
                gap_lo, gap_hi = stratified_gap_ci(difference, regions_all[mask], f"{first}|{second}|{threshold:.2f}|{pet_mode}")
                pair_rows.append({
                    "first": first,
                    "second": second,
                    "skill_threshold": threshold,
                    "pet_mode": pet_mode,
                    "mutually_qualified_gauges": int(mask.sum()),
                    "r7_spearman": r7_rho,
                    "median_absolute_r7_gap": r7_gap,
                    "median_signed_r7_difference_first_minus_second": (
                        float(np.nanmedian(difference)) if mask.any() else np.nan
                    ),
                    "r30_spearman": rho(a["r30"][mask], b["r30"][mask]),
                    "median_absolute_r30_gap": (
                        float(np.nanmedian(np.abs(a["r30"][mask] - b["r30"][mask])))
                        if mask.any() else np.nan
                    ),
                    "median_normalized_curve_gap_days_0_30": (
                        float(np.nanmedian(np.nanmean(curve_gap, axis=1)))
                        if mask.any() else np.nan
                    ),
                    "median_absolute_r7_gap_ci95_low": gap_lo,
                    "median_absolute_r7_gap_ci95_high": gap_hi,
                    "fraction_first_r7_higher": float(np.nanmean(difference > 0)) if mask.any() else np.nan,
                    "first_seed_pair_median_abs_r7_gap_max": ref_a["max"] if ref_a else np.nan,
                    "second_seed_pair_median_abs_r7_gap_max": ref_b["max"] if ref_b else np.nan,
                    "first_loo_seed_vs_median_gap_mean": ref_a["loo"] if ref_a else np.nan,
                    "second_loo_seed_vs_median_gap_mean": ref_b["loo"] if ref_b else np.nan,
                    "first_seed_fraction_higher_max_deviation_from_half": ref_a["max_deviation_from_half"] if ref_a else np.nan,
                    "second_seed_fraction_higher_max_deviation_from_half": ref_b["max_deviation_from_half"] if ref_b else np.nan,
                    "cross_over_within_seed_gap_ratio": r7_gap / seed_max if np.isfinite(seed_max) and seed_max > 0 else np.nan,
                    "cross_over_loo_seed_gap_ratio": r7_gap / seed_loo if np.isfinite(seed_loo) and seed_loo > 0 else np.nan,
                    "cross_gap_exceeds_within_seed_gap_ci": bool(np.isfinite(gap_lo) and np.isfinite(seed_max) and gap_lo > seed_max),
                    "material_disagreement": bool(
                        mask.sum() >= 10
                        and (r7_rho < RANK_THRESHOLD or r7_gap > MAGNITUDE_THRESHOLD)
                    ),
                })
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(OUT / "pet_climatology_pairwise_summary.csv", index=False)

    high = pair_frame[pair_frame["skill_threshold"].eq(0.60)]
    receipt = {
        "analysis": "PET-at-climatology intervention sensitivity",
        "pet_climatology_period": ["2003-10-05", "2018-12-31"],
        "target_discharge_used_for_pet": False,
        "architectures": list(MODELS),
        "seeds": list(SEEDS),
        "high_skill_zero_pet_disagreeing_pairs": int(
            high[high["pet_mode"].eq("zero")]["material_disagreement"].sum()
        ),
        "high_skill_climatology_pet_disagreeing_pairs": int(
            high[high["pet_mode"].eq("climatology")]["material_disagreement"].sum()
        ),
        "high_skill_zero_pet_pairs_exceeding_seed_noise": int(
            high[high["pet_mode"].eq("zero")]["cross_gap_exceeds_within_seed_gap_ci"].sum()
        ),
        "high_skill_climatology_pet_pairs_exceeding_seed_noise": int(
            high[high["pet_mode"].eq("climatology")]["cross_gap_exceeds_within_seed_gap_ci"].sum()
        ),
        "note": "material_disagreement uses the prespecified 0.05 / 0.75 rule only; the 0.05 rule is stricter than single-architecture seed noise, so the seed-noise counts are the defensible ones.",
        "high_skill_evaluable_pairs_per_mode": int(
            high[high["pet_mode"].eq("climatology")]["mutually_qualified_gauges"].ge(10).sum()
        ),
        "model_summary": "outputs/evaluation_pet_climatology/pet_climatology_model_summary.csv",
        "model_summary_sha256": sha256(OUT / "pet_climatology_model_summary.csv"),
        "pairwise_summary": "outputs/evaluation_pet_climatology/pet_climatology_pairwise_summary.csv",
        "pairwise_summary_sha256": sha256(OUT / "pet_climatology_pairwise_summary.csv"),
    }
    (OUT / "pet_climatology_sensitivity_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
