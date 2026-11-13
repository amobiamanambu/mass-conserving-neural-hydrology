#!/usr/bin/env python3
"""Independently recompute the final four-model national result.

This audit reads the frozen prediction and intervention archives rather than
the manuscript source tables.  It verifies checkpoint hashes, numerical
closure eligibility, target KGE, R7/R30 fingerprints, the principal pairwise
statistics, and the PET-at-climatology receipt.
"""

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
SOURCE_INPUTS = ROOT / "data/source_inputs"
FROZEN = ROOT / "outputs/frozen"
EVALUATION = ROOT / "outputs/evaluation"
AMENDED = ROOT / "outputs/evaluation_amended"
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
CLASS_BREAKS = (0.50, np.sqrt(0.50))
ATOL = 1e-10


def nanmedian_quiet(values: np.ndarray, axis: int) -> np.ndarray:
    """Take a NaN-aware median while preserving intentionally undefined gauges."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(simulated)
    if valid.sum() < 365:
        return np.nan
    obs, sim = observed[valid], simulated[valid]
    if np.mean(obs) <= 0 or np.std(obs, ddof=1) <= 0 or np.std(sim, ddof=1) <= 0:
        return np.nan
    correlation = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def rho(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.std(first[valid]) == 0 or np.std(second[valid]) == 0:
        return np.nan
    return float(spearmanr(first[valid], second[valid]).statistic)


def classes(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    result[valid] = np.digitize(values[valid], CLASS_BREAKS)
    return result


def same(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, atol=ATOL, rtol=0.0, equal_nan=True))


def main() -> None:
    findings: list[str] = []
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    target = np.load(SOURCE_INPUTS / "evaluation/q_target_2019_2024.npy").astype(np.float64)
    reported = pd.read_csv(
        AMENDED / "amended_gauge_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    reported["GAGE_ID"] = reported.GAGE_ID.str.zfill(8)
    if reported.GAGE_ID.tolist() != gauges:
        raise RuntimeError("Reported gauge order differs from test_3982.txt")

    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    eligibility = json.loads((AMENDED / "amended_eligibility.json").read_text())
    for model in MODELS:
        if not eligibility[model]["admitted_under_float64_audit"]:
            findings.append(f"model not admitted by closure audit: {model}")
        for seed in SEEDS:
            item = selection["architectures"][model]["runs"][str(seed)]
            checkpoint = ROOT / item["checkpoint"]
            if not checkpoint.exists() or sha256(checkpoint) != item["checkpoint_sha256"]:
                findings.append(f"checkpoint hash mismatch: {model} seed {seed}")
            if model == "author_dhbv":
                receipt = json.loads(
                    (FROZEN / f"{model}_seed{seed}_float64_replay_receipt.json").read_text()
                )
                if not receipt.get("passes_prespecified_tolerance_in_float64", False):
                    findings.append(f"float64 closure failed: {model} seed {seed}")
            else:
                receipt = json.loads(
                    (FROZEN / f"{model}_seed{seed}_intervention_receipt.json").read_text()
                )
                if not receipt.get("passes_closure", False):
                    findings.append(f"closure failed: {model} seed {seed}")

    model_data: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        predictions, seed_r7, seed_r30, seed_curves = [], [], [], []
        for seed in SEEDS:
            prediction = np.load(FROZEN / f"{model}_seed{seed}_stock_predictions.npz")
            if prediction["gauge_ids"].astype(str).tolist() != gauges:
                findings.append(f"prediction gauge mismatch: {model} seed {seed}")
            predictions.append(prediction["q_sim_mm_day"].astype(np.float64))
            fingerprint = pd.read_csv(
                FROZEN / f"{model}_seed{seed}_fingerprints.csv", dtype={"GAGE_ID": str}
            )
            curve_path = FROZEN / f"{model}_seed{seed}_intervention_curves.npz"
            if model == "author_dhbv":
                replay = FROZEN / f"{model}_seed{seed}_float64_replay_curves.npz"
                if replay.exists():
                    curve_path = replay
            curve_archive = np.load(curve_path)
            q = curve_archive["q_mm_day"].astype(np.float64)
            q0 = q[:, :, 0]
            valid = np.isfinite(q0) & (q0 >= 0.001)
            normalized = np.divide(
                q, q0[:, :, None], out=np.full_like(q, np.nan), where=valid[:, :, None]
            )
            seed_r7.append(nanmedian_quiet(np.where(valid, normalized[:, :, 7], np.nan), axis=1))
            seed_r30.append(nanmedian_quiet(np.where(valid, normalized[:, :, 30], np.nan), axis=1))
            seed_curves.append(nanmedian_quiet(normalized, axis=1))

        ensemble = np.mean(np.stack(predictions), axis=0)
        recomputed_kge = np.asarray([kge(target[i], ensemble[i]) for i in range(len(gauges))])
        r7 = nanmedian_quiet(np.stack(seed_r7), axis=0)
        r30 = nanmedian_quiet(np.stack(seed_r30), axis=0)
        curve = nanmedian_quiet(np.stack(seed_curves), axis=0)
        for label, actual, expected in (
            ("KGE", recomputed_kge, reported[f"{model}_kge"].to_numpy(float)),
            ("R7", r7, reported[f"{model}_r7"].to_numpy(float)),
            ("R30", r30, reported[f"{model}_r30"].to_numpy(float)),
        ):
            difference = float(np.nanmax(np.abs(actual - expected)))
            if difference > ATOL:
                findings.append(f"{label} mismatch {model}: {difference:.3e}")
        model_data[model] = {"kge": recomputed_kge, "r7": r7, "r30": r30, "curve": curve}

    pairwise = pd.read_csv(AMENDED / "amended_pairwise_summary.csv")
    rows_checked = 0
    for first, second in itertools.combinations(MODELS, 2):
        for threshold in THRESHOLDS:
            row = pairwise.loc[
                pairwise["first"].eq(first)
                & pairwise["second"].eq(second)
                & np.isclose(pairwise["skill_threshold"], threshold)
            ]
            if len(row) != 1:
                findings.append(f"missing pairwise row: {first}, {second}, {threshold}")
                continue
            row = row.iloc[0]
            a, b = model_data[first], model_data[second]
            mask = (
                np.isfinite(a["kge"]) & np.isfinite(b["kge"])
                & (a["kge"] >= threshold) & (b["kge"] >= threshold)
                & np.isfinite(a["r7"]) & np.isfinite(b["r7"])
            )
            delta = a["r7"][mask] - b["r7"][mask]
            curve_gap = np.abs(a["curve"][mask, :31] - b["curve"][mask, :31])
            checks = {
                "mutually_qualified_gauges": int(mask.sum()),
                "r7_spearman": rho(a["r7"][mask], b["r7"][mask]),
                "median_absolute_r7_gap": float(np.nanmedian(np.abs(delta))) if mask.any() else np.nan,
                "median_signed_r7_difference_first_minus_second": (
                    float(np.nanmedian(delta)) if mask.any() else np.nan
                ),
                "fraction_first_r7_higher": float(np.mean(delta > 0)) if mask.any() else np.nan,
                "median_normalized_curve_gap_days_0_30": (
                    float(np.nanmedian(np.nanmean(curve_gap, axis=1))) if mask.any() else np.nan
                ),
                "same_persistence_class_fraction": (
                    float(np.mean(classes(a["r7"][mask]) == classes(b["r7"][mask])))
                    if mask.any() else np.nan
                ),
            }
            for name, expected in checks.items():
                actual = row[name]
                passed = int(actual) == int(expected) if name == "mutually_qualified_gauges" else same(float(actual), float(expected))
                if not passed:
                    findings.append(
                        f"pairwise mismatch {first}/{second}/{threshold:.2f}/{name}: "
                        f"reported={actual}, recomputed={expected}"
                    )
            rows_checked += 1

    pet_receipt = ROOT / "outputs/evaluation_pet_climatology/pet_climatology_sensitivity_receipt.json"
    if not pet_receipt.exists():
        findings.append("PET-at-climatology receipt is missing")
    else:
        pet = json.loads(pet_receipt.read_text())
        for file_key, hash_key in (
            ("model_summary", "model_summary_sha256"),
            ("pairwise_summary", "pairwise_summary_sha256"),
        ):
            path = ROOT / pet[file_key]
            if not path.exists() or sha256(path) != pet[hash_key]:
                findings.append(f"PET sensitivity hash mismatch: {path}")
        pet_curves = ROOT / "outputs/sensitivity_pet_climatology"
        for model in MODELS:
            for seed in SEEDS:
                receipt_path = pet_curves / (
                    f"{model}_seed{seed}_pet_climatology_float64_replay_receipt.json"
                    if model == "author_dhbv"
                    else f"{model}_seed{seed}_pet_climatology_intervention_receipt.json"
                )
                if not receipt_path.exists():
                    findings.append(f"PET intervention receipt missing: {model} seed {seed}")
                    continue
                receipt = json.loads(receipt_path.read_text())
                passed = (
                    receipt.get("passes_prespecified_tolerance_in_float64", False)
                    if model == "author_dhbv" else receipt.get("passes_closure", False)
                )
                if not passed:
                    findings.append(f"PET closure failed: {model} seed {seed}")
                curve_path = ROOT / receipt["curves"]
                if not curve_path.exists() or sha256(curve_path) != receipt["curves_sha256"]:
                    findings.append(f"PET curve hash mismatch: {model} seed {seed}")

    audit = {
        "status": "pass" if not findings else "fail",
        "target_gauges": len(gauges),
        "architectures": list(MODELS),
        "seeds": list(SEEDS),
        "pair_rows_checked": rows_checked,
        "numeric_absolute_tolerance": ATOL,
        "findings": findings,
    }
    path = AMENDED / "independent_amended_result_audit.json"
    path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if findings:
        raise RuntimeError("Final amended national audit failed")


if __name__ == "__main__":
    main()
