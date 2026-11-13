#!/usr/bin/env python3
"""Independently recompute the national headline metrics from frozen arrays."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/parquet_runtime"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


SOURCE_INPUTS = ROOT / "data" / "source_inputs"
FROZEN = ROOT / "outputs/frozen"
OUTPUT = ROOT / "outputs/evaluation"
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)


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
    if np.std(obs, ddof=1) <= 0 or np.mean(obs) <= 0:
        return np.nan
    correlation = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rho(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    return float(spearmanr(first[valid], second[valid]).statistic) if valid.sum() >= 3 else np.nan


def classes(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid] = np.digitize(values[valid], [0.50, np.sqrt(0.50)]).astype(np.int8)
    return result


def check_close(label: str, actual: float, expected: float, tolerance: float = 1e-10) -> float:
    if np.isnan(actual) and np.isnan(expected):
        return 0.0
    difference = abs(actual - expected)
    if not np.isfinite(difference) or difference > tolerance:
        raise RuntimeError(f"{label}: {actual} != {expected} (difference {difference})")
    return float(difference)


def main() -> None:
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != 3982:
        raise RuntimeError("Frozen target list is not 3,982 gauges")
    target = np.load(SOURCE_INPUTS / "evaluation/q_target_2019_2024.npy").astype(np.float64)
    saved_gauges = pd.read_csv(
        OUTPUT / "national_gauge_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    saved_gauges["GAGE_ID"] = saved_gauges.GAGE_ID.str.zfill(8)
    if saved_gauges.GAGE_ID.tolist() != gauges:
        raise RuntimeError("Saved national metrics do not match the target order")

    anchor_receipt = json.loads(
        (ROOT / "provenance/forcing_only_anchor_receipt.json").read_text()
    )
    anchor_path = ROOT / anchor_receipt["archive"]
    anchors = np.load(anchor_path)
    if sha256(anchor_path) != anchor_receipt["archive_sha256"]:
        raise RuntimeError("Forcing-anchor archive hash mismatch")
    if anchor_receipt["uses_target_discharge"] or anchor_receipt["uses_model_predictions_or_states"]:
        raise RuntimeError("Anchor receipt does not establish outcome independence")
    if anchors["eval_time_index"].shape != (3982, 12):
        raise RuntimeError("Forcing-anchor array has the wrong shape")

    saved_architecture = pd.read_csv(OUTPUT / "national_architecture_summary.csv").set_index("model")
    saved_pairwise = pd.read_csv(OUTPUT / "national_pairwise_skill_conditioned_summary.csv")
    registry = json.loads((ROOT / "provenance/preopening_freeze_registry.json").read_text())
    eligible_models = tuple(
        model for model in MODELS
        if registry["closure_eligibility_all_three_seeds"][model]
    )
    predictions: dict[str, np.ndarray] = {}
    fingerprints: dict[str, np.ndarray] = {}
    curves: dict[str, np.ndarray] = {}
    kges: dict[str, np.ndarray] = {}
    maximum_metric_difference = 0.0
    closure = {}

    for model in MODELS:
        seed_predictions, seed_fingerprints, seed_curves = [], [], []
        model_receipts = []
        for seed in SEEDS:
            prediction_archive = np.load(FROZEN / f"{model}_seed{seed}_stock_predictions.npz")
            if prediction_archive["gauge_ids"].astype(str).tolist() != gauges:
                raise RuntimeError(f"Prediction gauge mismatch for {model} seed {seed}")
            seed_predictions.append(prediction_archive["q_sim_mm_day"].astype(np.float64))

            fingerprint = pd.read_csv(
                FROZEN / f"{model}_seed{seed}_fingerprints.csv", dtype={"GAGE_ID": str}
            )
            if fingerprint.GAGE_ID.str.zfill(8).tolist() != gauges:
                raise RuntimeError(f"Fingerprint gauge mismatch for {model} seed {seed}")
            seed_fingerprints.append(fingerprint.median_r7.to_numpy(float))

            curve_archive = np.load(FROZEN / f"{model}_seed{seed}_intervention_curves.npz")
            q = curve_archive["q_mm_day"].astype(np.float64)
            q0 = q[:, :, 0]
            valid = np.isfinite(q0) & (q0 >= 0.001)
            normalized = np.divide(
                q, q0[:, :, None], out=np.full_like(q, np.nan), where=valid[:, :, None]
            )
            seed_curves.append(np.nanmedian(normalized, axis=1))

            receipt_path = FROZEN / f"{model}_seed{seed}_intervention_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("forcing_anchor_archive_sha256") != anchor_receipt["archive_sha256"]:
                raise RuntimeError(f"Wrong anchor archive for {model} seed {seed}")
            model_receipts.append(receipt)

        predictions[model] = np.mean(np.stack(seed_predictions), axis=0)
        fingerprints[model] = np.nanmedian(np.stack(seed_fingerprints), axis=0)
        curves[model] = np.nanmedian(np.stack(seed_curves), axis=0)
        kges[model] = np.asarray([
            kge(target[index], predictions[model][index]) for index in range(len(gauges))
        ])

        saved_kge = saved_gauges[f"{model}_kge"].to_numpy(float)
        saved_r7 = saved_gauges[f"{model}_r7"].to_numpy(float)
        maximum_metric_difference = max(
            maximum_metric_difference,
            float(np.nanmax(np.abs(kges[model] - saved_kge))),
            float(np.nanmax(np.abs(fingerprints[model] - saved_r7))),
        )
        row = saved_architecture.loc[model]
        maximum_metric_difference = max(
            maximum_metric_difference,
            check_close(f"{model} median KGE", float(np.nanmedian(kges[model])), row.median_target_kge),
            check_close(
                f"{model} fraction KGE>=0.30",
                float(np.mean(kges[model] >= 0.30)),
                row.fraction_kge_ge_0_30,
            ),
            check_close(
                f"{model} fraction KGE>=0.60",
                float(np.mean(kges[model] >= 0.60)),
                row.fraction_kge_ge_0_60,
            ),
        )
        all_three_seeds_pass = all(
            receipt.get("passes_closure") is True for receipt in model_receipts
        )
        if all_three_seeds_pass != bool(saved_architecture.loc[model].closure_eligible):
            raise RuntimeError(f"Closure eligibility mismatch for {model}")
        closure[model] = {
            "all_three_seeds_pass": all_three_seeds_pass,
            "receipt_sha256": {
                str(seed): sha256(FROZEN / f"{model}_seed{seed}_intervention_receipt.json")
                for seed in SEEDS
            },
        }

    if maximum_metric_difference > 1e-10:
        raise RuntimeError(f"Saved gauge metrics differ from raw arrays by {maximum_metric_difference}")

    pair_checks = []
    for first, second in itertools.combinations(eligible_models, 2):
        for threshold in THRESHOLDS:
            mask = (
                np.isfinite(kges[first])
                & np.isfinite(kges[second])
                & (kges[first] >= threshold)
                & (kges[second] >= threshold)
                & np.isfinite(fingerprints[first])
                & np.isfinite(fingerprints[second])
            )
            saved = saved_pairwise.loc[
                saved_pairwise["first"].eq(first)
                & saved_pairwise["second"].eq(second)
                & np.isclose(saved_pairwise.skill_threshold, threshold)
            ]
            if len(saved) != 1:
                raise RuntimeError(f"Missing saved pair row for {first}, {second}, {threshold}")
            saved = saved.iloc[0]
            n = int(mask.sum())
            if n != int(saved.mutually_qualified_gauges):
                raise RuntimeError(f"Qualified-gauge count mismatch for {first}, {second}, {threshold}")
            rank = rho(fingerprints[first][mask], fingerprints[second][mask])
            gap = float(np.nanmedian(np.abs(fingerprints[first][mask] - fingerprints[second][mask])))
            same_class = float(np.mean(classes(fingerprints[first])[mask] == classes(fingerprints[second])[mask]))
            curve_gap = float(np.nanmedian(np.nanmean(
                np.abs(curves[first][mask, :31] - curves[second][mask, :31]), axis=1
            )))
            discrepancies = {
                "r7_spearman": check_close("pair rank", rank, saved.r7_spearman),
                "median_absolute_r7_gap": check_close("pair gap", gap, saved.median_absolute_r7_gap),
                "same_persistence_class_fraction": check_close(
                    "pair class", same_class, saved.same_persistence_class_fraction
                ),
                "median_normalized_curve_gap_days_0_30": check_close(
                    "pair curve", curve_gap, saved.median_normalized_curve_gap_days_0_30
                ),
            }
            maximum_metric_difference = max(maximum_metric_difference, *discrepancies.values())
            pair_checks.append({
                "first": first,
                "second": second,
                "skill_threshold": threshold,
                "qualified_gauges": n,
                "maximum_saved_metric_difference": max(discrepancies.values()),
            })

    audit = {
        "status": "pass",
        "target_gauges": len(gauges),
        "target_days": int(target.shape[1]),
        "forcing_anchors": {
            "gauges": int(anchors["eval_time_index"].shape[0]),
            "anchors_per_gauge": int(anchors["eval_time_index"].shape[1]),
            "archive_sha256": anchor_receipt["archive_sha256"],
            "uses_target_discharge": False,
            "uses_predictions_or_states": False,
        },
        "closure": closure,
        "maximum_recomputed_vs_saved_difference": maximum_metric_difference,
        "pair_rows_checked": len(pair_checks),
        "source_hashes": {
            "target_discharge": sha256(SOURCE_INPUTS / "evaluation/q_target_2019_2024.npy"),
            "gauge_metrics": sha256(OUTPUT / "national_gauge_metrics.csv.gz"),
            "architecture_summary": sha256(OUTPUT / "national_architecture_summary.csv"),
            "pairwise_summary": sha256(OUTPUT / "national_pairwise_skill_conditioned_summary.csv"),
        },
    }
    output_path = OUTPUT / "independent_final_audit.json"
    output_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
