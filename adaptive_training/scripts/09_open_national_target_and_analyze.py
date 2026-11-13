#!/usr/bin/env python3
"""Verify every frozen receipt, then open and analyze the national targets once."""

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
OUTPUT = ROOT / "outputs/evaluation"
SEEDS = (11, 29, 47)
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
RANK_AGREEMENT_THRESHOLD = 0.75
MAGNITUDE_AGREEMENT_THRESHOLD = 0.05
MINIMUM_PAIR_GAUGES = 10
MINIMUM_HIGH_SKILL_GAUGES = 30
BOOTSTRAP_ITERATIONS = 2000
TARGET_GAUGES = 3982
TARGET_DAYS = 2192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    obs, sim = observed[valid], predicted[valid]
    if np.std(obs, ddof=1) <= 0 or np.mean(obs) <= 0:
        return np.nan
    correlation = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt(
        (correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
    ))


def rho(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(first[valid], second[valid]).statistic)


def persistence_class(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values)
    result[valid] = np.digitize(
        values[valid], [0.50, np.sqrt(0.50)]
    ).astype(np.int8)
    return result


def median_hydrograph_correlation(
    first: np.ndarray, second: np.ndarray, mask: np.ndarray
) -> float:
    values = []
    for index in np.flatnonzero(mask):
        valid = np.isfinite(first[index]) & np.isfinite(second[index])
        if (
            valid.sum() >= 365
            and np.std(first[index, valid]) > 0
            and np.std(second[index, valid]) > 0
        ):
            values.append(np.corrcoef(first[index, valid], second[index, valid])[0, 1])
    return float(np.nanmedian(values)) if values else np.nan


def stratified_uncertainty(
    first: np.ndarray,
    second: np.ndarray,
    regions: np.ndarray,
    mask: np.ndarray,
    seed_key: str,
) -> dict[str, float]:
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        return {
            "r7_spearman_ci95_low": np.nan,
            "r7_spearman_ci95_high": np.nan,
            "median_absolute_r7_gap_ci95_low": np.nan,
            "median_absolute_r7_gap_ci95_high": np.nan,
        }
    groups = [indices[regions[indices] == item] for item in np.unique(regions[indices])]
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    rank_values = np.full(BOOTSTRAP_ITERATIONS, np.nan)
    gap_values = np.full(BOOTSTRAP_ITERATIONS, np.nan)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sample = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        rank_values[iteration] = rho(first[sample], second[sample])
        gap_values[iteration] = np.nanmedian(np.abs(first[sample] - second[sample]))
    return {
        "r7_spearman_ci95_low": float(np.nanquantile(rank_values, 0.025)),
        "r7_spearman_ci95_high": float(np.nanquantile(rank_values, 0.975)),
        "median_absolute_r7_gap_ci95_low": float(np.nanquantile(gap_values, 0.025)),
        "median_absolute_r7_gap_ci95_high": float(np.nanquantile(gap_values, 0.975)),
    }


def verify_frozen_outputs() -> dict:
    plan = ROOT / "provenance/preopening_analysis_plan.json"
    anchor_receipt_path = ROOT / "provenance/forcing_only_anchor_receipt.json"
    anchor_receipt = json.loads(anchor_receipt_path.read_text())
    anchor_path = ROOT / anchor_receipt["archive"]
    if anchor_receipt.get("uses_target_discharge") is not False:
        raise RuntimeError("Forcing-anchor receipt does not preserve target blinding")
    if anchor_receipt.get("uses_model_predictions_or_states") is not False:
        raise RuntimeError("Forcing-anchor receipt depends on model outcomes")
    anchor_hash = sha256(anchor_path)
    if anchor_hash != anchor_receipt["archive_sha256"]:
        raise RuntimeError("Forcing-anchor archive hash mismatch")
    parity_paths = {
        "author_source_parity": ROOT / "provenance/source_parity/author_source_parity.json",
        "ported_model_equivalence": ROOT / "provenance/source_parity/ported_model_equivalence.json",
    }
    amendment_paths = (
        ROOT / "provenance/protocol_amendment_forcing_only_anchors.json",
        ROOT / "provenance/protocol_amendment_mcp_stable_partition.json",
        ROOT / "provenance/protocol_amendment_batched_evaluator.json",
    )
    for amendment_path in amendment_paths:
        amendment = json.loads(amendment_path.read_text())
        if amendment.get("target_discharge_accessed") is not False:
            raise RuntimeError(f"Protocol amendment was not outcome-blind: {amendment_path}")
    stable_mcp_validation_path = ROOT / "provenance/stable_mcp_partition_validation.json"
    stable_mcp_validation = json.loads(stable_mcp_validation_path.read_text())
    if stable_mcp_validation.get("passes") is not True:
        raise RuntimeError("Overflow-safe MCP partition did not pass source parity")
    for label, parity_path in parity_paths.items():
        parity = json.loads(parity_path.read_text())
        if parity.get("all_pass") is not True:
            raise RuntimeError(f"Inherited parity evidence failed: {label}")
    registry: dict = {
        "national_target_discharge_accessed": False,
        "analysis_plan": {
            "path": str(plan.relative_to(ROOT)),
            "sha256": sha256(plan),
        },
        "analysis_code": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "forcing_anchor_archive": {
            "path": str(anchor_path.relative_to(ROOT)),
            "sha256": anchor_hash,
            "receipt": str(anchor_receipt_path.relative_to(ROOT)),
            "uses_target_discharge": False,
            "uses_model_predictions_or_states": False,
        },
        "inherited_parity_evidence": {
            label: {"path": str(path), "sha256": sha256(path), "all_pass": True}
            for label, path in parity_paths.items()
        },
        "protocol_amendments": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "target_discharge_accessed": False,
            }
            for path in amendment_paths
        ],
        "stable_mcp_partition_validation": {
            "path": str(stable_mcp_validation_path.relative_to(ROOT)),
            "sha256": sha256(stable_mcp_validation_path),
            "passes": True,
        },
        "files": [],
        "closure_eligibility_all_three_seeds": {},
    }
    frozen = ROOT / "outputs/frozen"
    for model in MODELS:
        closure_passes = []
        for seed in SEEDS:
            prediction_receipt_path = frozen / f"{model}_seed{seed}_prediction_receipt.json"
            prediction_receipt = json.loads(prediction_receipt_path.read_text())
            if prediction_receipt.get("target_discharge_accessed") is not False:
                raise RuntimeError(f"Broken target blind: {prediction_receipt_path}")
            for key in ("framework_evaluator_result", "frozen_prediction_archive"):
                path = ROOT / prediction_receipt[key]
                actual = sha256(path)
                if actual != prediction_receipt[f"{key}_sha256"]:
                    raise RuntimeError(f"Hash mismatch: {path}")
                registry["files"].append({"path": str(path), "sha256": actual})

            intervention_receipt_path = frozen / f"{model}_seed{seed}_intervention_receipt.json"
            intervention_receipt = json.loads(intervention_receipt_path.read_text())
            if intervention_receipt.get("target_discharge_accessed") is not False:
                raise RuntimeError(f"Broken intervention blind: {intervention_receipt_path}")
            if intervention_receipt.get("forcing_anchor_archive_sha256") != anchor_hash:
                raise RuntimeError(f"Intervention used the wrong forcing anchors: {intervention_receipt_path}")
            for key in ("curves", "fingerprints"):
                path = ROOT / intervention_receipt[key]
                actual = sha256(path)
                if actual != intervention_receipt[f"{key}_sha256"]:
                    raise RuntimeError(f"Hash mismatch: {path}")
                registry["files"].append({"path": str(path), "sha256": actual})
            closure_passes.append(bool(intervention_receipt["passes_closure"]))
        registry["closure_eligibility_all_three_seeds"][model] = bool(all(closure_passes))

    path = ROOT / "provenance/preopening_freeze_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n")
    digest = sha256(path)
    (ROOT / "provenance/preopening_freeze_registry.sha256").write_text(digest + "\n")
    return registry


def load_outputs(
    model: str, gauges: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = ROOT / "outputs/frozen"
    predictions, fingerprints, curves = [], [], []
    for seed in SEEDS:
        archive = np.load(frozen / f"{model}_seed{seed}_stock_predictions.npz")
        if not np.array_equal(archive["gauge_ids"].astype(str), np.asarray(gauges)):
            raise RuntimeError(f"Prediction gauge-order mismatch: {model} seed {seed}")
        predictions.append(archive["q_sim_mm_day"].astype(np.float64))

        frame = pd.read_csv(
            frozen / f"{model}_seed{seed}_fingerprints.csv", dtype={"GAGE_ID": str}
        )
        if frame.GAGE_ID.tolist() != gauges:
            raise RuntimeError(f"Fingerprint gauge-order mismatch: {model} seed {seed}")
        fingerprints.append(frame.median_r7.to_numpy(float))

        curve_archive = np.load(frozen / f"{model}_seed{seed}_intervention_curves.npz")
        if not np.array_equal(curve_archive["gauge_ids"].astype(str), np.asarray(gauges)):
            raise RuntimeError(f"Curve gauge-order mismatch: {model} seed {seed}")
        q = curve_archive["q_mm_day"].astype(np.float64)
        q0 = q[:, :, 0]
        valid = np.isfinite(q0) & (q0 >= 0.001)
        normalized = np.divide(
            q,
            q0[:, :, None],
            out=np.full_like(q, np.nan),
            where=valid[:, :, None],
        )
        curves.append(np.nanmedian(normalized, axis=1))

    return (
        np.mean(np.stack(predictions), axis=0),
        np.nanmedian(np.stack(fingerprints), axis=0),
        np.nanmedian(np.stack(curves), axis=0),
    )


def main() -> None:
    registry = verify_frozen_outputs()
    eligible_models = tuple(
        model for model in MODELS
        if registry["closure_eligibility_all_three_seeds"][model]
    )
    ineligible_models = tuple(model for model in MODELS if model not in eligible_models)
    if len(eligible_models) < 2:
        raise RuntimeError("Fewer than two lineages passed all three frozen closure audits")

    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} targets, found {len(gauges)}")

    # This is the only target-streamflow opening line in the national experiment.
    target = np.load(SOURCE_INPUTS / "evaluation/q_target_2019_2024.npy").astype(np.float64)
    if target.shape != (TARGET_GAUGES, TARGET_DAYS):
        raise RuntimeError(f"Unexpected target-discharge shape: {target.shape}")
    station = pd.read_csv(
        SOURCE_INPUTS / "processed/station_index.csv", dtype={"GAGE_ID": str}
    )
    target_station = station.loc[station.role.eq("target")].reset_index(drop=True)
    if target_station.GAGE_ID.tolist() != gauges:
        raise RuntimeError("Target-discharge order mismatch")

    predictions, fingerprints, curves, kges = {}, {}, {}, {}
    gauge_frame = target_station.copy()
    for model in MODELS:
        prediction, fingerprint, curve = load_outputs(model, gauges)
        predictions[model] = prediction
        fingerprints[model] = fingerprint
        curves[model] = curve
        kges[model] = np.asarray([
            kge(target[index], prediction[index]) for index in range(TARGET_GAUGES)
        ])
        gauge_frame[f"{model}_kge"] = kges[model]
        gauge_frame[f"{model}_r7"] = fingerprints[model]

    observed = pd.read_parquet(
        SOURCE_INPUTS / "observed_recessions/recession_summary.parquet"
    )
    observed["GAGE_ID"] = observed.GAGE_ID.astype(str).str.zfill(8)
    observed = observed.loc[observed.scenario.eq("primary")].set_index("GAGE_ID").reindex(gauges)
    gauge_frame["observed_r7"] = observed.observed_median_r7.to_numpy(float)
    gauge_frame["observed_usable"] = observed.usable.fillna(False).to_numpy(bool)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    gauge_frame.to_csv(OUTPUT / "national_gauge_metrics.csv.gz", index=False)

    architecture = pd.DataFrame([
        {
            "model": model,
            "closure_eligible": registry["closure_eligibility_all_three_seeds"][model],
            "finite_kge_gauges": int(np.isfinite(kges[model]).sum()),
            "median_target_kge": float(np.nanmedian(kges[model])),
            "fraction_kge_ge_0_30": float(np.nanmean(kges[model] >= 0.30)),
            "fraction_kge_ge_0_60": float(np.nanmean(kges[model] >= 0.60)),
            "finite_r7_gauges": int(np.isfinite(fingerprints[model]).sum()),
            "median_r7": float(np.nanmedian(fingerprints[model])),
        }
        for model in MODELS
    ])
    architecture.to_csv(OUTPUT / "national_architecture_summary.csv", index=False)

    regions = gauge_frame.ecoregion.fillna("Unknown").astype(str).to_numpy()
    observed_values = gauge_frame.observed_r7.to_numpy(float)
    observed_usable = gauge_frame.observed_usable.to_numpy(bool)
    records = []
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
            n = int(mask.sum())
            first_class = persistence_class(fingerprints[first])
            second_class = persistence_class(fingerprints[second])
            observed_mask = mask & observed_usable & np.isfinite(observed_values)
            curve_gap = (
                float(np.nanmedian(np.nanmean(np.abs(
                    curves[first][mask, :31] - curves[second][mask, :31]
                ), axis=1)))
                if n else np.nan
            )
            records.append({
                "first": first,
                "second": second,
                "skill_threshold": threshold,
                "mutually_qualified_gauges": n,
                "median_hydrograph_correlation": median_hydrograph_correlation(
                    predictions[first], predictions[second], mask
                ),
                "r7_spearman": rho(fingerprints[first][mask], fingerprints[second][mask]),
                "median_absolute_r7_gap": (
                    float(np.nanmedian(np.abs(
                        fingerprints[first][mask] - fingerprints[second][mask]
                    ))) if n else np.nan
                ),
                "median_normalized_curve_gap_days_0_30": curve_gap,
                "same_persistence_class_fraction": (
                    float(np.mean(first_class[mask] == second_class[mask])) if n else np.nan
                ),
                "observed_usable_gauges": int(observed_mask.sum()),
                "first_observed_rho": rho(
                    fingerprints[first][observed_mask], observed_values[observed_mask]
                ),
                "second_observed_rho": rho(
                    fingerprints[second][observed_mask], observed_values[observed_mask]
                ),
                "first_observed_median_absolute_error": (
                    float(np.nanmedian(np.abs(
                        fingerprints[first][observed_mask] - observed_values[observed_mask]
                    ))) if observed_mask.any() else np.nan
                ),
                "second_observed_median_absolute_error": (
                    float(np.nanmedian(np.abs(
                        fingerprints[second][observed_mask] - observed_values[observed_mask]
                    ))) if observed_mask.any() else np.nan
                ),
                **stratified_uncertainty(
                    fingerprints[first], fingerprints[second], regions, mask,
                    f"{first}|{second}|{threshold:.2f}",
                ),
            })

    pairwise = pd.DataFrame(records)
    pairwise["evaluable_pair"] = (
        pairwise.mutually_qualified_gauges.ge(MINIMUM_PAIR_GAUGES)
        & np.isfinite(pairwise.r7_spearman)
    )
    pairwise["material_disagreement"] = (
        pairwise.evaluable_pair
        & (
            pairwise.r7_spearman.lt(RANK_AGREEMENT_THRESHOLD)
            | pairwise.median_absolute_r7_gap.gt(MAGNITUDE_AGREEMENT_THRESHOLD)
        )
    )
    pairwise["high_skill_convergence"] = (
        pairwise.skill_threshold.eq(0.60)
        & pairwise.mutually_qualified_gauges.ge(MINIMUM_HIGH_SKILL_GAUGES)
        & pairwise.r7_spearman.ge(RANK_AGREEMENT_THRESHOLD)
        & pairwise.median_absolute_r7_gap.le(MAGNITUDE_AGREEMENT_THRESHOLD)
    )
    pairwise.to_csv(OUTPUT / "national_pairwise_skill_conditioned_summary.csv", index=False)

    threshold30 = pairwise.loc[pairwise.skill_threshold.eq(0.30)]
    threshold60 = pairwise.loc[pairwise.skill_threshold.eq(0.60)]
    discovery = {
        "population": "3,982 unseen gauges after a 500/200 donor/development split",
        "gauges_with_valid_forcing_anchors": int(
            gauge_frame[[f"{model}_r7" for model in eligible_models]].notna().any(axis=1).sum()
        ),
        "target_discharge_opened_after_freeze": True,
        "preopening_registry": "provenance/preopening_freeze_registry.json",
        "preopening_registry_sha256": sha256(
            ROOT / "provenance/preopening_freeze_registry.json"
        ),
        "closure_eligible_models": list(eligible_models),
        "closure_ineligible_models": list(ineligible_models),
        "evaluable_pairs_at_kge_0_30": int(threshold30.evaluable_pair.sum()),
        "materially_disagreeing_pairs_at_kge_0_30": int(
            threshold30.material_disagreement.sum()
        ),
        "high_skill_evaluable_pairs_at_kge_0_60": int(
            threshold60.mutually_qualified_gauges.ge(MINIMUM_HIGH_SKILL_GAUGES).sum()
        ),
        "convergent_pairs_at_kge_0_60": threshold60.loc[
            threshold60.high_skill_convergence, ["first", "second"]
        ].to_dict("records"),
        "central_non_guarantee_supported": bool(threshold30.material_disagreement.any()),
        "interpretation": (
            "A valid skill-conditioned counterexample supports only the narrow "
            "non-guarantee claim. Convergent pairs are retained and reported."
        ),
    }
    (OUTPUT / "national_framework_discovery.json").write_text(
        json.dumps(discovery, indent=2) + "\n"
    )
    print(architecture.to_string(index=False))
    print("\nKGE >= 0.30\n", threshold30.to_string(index=False))
    print("\nKGE >= 0.60\n", threshold60.to_string(index=False))
    print("\n", json.dumps(discovery, indent=2))


if __name__ == "__main__":
    main()
