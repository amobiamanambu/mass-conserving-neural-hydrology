#!/usr/bin/env python3
"""Run post-opening subgroup, seed, and connected-gauge diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/evaluation"
FROZEN = ROOT / "outputs/frozen"
SOURCE_INPUTS = ROOT / "data" / "source_inputs"
EDGES = SOURCE_INPUTS / "nldi/river_network_edges.csv"
REFERENCE = SOURCE_INPUTS / "reference_only"
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.30, 0.60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return np.nan
    return float(spearmanr(first[valid], second[valid]).statistic)


def median_absolute_difference(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    return float(np.nanmedian(np.abs(first[valid] - second[valid]))) if valid.any() else np.nan


def seed_stability(gauges: list[str]) -> pd.DataFrame:
    records = []
    for model in MODELS:
        values = []
        for seed in SEEDS:
            frame = pd.read_csv(
                FROZEN / f"{model}_seed{seed}_fingerprints.csv", dtype={"GAGE_ID": str}
            )
            if frame.GAGE_ID.tolist() != gauges:
                raise RuntimeError(f"Seed fingerprint order mismatch: {model} seed {seed}")
            values.append(frame.median_r7.to_numpy(float))
        for first_seed, second_seed in itertools.combinations(range(3), 2):
            records.append({
                "model": model,
                "first_seed": SEEDS[first_seed],
                "second_seed": SEEDS[second_seed],
                "finite_gauges": int(np.sum(
                    np.isfinite(values[first_seed]) & np.isfinite(values[second_seed])
                )),
                "r7_spearman": rho(values[first_seed], values[second_seed]),
                "median_absolute_r7_difference": median_absolute_difference(
                    values[first_seed], values[second_seed]
                ),
            })
    return pd.DataFrame(records)


def subgroup_pairs(frame: pd.DataFrame, eligible_models: tuple[str, ...]) -> pd.DataFrame:
    records = []
    groups = {
        "all_targets": np.ones(len(frame), dtype=bool),
        "reference_targets": frame.reference_status.eq("Reference").to_numpy(),
        "nonreference_targets": frame.reference_status.ne("Reference").to_numpy(),
        "tier1_targets": frame.tier_short.eq("Tier 1").to_numpy(),
        "tier2_targets": frame.tier_short.eq("Tier 2").to_numpy(),
    }
    for group_name, group_mask in groups.items():
        for first, second in itertools.combinations(eligible_models, 2):
            first_kge = frame[f"{first}_kge"].to_numpy(float)
            second_kge = frame[f"{second}_kge"].to_numpy(float)
            first_r7 = frame[f"{first}_r7"].to_numpy(float)
            second_r7 = frame[f"{second}_r7"].to_numpy(float)
            for threshold in THRESHOLDS:
                mask = (
                    group_mask
                    & np.isfinite(first_kge)
                    & np.isfinite(second_kge)
                    & (first_kge >= threshold)
                    & (second_kge >= threshold)
                    & np.isfinite(first_r7)
                    & np.isfinite(second_r7)
                )
                records.append({
                    "group": group_name,
                    "first": first,
                    "second": second,
                    "skill_threshold": threshold,
                    "gauges": int(mask.sum()),
                    "r7_spearman": rho(first_r7[mask], second_r7[mask]),
                    "median_absolute_r7_difference": median_absolute_difference(
                        first_r7[mask], second_r7[mask]
                    ),
                })
    result = pd.DataFrame(records)
    result["material_disagreement"] = (
        result.gauges.ge(10)
        & (
            result.r7_spearman.lt(0.75)
            | result.median_absolute_r7_difference.gt(0.05)
        )
    )
    result["high_skill_convergence"] = (
        result.skill_threshold.eq(0.60)
        & result.gauges.ge(30)
        & result.r7_spearman.ge(0.75)
        & result.median_absolute_r7_difference.le(0.05)
    )
    return result


def nested_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("GAGE_ID")
    edges = pd.read_csv(EDGES, dtype={"upstream_gage": str, "downstream_gage": str})
    edges["upstream_gage"] = edges.upstream_gage.astype(str).str.zfill(8)
    edges["downstream_gage"] = edges.downstream_gage.astype(str).str.zfill(8)
    edges = edges[
        edges.upstream_gage.isin(indexed.index)
        & edges.downstream_gage.isin(indexed.index)
    ].copy()
    edges["network_distance_km"] = pd.to_numeric(edges.network_distance_km, errors="coerce")
    records = []
    for model in MODELS:
        for threshold in THRESHOLDS:
            upstream = indexed.loc[edges.upstream_gage]
            downstream = indexed.loc[edges.downstream_gage]
            valid = (
                upstream[f"{model}_kge"].to_numpy(float) >= threshold
            ) & (
                downstream[f"{model}_kge"].to_numpy(float) >= threshold
            ) & upstream.observed_usable.to_numpy(bool) & downstream.observed_usable.to_numpy(bool)
            model_up = upstream[f"{model}_r7"].to_numpy(float)
            model_down = downstream[f"{model}_r7"].to_numpy(float)
            observed_up = upstream.observed_r7.to_numpy(float)
            observed_down = downstream.observed_r7.to_numpy(float)
            valid &= (
                np.isfinite(model_up)
                & np.isfinite(model_down)
                & np.isfinite(observed_up)
                & np.isfinite(observed_down)
            )
            model_contrast = model_down[valid] - model_up[valid]
            observed_contrast = observed_down[valid] - observed_up[valid]
            records.append({
                "model": model,
                "skill_threshold": threshold,
                "pairs": int(valid.sum()),
                "contrast_vs_observed_spearman": rho(model_contrast, observed_contrast),
                "median_absolute_contrast_error": median_absolute_difference(
                    model_contrast, observed_contrast
                ),
                "median_network_distance_km": float(
                    np.nanmedian(edges.network_distance_km.to_numpy(float)[valid])
                ) if valid.any() else np.nan,
            })
    return pd.DataFrame(records)


def model_observation(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    observed = frame.observed_r7.to_numpy(float)
    usable = frame.observed_usable.to_numpy(bool) & np.isfinite(observed)
    for model in MODELS:
        kge = frame[f"{model}_kge"].to_numpy(float)
        r7 = frame[f"{model}_r7"].to_numpy(float)
        for threshold in THRESHOLDS:
            mask = usable & np.isfinite(kge) & (kge >= threshold) & np.isfinite(r7)
            records.append({
                "model": model,
                "skill_threshold": threshold,
                "gauges": int(mask.sum()),
                "model_observed_r7_spearman": rho(r7[mask], observed[mask]),
                "model_observed_median_absolute_error": median_absolute_difference(
                    r7[mask], observed[mask]
                ),
            })
    return pd.DataFrame(records)


def main() -> None:
    gauge_path = OUTPUT / "national_gauge_metrics.csv.gz"
    pair_path = OUTPUT / "national_pairwise_skill_conditioned_summary.csv"
    discovery_path = OUTPUT / "national_framework_discovery.json"
    if not all(path.exists() for path in (gauge_path, pair_path, discovery_path)):
        raise FileNotFoundError("The frozen national opening analysis has not completed")
    frame = pd.read_csv(gauge_path, dtype={"GAGE_ID": str})
    frame["GAGE_ID"] = frame.GAGE_ID.astype(str).str.zfill(8)
    if len(frame) != 3982 or frame.GAGE_ID.nunique() != 3982:
        raise RuntimeError("National gauge table does not contain 3,982 unique targets")

    discovery = json.loads(discovery_path.read_text())
    eligible_models = tuple(discovery["closure_eligible_models"])
    seeds = seed_stability(frame.GAGE_ID.tolist())
    subgroups = subgroup_pairs(frame, eligible_models)
    nested = nested_pairs(frame)
    observation = model_observation(frame)
    seeds.to_csv(OUTPUT / "national_seed_stability_summary.csv", index=False)
    subgroups.to_csv(OUTPUT / "national_subgroup_pairwise_summary.csv", index=False)
    nested.to_csv(OUTPUT / "national_nested_pair_summary.csv", index=False)
    observation.to_csv(OUTPUT / "national_model_observation_summary.csv", index=False)

    pairwise = pd.read_csv(pair_path)
    reference_summary_path = REFERENCE / "combined_architecture_summary.csv"
    reference_pair_path = REFERENCE / "combined_pairwise_skill_conditioned_summary.csv"
    reference_summary = pd.read_csv(reference_summary_path)
    reference_pairs = pd.read_csv(reference_pair_path)
    ref30 = reference_pairs.loc[reference_pairs.skill_threshold.eq(0.30)]
    ref60 = reference_pairs.loc[reference_pairs.skill_threshold.eq(0.60)]
    nat30 = pairwise.loc[pairwise.skill_threshold.eq(0.30)]
    nat60 = pairwise.loc[pairwise.skill_threshold.eq(0.60)]

    source_files = [
        gauge_path,
        pair_path,
        discovery_path,
        ROOT / "data/forcing_only_anchors.npz",
        ROOT / "provenance/forcing_only_anchor_receipt.json",
        ROOT / "provenance/preopening_analysis_plan.json",
        ROOT / "provenance/protocol_amendment_forcing_only_anchors.json",
        ROOT / "provenance/protocol_amendment_mcp_stable_partition.json",
        ROOT / "provenance/stable_mcp_partition_validation.json",
        ROOT / "provenance/preopening_freeze_registry.json",
        reference_summary_path,
        reference_pair_path,
        EDGES,
    ]
    audit = {
        "population_verified": True,
        "target_gauges": 3982,
        "forcing_anchor_gauges": int(
            frame[[f"{model}_r7" for model in eligible_models]].notna().any(axis=1).sum()
        ),
        "closure_eligible_models": list(eligible_models),
        "national": {
            "evaluable_pairs_kge_0_30": int(nat30.evaluable_pair.sum()),
            "materially_disagreeing_pairs_kge_0_30": int(nat30.material_disagreement.sum()),
            "high_skill_evaluable_pairs_kge_0_60": int(
                nat60.mutually_qualified_gauges.ge(30).sum()
            ),
            "high_skill_convergent_pairs_kge_0_60": nat60.loc[
                nat60.high_skill_convergence, ["first", "second"]
            ].to_dict("records"),
        },
        "reference_only": {
            "architecture_summary": reference_summary.to_dict("records"),
            "evaluable_pairs_kge_0_30": int(ref30.evaluable_pair.sum()),
            "materially_disagreeing_pairs_kge_0_30": int(ref30.material_disagreement.sum()),
            "high_skill_evaluable_pairs_kge_0_60": int(
                ref60.mutually_qualified_gauges.ge(30).sum()
            ),
            "high_skill_convergent_pairs_kge_0_60": ref60.loc[
                ref60.high_skill_convergence, ["first", "second"]
            ].to_dict("records"),
        },
        "central_non_guarantee_supported": bool(
            discovery["central_non_guarantee_supported"]
            and nat30.material_disagreement.any()
            and ref30.material_disagreement.any()
        ),
        "source_hashes": {str(path): sha256(path) for path in source_files},
    }
    (OUTPUT / "national_scientific_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))
    print("\nNational model-observation diagnostics\n", observation.to_string(index=False))
    print("\nNational connected-gauge diagnostics\n", nested.to_string(index=False))


if __name__ == "__main__":
    main()
