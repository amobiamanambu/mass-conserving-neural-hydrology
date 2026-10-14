#!/usr/bin/env python3
"""Audit NHDPlusV2.1 pair direction and local-distance sensitivity.

This script uses archived model-specific-storage-quantile fingerprints. It does
not refit a model or alter the primary identification population.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
EDGES = ROOT / "lowflow_spatial_dynamics/data_external/nldi/river_network_edges.csv"
GAUGES = PROJECT / "data/processed/station_index.csv"
ANCHORS = (
    PROJECT
    / "outputs/identification/strengthened_test/anchor_design_gauge_comparison.csv"
)
OUTPUT = PROJECT / "outputs/identification/nested_pair_topology_sensitivity.csv"
AUDIT = PROJECT / "outputs/identification/nested_pair_topology_audit.json"


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation computed as Pearson correlation of average ranks."""
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average")
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average")
    return float(x_rank.corr(y_rank))


def contrast_metrics(
    pairs: pd.DataFrame, fingerprints: pd.DataFrame, model: str
) -> dict[str, float | int]:
    downstream_model = fingerprints.loc[pairs.downstream_gage, model].to_numpy(float)
    upstream_model = fingerprints.loc[pairs.upstream_gage, model].to_numpy(float)
    downstream_observed = fingerprints.loc[
        pairs.downstream_gage, "observed_median_r7"
    ].to_numpy(float)
    upstream_observed = fingerprints.loc[
        pairs.upstream_gage, "observed_median_r7"
    ].to_numpy(float)
    model_contrast = downstream_model - upstream_model
    observed_contrast = downstream_observed - upstream_observed
    return {
        "pairs": int(len(pairs)),
        "contrast_spearman_rho": spearman_rho(model_contrast, observed_contrast),
        "median_absolute_contrast_error": float(
            np.median(np.abs(model_contrast - observed_contrast))
        ),
    }


def main() -> None:
    fingerprints = pd.read_csv(ANCHORS, dtype={"GAGE_ID": str})
    fingerprints = fingerprints[
        fingerprints.anchor_design.eq("model_specific_storage_quantile")
    ].copy()
    fingerprints["GAGE_ID"] = fingerprints.GAGE_ID.astype(str).str.zfill(8)
    fingerprints = fingerprints.set_index("GAGE_ID")

    edges = pd.read_csv(
        EDGES, dtype={"upstream_gage": str, "downstream_gage": str}
    )
    edges["upstream_gage"] = edges.upstream_gage.astype(str).str.zfill(8)
    edges["downstream_gage"] = edges.downstream_gage.astype(str).str.zfill(8)
    pairs = edges[
        edges.upstream_gage.isin(fingerprints.index)
        & edges.downstream_gage.isin(fingerprints.index)
    ].copy()
    pairs["network_distance_km"] = pd.to_numeric(
        pairs.network_distance_km, errors="coerce"
    )

    records: list[dict[str, float | int | str]] = []
    for maximum_distance in (25.0, 50.0, 100.0, np.inf):
        local = pairs[pairs.network_distance_km <= maximum_distance]
        for model in ("dhbv", "fluxgate"):
            records.append(
                {
                    "maximum_network_distance_km": (
                        "all" if np.isinf(maximum_distance) else int(maximum_distance)
                    ),
                    "model": model,
                    **contrast_metrics(local, fingerprints, model),
                }
            )
    pd.DataFrame(records).to_csv(OUTPUT, index=False)

    stations = pd.read_csv(GAUGES, dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.astype(str).str.zfill(8)
    area = stations.set_index("GAGE_ID").area_km2
    pairs["upstream_area_km2"] = pairs.upstream_gage.map(area)
    pairs["downstream_area_km2"] = pairs.downstream_gage.map(area)
    has_area = pairs[["upstream_area_km2", "downstream_area_km2"]].notna().all(axis=1)
    expected_area_order = (
        pairs.downstream_area_km2 >= pairs.upstream_area_km2
    ) & has_area
    exceptions = pairs.loc[
        has_area & ~expected_area_order,
        [
            "upstream_gage",
            "downstream_gage",
            "edge_type",
            "network_distance_km",
            "upstream_area_km2",
            "downstream_area_km2",
        ],
    ]
    retained = pairs.loc[~pairs.index.isin(exceptions.index)]
    audit = {
        "network": "NHDPlus Version 2.1 (NHDPlusV2.1; commonly NHDPlusV2)",
        "topology_rule": "same-reach measure and main-path DnHydroseq",
        "pairs": int(len(pairs)),
        "edge_types": {
            str(key): int(value) for key, value in pairs.edge_type.value_counts().items()
        },
        "network_distance_km": {
            "median": float(pairs.network_distance_km.median()),
            "percentile_90": float(pairs.network_distance_km.quantile(0.90)),
            "maximum": float(pairs.network_distance_km.max()),
        },
        "area_available_pairs": int(has_area.sum()),
        "pairs_with_non_decreasing_area": int(expected_area_order.sum()),
        "non_decreasing_area_fraction": float(expected_area_order.sum() / has_area.sum()),
        "area_order_exceptions": exceptions.to_dict(orient="records"),
        "contrast_metrics_after_excluding_area_order_exceptions": {
            model: contrast_metrics(retained, fingerprints, model)
            for model in ("dhbv", "fluxgate")
        },
        "caution": (
            "DnHydroseq represents the NHDPlusV2.1 main downstream path; "
            "divergent diversion branches are not represented."
        ),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
