#!/usr/bin/env python3
"""Summarize anchor-design and model-side q0 sensitivities."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


EXPERIMENT = Path(__file__).resolve().parents[1]
MODELS = ("dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr")
SEEDS = (11, 29, 47)
THRESHOLDS = (0.0, 0.30, 0.40, 0.50, 0.60)
Q0_COLUMNS = {0.0001: "median_r7_q0_0001", 0.001: "median_r7_q0_001", 0.01: "median_r7_q0_01"}


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def mad(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.median(np.abs(x[valid] - y[valid]))) if valid.any() else np.nan


def load_design(design: str, value_column: str) -> pd.DataFrame:
    output = None
    for model in MODELS:
        seeds = []
        for seed in SEEDS:
            path = EXPERIMENT / f"outputs/interventions/{model}_qet_seed{seed}_{design}_fingerprints.csv.gz"
            table = pd.read_csv(path, dtype={"GAGE_ID": str}, usecols=["GAGE_ID", value_column]).set_index("GAGE_ID")
            seeds.append(table[value_column])
        model_value = pd.concat(seeds, axis=1).median(axis=1).rename(model)
        output = model_value.to_frame() if output is None else output.join(model_value, how="outer")
    return output


def main() -> None:
    evaluation = EXPERIMENT / "outputs/evaluation"
    gauge = pd.read_csv(evaluation / "gauge_level_reference_only.csv.gz", dtype={"GAGE_ID": str}).set_index("GAGE_ID")
    rows = []
    design_frames = {}
    for design in ("common_date", "model_specific"):
        for q0, column in Q0_COLUMNS.items():
            response = load_design(design, column)
            design_frames[(design, q0)] = response
            for threshold in THRESHOLDS:
                for first, second in itertools.combinations(MODELS, 2):
                    mask = (
                        (gauge[f"{first}_kge"] >= threshold)
                        & (gauge[f"{second}_kge"] >= threshold)
                        & response[first].notna() & response[second].notna()
                    )
                    cross_rho = rho(response.loc[mask, first], response.loc[mask, second])
                    cross_mad = mad(response.loc[mask, first], response.loc[mask, second])
                    rows.append(
                        {"anchor_design": design, "q0_threshold_mm_day": q0,
                         "skill_threshold": threshold, "first": first, "second": second,
                         "n": int(mask.sum()), "cross_architecture_rho": cross_rho,
                         "median_absolute_r7_difference": cross_mad,
                         "material_disagreement": bool(mask.sum() >= 10 and (cross_rho < 0.75 or cross_mad > 0.05)) if np.isfinite(cross_rho) else False}
                    )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(evaluation / "anchor_q0_pairwise_sensitivity.csv", index=False)

    anchor_rows = []
    common = design_frames[("common_date", 0.001)]
    specific = design_frames[("model_specific", 0.001)]
    for model in MODELS:
        mask = (gauge[f"{model}_kge"] >= 0.30) & common[model].notna() & specific[model].notna()
        anchor_rows.append(
            {"model": model, "n": int(mask.sum()),
             "common_vs_model_specific_rho": rho(common.loc[mask, model], specific.loc[mask, model]),
             "common_vs_model_specific_mad": mad(common.loc[mask, model], specific.loc[mask, model])}
        )
    pd.DataFrame(anchor_rows).to_csv(evaluation / "anchor_design_agreement.csv", index=False)

    seed_rows = []
    for model in MODELS:
        seed_values = []
        for seed in SEEDS:
            path = EXPERIMENT / f"outputs/interventions/{model}_qet_seed{seed}_common_date_fingerprints.csv.gz"
            seed_values.append(
                pd.read_csv(path, dtype={"GAGE_ID": str}, usecols=["GAGE_ID", "median_r7_q0_001"])
                .set_index("GAGE_ID")["median_r7_q0_001"]
            )
        mask = gauge[f"{model}_kge"] >= 0.30
        local_rho, local_mad = [], []
        for first, second in itertools.combinations(range(len(SEEDS)), 2):
            valid = mask & seed_values[first].notna() & seed_values[second].notna()
            local_rho.append(rho(seed_values[first][valid], seed_values[second][valid]))
            local_mad.append(mad(seed_values[first][valid], seed_values[second][valid]))
        seed_rows.append(
            {"model": model, "n": int(mask.sum()),
             "median_seed_pair_rho": float(np.nanmedian(local_rho)),
             "median_seed_pair_mad": float(np.nanmedian(local_mad)),
             "minimum_seed_pair_rho": float(np.nanmin(local_rho)),
             "maximum_seed_pair_mad": float(np.nanmax(local_mad))}
        )
    seed_stability = pd.DataFrame(seed_rows)
    seed_stability.to_csv(evaluation / "seed_stability.csv", index=False)

    key = sensitivity.loc[
        (sensitivity["first"] == "dhbv") & (sensitivity["second"] == "fluxgate")
        & sensitivity.skill_threshold.isin([0.30, 0.60])
    ]
    print(key.to_string(index=False))
    print(pd.DataFrame(anchor_rows).to_string(index=False))
    print(seed_stability.to_string(index=False))


if __name__ == "__main__":
    main()
