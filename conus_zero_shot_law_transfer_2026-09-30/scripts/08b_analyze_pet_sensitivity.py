#!/usr/bin/env python3
"""Summarize signed bias and the climatological-PET reviewer sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (11, 29, 47)
ARCHITECTURES = ("dhbv", "fluxgate")


def rho(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    observed = frame.observed_median_r7.to_numpy(float)
    dhbv = frame[f"dhbv_{prefix}_r7"].to_numpy(float)
    fluxgate = frame[f"fluxgate_{prefix}_r7"].to_numpy(float)
    result: dict[str, float | int] = {"gauges": int(len(frame))}
    result["cross_architecture_rho"] = rho(dhbv, fluxgate)
    result["cross_architecture_median_absolute_difference"] = float(np.nanmedian(np.abs(dhbv - fluxgate)))
    for architecture, modeled in (("dhbv", dhbv), ("fluxgate", fluxgate)):
        difference = modeled - observed
        result[f"{architecture}_observed_rho"] = rho(modeled, observed)
        result[f"{architecture}_observed_median_absolute_error"] = float(np.nanmedian(np.abs(difference)))
        result[f"{architecture}_observed_median_signed_error"] = float(np.nanmedian(difference))
        result[f"{architecture}_observed_mean_signed_error"] = float(np.nanmean(difference))
        result[f"{architecture}_fraction_above_observed"] = float(np.nanmean(difference > 0))
        result[f"{architecture}_median_r7"] = float(np.nanmedian(modeled))
    return result


def main() -> None:
    frame = pd.read_parquet(
        PROJECT / "outputs/identification/primary_identification_population.parquet"
    ).copy()
    frame.GAGE_ID = frame.GAGE_ID.astype(str).str.zfill(8)

    for architecture in ARCHITECTURES:
        members = []
        for seed in SEEDS:
            table = pd.read_csv(
                PROJECT
                / f"outputs/law_interventions_pet_climatology/{architecture}_qet_seed{seed}_fingerprints.csv",
                dtype={"GAGE_ID": str},
            ).set_index("GAGE_ID")
            members.append(table.loc[frame.GAGE_ID, "median_r7"].to_numpy(float))
        frame[f"{architecture}_pet_climatology_r7"] = np.nanmedian(np.column_stack(members), axis=1)
        frame[f"{architecture}_zero_pet_r7"] = frame[f"{architecture}_qet_r7"]
        frame[f"{architecture}_pet_effect_r7"] = (
            frame[f"{architecture}_pet_climatology_r7"] - frame[f"{architecture}_zero_pet_r7"]
        )

    zero_summary = summarize(frame, "zero_pet")
    pet_summary = summarize(frame, "pet_climatology")
    change_summary = {}
    for architecture in ARCHITECTURES:
        delta = frame[f"{architecture}_pet_effect_r7"].to_numpy(float)
        change_summary[architecture] = {
            "median_signed_change_in_r7": float(np.nanmedian(delta)),
            "median_absolute_change_in_r7": float(np.nanmedian(np.abs(delta))),
            "p025_p975_change_in_r7": [float(x) for x in np.nanquantile(delta, [0.025, 0.975])],
        }

    rng = np.random.default_rng(20261001)
    groups = [group.index.to_numpy() for _, group in frame.groupby("ecoregion")]
    bootstrap_records = []
    for draw in range(1000):
        chosen = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
        sample = frame.loc[chosen]
        record = {"draw": draw}
        for label in ("zero_pet", "pet_climatology"):
            metrics = summarize(sample, label)
            for key, value in metrics.items():
                if key != "gauges":
                    record[f"{label}_{key}"] = value
        bootstrap_records.append(record)
    bootstrap = pd.DataFrame(bootstrap_records)

    intervals = {}
    for column in bootstrap.columns:
        if column != "draw":
            intervals[column] = [float(x) for x in bootstrap[column].quantile([0.025, 0.975])]

    result = {
        "population": "primary skill-qualified population",
        "gauges": int(len(frame)),
        "zero_pet": zero_summary,
        "climatological_pet": pet_summary,
        "change_from_zero_pet": change_summary,
        "bootstrap_ci95": intervals,
        "interpretation": (
            "Modeled R7 is biased below, not above, observed warm-season recession R7 under the zero-PET intervention. "
            "Restoring climatological PET does not rescue observational rank agreement."
        ),
    }

    output = PROJECT / "outputs/identification"
    frame.to_csv(output / "pet_climatology_sensitivity_gauge.csv.gz", index=False)
    bootstrap.to_csv(output / "pet_climatology_sensitivity_bootstrap.csv.gz", index=False)
    (output / "pet_climatology_sensitivity_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
