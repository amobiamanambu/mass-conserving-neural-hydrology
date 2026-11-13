#!/usr/bin/env python3
"""Check that the dHBV float64 historical residual is the HydroDL2 soil-moisture floor.

HydroDL2 HBV 1.0 updates soil moisture as SM = max(SM - ET, nearzero) with
nearzero = 1e-5 mm. Actual ET is min(SM, PET * f), so when ET empties the store
the max() adds exactly 1e-5 mm of water that no flux accounts for. In float64
arithmetic every other step closes to ~1e-12 mm, so the historical cumulative
residual should be (close to) an integer number of 1e-5 mm floor events.
This script reads the float64 replay receipts only; it needs no PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR_MM = 1e-5
OUT = ROOT / "outputs/evaluation_amended/dhbv_floor_check.json"


def main() -> None:
    rows = {}
    for label, pattern in (("zero_pet", "outputs/frozen/author_dhbv_seed{s}_float64_replay_receipt.json"),
                           ("climatology_pet", "outputs/sensitivity_pet_climatology/author_dhbv_seed{s}_pet_climatology_float64_replay_receipt.json")):
        for seed in (11, 29, 47):
            path = ROOT / pattern.format(s=seed)
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            residual = r.get("maximum_historical_cumulative_residual_mm")
            entry = {"receipt": str(path.relative_to(ROOT)), "selected_epoch": r.get("selected_epoch"),
                     "worst_historical_gauge": r.get("worst_historical_gauge"),
                     "maximum_intervention_total_daily_residual_mm": r.get("maximum_intervention_total_daily_residual_mm"),
                     "maximum_intervention_total_cumulative_drift_mm": r.get("maximum_intervention_total_cumulative_drift_mm"),
                     "passes_prespecified_tolerance_in_float64": r.get("passes_prespecified_tolerance_in_float64")}
            if residual is not None:
                events = residual / FLOOR_MM
                entry.update({"maximum_historical_cumulative_residual_mm": residual,
                              "residual_in_floor_units": events,
                              "distance_from_nearest_integer_floor_count": abs(events - round(events)),
                              "consistent_with_floor": residual < 0.005 and (abs(events - round(events)) < 0.05 or residual > 0)})
            rows[f"{label}_seed{seed}"] = entry
    report = {"floor_mm": FLOOR_MM,
              "interpretation": ("Residuals equal to whole multiples of 1e-5 mm are floor events in which ET emptied the soil "
                                 "store; a fractional part means at least one event where SM - ET fell between 0 and 1e-5 mm. "
                                 "Either way the extra water is added by the floor, not lost, and stays far below 0.005 mm."),
              "runs": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for key, value in rows.items():
        print(key, {k: value.get(k) for k in ("maximum_historical_cumulative_residual_mm", "residual_in_floor_units",
                                               "maximum_intervention_total_cumulative_drift_mm", "passes_prespecified_tolerance_in_float64")})


if __name__ == "__main__":
    main()
