#!/usr/bin/env python3
"""Freeze blind author-source dHBV drainage curves with routing-buffer closure.

The intervention uses the trained author LSTM continuously across the 365-day
history and 91-day zero-input future.  Future HBV state transitions call the
original HydroDL2 core one day at a time, and routing calls the original
HydroDL2 gamma unit-hydrograph functions.  This exposes a daily routing buffer
without rewriting either the HBV process equations or the routing equations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("NGEN_SILENT", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from hydrodl2.core.calc import uh_conv, uh_gamma  # noqa: E402
from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402


ANCHORS = ROOT / "data/source_inputs/reference_only/common_forcing_anchors.npz"
MODEL = "author_dhbv"
SEEDS = (11, 29, 47)
HISTORY_DAYS = 365
FUTURE_DAYS = 91
PHYSICAL_PARAMETER_COUNT = 13
DYNAMIC_PARAMETER_INDICES = (0, 12)  # parBETA and parBETAET in HydroDL2 HBV 1.0
FIXED_PARAMETER_INDICES = tuple(
    index for index in range(15) if index not in DYNAMIC_PARAMETER_INDICES
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_constant(tester, feature: str, raw_value: float) -> float:
    center = float(tester.scaler["xarray_feature_center"][feature].values)
    scale = float(tester.scaler["xarray_feature_scale"][feature].values)
    if not np.isfinite(scale) or scale == 0:
        raise RuntimeError(f"Invalid normalization scale for {feature}: {scale}")
    return (raw_value - center) / scale


def extend_batch(batch: dict, tester) -> dict:
    batch_size = batch["x_s"].shape[0]
    raw_values = {"pr_mm": 0.0, "tmean_c": 10.0, "pet_mm": 0.0}
    x_d = {}
    for feature, raw_value in raw_values.items():
        normalized = normalized_constant(tester, feature, raw_value)
        future_normalized = torch.full(
            (batch_size, FUTURE_DAYS, 1), normalized, dtype=torch.float32
        )
        future_raw = torch.full(
            (batch_size, FUTURE_DAYS, 1), raw_value, dtype=torch.float32
        )
        x_d[feature] = torch.cat([batch["x_d"][feature].float(), future_normalized], dim=1)
        copy_name = f"{feature}_copy1"
        x_d[copy_name] = torch.cat([batch["x_d"][copy_name].float(), future_raw], dim=1)
    return {"x_d": x_d, "x_s": batch["x_s"].float()}


def route_with_author_uh(
    hbv, q_unrouted: torch.Tensor, historical_parameters: torch.Tensor
) -> torch.Tensor:
    _, normalized_route = hbv._unpack_parameters(historical_parameters)
    route = hbv._descale_route_parameters(normalized_route)
    steps = q_unrouted.shape[1]
    unit_hydrograph = uh_gamma(
        route["route_a"].repeat(steps, 1).unsqueeze(-1),
        route["route_b"].repeat(steps, 1).unsqueeze(-1),
        lenF=hbv.routing_len,
    ).permute(1, 2, 0)
    return uh_conv(q_unrouted.unsqueeze(1), unit_hydrograph).squeeze(1)


def run_intervention(model, tester, batch: dict) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        historical_parameters = model.parameterizer(model._parameter_inputs(batch))
        historical = model(batch)
        historical_states = tuple(state.clone() for state in model.hbv._state_cache)
        extended = extend_batch(batch, tester)
        extended_parameters = model.parameterizer(model._parameter_inputs(extended))

        # HydroDL2 treats all non-dynamic process and routing parameters as the
        # final sequence value. Preserve the value identified at the historical
        # anchor so appending an intervention cannot retroactively alter the
        # historical parameterization.
        fixed = list(FIXED_PARAMETER_INDICES)
        extended_parameters[-1, :, fixed] = historical_parameters[-1, :, fixed]

        hbv = model.hbv
        original_routing = hbv.routing
        original_cache = hbv.cache_states
        original_states = hbv.states
        original_routing_state = hbv.routing_state
        # Keep author routing enabled inside each one-day source call. HBV 1.0
        # computes its BFI diagnostic from routed component tensors even when
        # only the unrouted process flux is needed; disabling routing therefore
        # leaves those tensors as None. The one-day routed output is ignored.
        # Drainage is routed once, independently, below with the original
        # full-history author unit hydrograph.
        hbv.routing = True
        hbv.cache_states = True
        hbv.states = tuple(state.clone() for state in historical_states)
        hbv.routing_state = None

        q_future_unrouted = []
        et_future = []
        physical_storage_after = []
        process_daily_residual = []
        state_minimum = torch.tensor(float("inf"))
        try:
            previous_storage = torch.stack(historical_states, dim=0).sum(dim=(0, 2))
            future_raw = model._raw_forcing(extended)[HISTORY_DAYS:]
            for day in range(FUTURE_DAYS):
                parameters = extended_parameters[HISTORY_DAYS + day:HISTORY_DAYS + day + 1].clone()
                parameters[:, :, fixed] = historical_parameters[-1:, :, fixed]
                flux = hbv({"x_phy": future_raw[day:day + 1]}, parameters)
                states = tuple(state.clone() for state in hbv._state_cache)
                storage = torch.stack(states, dim=0).sum(dim=(0, 2))
                q_day = flux["streamflow_no_rout"][0, :, 0]
                et_day = flux["AET_hydro"][0, :, 0]
                process_daily_residual.append(previous_storage - q_day - et_day - storage)
                q_future_unrouted.append(q_day)
                et_future.append(et_day)
                physical_storage_after.append(storage)
                state_minimum = torch.minimum(
                    state_minimum, torch.stack(states).min().cpu()
                )
                previous_storage = storage
        finally:
            hbv.routing = original_routing
            hbv.cache_states = original_cache
            hbv.states = original_states
            hbv.routing_state = original_routing_state

        q_future_unrouted = torch.stack(q_future_unrouted, dim=1)
        et_future = torch.stack(et_future, dim=1)
        physical_storage_after = torch.stack(physical_storage_after, dim=1)
        process_daily_residual = torch.stack(process_daily_residual, dim=1)
        q_history_unrouted = historical["q_unrouted"]
        q_all_unrouted = torch.cat([q_history_unrouted, q_future_unrouted], dim=1)
        q_all_routed = route_with_author_uh(hbv, q_all_unrouted, historical_parameters)
        q_history_routed = q_all_routed[:, :HISTORY_DAYS]
        q_future_routed = q_all_routed[:, HISTORY_DAYS:]
        historical_route_difference = (
            q_history_routed - historical["q"]
        ).abs().max()

        routing_buffer_after = q_all_unrouted.cumsum(dim=1) - q_all_routed.cumsum(dim=1)
        routing_before = routing_buffer_after[:, HISTORY_DAYS - 1]
        future_routing_after = routing_buffer_after[:, HISTORY_DAYS:]
        initial_physical = torch.stack(historical_states, dim=0).sum(dim=(0, 2))
        total_before = torch.cat(
            [
                (initial_physical + routing_before).unsqueeze(1),
                physical_storage_after[:, :-1] + future_routing_after[:, :-1],
            ],
            dim=1,
        )
        total_after = physical_storage_after + future_routing_after
        total_daily_residual = total_before - et_future - q_future_routed - total_after

        initial_hbv_storage = 5.0 * hbv.initial_state_value
        historical_final_total = initial_physical + routing_before
        historical_cumulative_residual = (
            initial_hbv_storage
            + model._raw_forcing(batch)[:, :, 0].transpose(0, 1).sum(dim=1)
            - historical["et"].sum(dim=1)
            - historical["q"].sum(dim=1)
            - historical_final_total
        )
    return {
        "q": q_future_routed,
        "q_unrouted": q_future_unrouted,
        "et": et_future,
        "storage": total_after,
        "process_daily_residual": process_daily_residual,
        "total_daily_residual": total_daily_residual,
        "historical_cumulative_residual": historical_cumulative_residual,
        "historical_route_difference": historical_route_difference,
        "minimum_state": state_minimum,
        "minimum_routing_buffer": routing_buffer_after.min(),
    }


def summarize(curves: np.ndarray, threshold: float = 0.001) -> dict:
    q0 = curves[:, 0]
    valid = np.isfinite(q0) & (q0 >= threshold)
    r7 = np.full(len(curves), np.nan)
    r30 = np.full(len(curves), np.nan)
    r7[valid] = curves[valid, 7] / q0[valid]
    r30[valid] = curves[valid, 30] / q0[valid]
    return {
        "valid_anchors": int(valid.sum()),
        "median_q0_mm_day": float(np.nanmedian(np.where(valid, q0, np.nan))),
        "median_r7": float(np.nanmedian(r7)),
        "median_r30": float(np.nanmedian(r30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args()
    selection = json.loads(
        (ROOT / "provenance/author_development_epoch_selection.json").read_text()
    )
    info = selection["architectures"][MODEL]
    epoch = int(info["selected_epoch"])
    run_dir = ROOT / info["runs"][str(args.seed)]["run_dir"]
    prediction_receipt = json.loads(
        (ROOT / f"outputs/frozen/{MODEL}_seed{args.seed}_prediction_receipt.json").read_text()
    )
    if prediction_receipt.get("target_discharge_accessed") is not False:
        raise RuntimeError("Target blinding receipt failed")

    cfg = Config(run_dir / "config.yml")
    tester = get_tester(cfg, run_dir, "test", True)
    tester._load_weights(epoch)
    model = tester.model.eval()
    anchors = np.load(ANCHORS)
    gauges = (ROOT / "data/basin_lists/test_413.txt").read_text().split()
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Anchor order mismatch")

    q_curve = np.full((413, 12, FUTURE_DAYS), np.nan, dtype=np.float64)
    storage_curve = np.full_like(q_curve, np.nan)
    records = []
    maximum_process_daily = 0.0
    maximum_total_daily = 0.0
    maximum_total_cumulative = 0.0
    maximum_historical_cumulative = 0.0
    maximum_route_difference = 0.0
    minimum_state = np.inf
    minimum_routing_buffer = np.inf
    for basin_index, gauge in enumerate(gauges):
        dataset = tester._get_dataset(gauge)
        indices = anchors["eval_time_index"][basin_index].astype(int)
        batch = dataset.collate_fn([dataset[int(index)] for index in indices])
        result = run_intervention(model, tester, batch)
        q_curve[basin_index] = result["q"].cpu().numpy()
        storage_curve[basin_index] = result["storage"].cpu().numpy()
        maximum_process_daily = max(
            maximum_process_daily, float(result["process_daily_residual"].abs().max())
        )
        maximum_total_daily = max(
            maximum_total_daily, float(result["total_daily_residual"].abs().max())
        )
        maximum_total_cumulative = max(
            maximum_total_cumulative,
            float(result["total_daily_residual"].cumsum(dim=1).abs().max()),
        )
        maximum_historical_cumulative = max(
            maximum_historical_cumulative,
            float(result["historical_cumulative_residual"].abs().max()),
        )
        maximum_route_difference = max(
            maximum_route_difference, float(result["historical_route_difference"])
        )
        minimum_state = min(minimum_state, float(result["minimum_state"]))
        minimum_routing_buffer = min(
            minimum_routing_buffer, float(result["minimum_routing_buffer"])
        )
        records.append({
            "GAGE_ID": gauge,
            "model": MODEL,
            "seed": args.seed,
            **summarize(q_curve[basin_index]),
        })
        if (basin_index + 1) % 25 == 0 or basin_index + 1 == 413:
            print(f"{MODEL} seed {args.seed}: {basin_index + 1}/413 gauges", flush=True)

    output_dir = ROOT / "outputs/frozen"
    curve_path = output_dir / f"{MODEL}_seed{args.seed}_intervention_curves.npz"
    np.savez_compressed(
        curve_path,
        gauge_ids=np.asarray(gauges),
        quantiles=anchors["quantiles"],
        antecedent_time_index=anchors["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    fingerprint_path = output_dir / f"{MODEL}_seed{args.seed}_fingerprints.csv"
    pd.DataFrame(records).to_csv(fingerprint_path, index=False)
    tolerance = 0.005  # HydroDL2 HBV 1.0 allocates process arrays as float32.
    receipt = {
        "model": MODEL,
        "seed": args.seed,
        "selected_epoch": epoch,
        "target_discharge_accessed": False,
        "target_gauges": 413,
        "anchors_per_gauge": 12,
        "historical_window_days": HISTORY_DAYS,
        "intervention_days": FUTURE_DAYS,
        "intervention": "P=0, PET=0, T=10 C",
        "process_equations": "original HydroDL2 HBV 1.0 called one day at a time",
        "routing_equations": "original HydroDL2 uh_gamma and uh_conv",
        "closure_tolerance_mm": tolerance,
        "maximum_intervention_process_daily_residual_mm": maximum_process_daily,
        "maximum_intervention_total_daily_residual_mm": maximum_total_daily,
        "maximum_intervention_total_cumulative_drift_mm": maximum_total_cumulative,
        "maximum_historical_cumulative_residual_mm": maximum_historical_cumulative,
        "maximum_external_vs_author_historical_routing_difference_mm_day": maximum_route_difference,
        "minimum_future_physical_state_mm": minimum_state,
        "minimum_routing_buffer_mm": minimum_routing_buffer,
        "passes_closure": bool(
            max(
                maximum_process_daily,
                maximum_total_daily,
                maximum_total_cumulative,
                maximum_historical_cumulative,
                maximum_route_difference,
            ) <= tolerance
            and minimum_state >= -tolerance
            and minimum_routing_buffer >= -tolerance
        ),
        "curves": str(curve_path.relative_to(ROOT)),
        "curves_sha256": sha256(curve_path),
        "fingerprints": str(fingerprint_path.relative_to(ROOT)),
        "fingerprints_sha256": sha256(fingerprint_path),
    }
    receipt_path = output_dir / f"{MODEL}_seed{args.seed}_intervention_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
