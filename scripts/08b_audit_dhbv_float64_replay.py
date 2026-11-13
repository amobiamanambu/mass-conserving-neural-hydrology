#!/usr/bin/env python3
"""Float64 replay of the author-source dHBV historical year and zero-input intervention.

Post-opening numerical audit (2026-11-08).  The HydroDL2 HBV 1.0 equations are not
changed.  The only modification is that the float32 dtype literals inside the author
module are rewritten to float64 at import time, so that the water-balance residual
is evaluated at double precision, exactly as the native and adapter replays were.
The frozen float32 archive is not modified; this script writes separate
float64 receipts, curves and per-day states next to it.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("NGEN_SILENT", "1")
if sys.stdin is None:  # hydrodl2's licence check calls sys.stdin.isatty(); a closed stdin (some launchers) crashes it
    sys.stdin = open(os.devnull)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

import hydrodl2.models.hbv.hbv as author_hbv_module  # noqa: E402
from hydrodl2.core.calc import uh_conv, uh_gamma  # noqa: E402


ANCHORS = ROOT / "data/forcing_only_anchors.npz"
MODEL = "author_dhbv"
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
HISTORY_DAYS = 365
FUTURE_DAYS = 91
DYNAMIC_PARAMETER_INDICES = (0, 12)  # parBETA and parBETAET in HydroDL2 HBV 1.0
FIXED_PARAMETER_INDICES = tuple(i for i in range(15) if i not in DYNAMIC_PARAMETER_INDICES)
PRESPECIFIED_TOLERANCE_MM = 0.005  # unchanged dHBV tolerance from script 08
STRICT_TOLERANCE_MM = 1e-9  # native/adapter tolerance, reported for comparison


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_float64_hbv_class():
    """Return the author Hbv class with float32 dtype literals rewritten to float64."""
    source = inspect.getsource(author_hbv_module)
    count = source.count("torch.float32")
    patched = source.replace("torch.float32", "torch.float64")
    module = types.ModuleType("hydrodl2_hbv_float64_audit")
    module.__file__ = str(author_hbv_module.__file__) + ".float64audit"
    exec(compile(patched, module.__file__, "exec"), module.__dict__)
    base_hbv = module.Hbv

    class Hbv(base_hbv):  # same class name so config['dynamic_params']['Hbv'] resolves
        def _init_states(self, ngrid: int):
            return tuple(
                torch.full(
                    (ngrid, self.nmul),
                    float(self.initial_state_value),
                    dtype=torch.float64,
                    device=self.device,
                )
                for _ in range(len(self.state_names))
            )

    return Hbv, hashlib.sha256(source.encode()).hexdigest(), count


def float64_hbv_from(model) -> torch.nn.Module:
    Hbv64, _, _ = build_float64_hbv_class()
    original = model.hbv
    config = {
        "warmup": original.warmup,
        "warmup_states": original.warmup_states,
        "dy_drop": original.dy_drop,
        "dynamic_params": {"Hbv": list(original.dynamic_params)},
        "variables": list(original.variables),
        "routing": original.routing,
        "comprout": original.comprout,
        "nearzero": original.nearzero,
        "nmul": original.nmul,
        "cache_states": False,
    }
    hbv64 = Hbv64(config, device=torch.device("cpu"))
    if list(hbv64.parameter_bounds.keys()) != list(original.parameter_bounds.keys()):
        raise RuntimeError("Float64 HBV parameter layout differs from the author instance")
    if hbv64.routing_len != original.routing_len:
        raise RuntimeError("Routing length mismatch")
    return hbv64


def double_batch(batch: dict) -> dict:
    return {
        "x_d": {name: value.double() for name, value in batch["x_d"].items()},
        "x_s": batch["x_s"].double(),
    }


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
        future_normalized = torch.full((batch_size, FUTURE_DAYS, 1), normalized, dtype=torch.float64)
        future_raw = torch.full((batch_size, FUTURE_DAYS, 1), raw_value, dtype=torch.float64)
        x_d[feature] = torch.cat([batch["x_d"][feature].double(), future_normalized], dim=1)
        copy_name = f"{feature}_copy1"
        x_d[copy_name] = torch.cat([batch["x_d"][copy_name].double(), future_raw], dim=1)
    return {"x_d": x_d, "x_s": batch["x_s"].double()}


def route_with_author_uh(hbv, q_unrouted: torch.Tensor, historical_parameters: torch.Tensor) -> torch.Tensor:
    _, normalized_route = hbv._unpack_parameters(historical_parameters)
    route = hbv._descale_route_parameters(normalized_route)
    steps = q_unrouted.shape[1]
    unit_hydrograph = uh_gamma(
        route["route_a"].repeat(steps, 1).unsqueeze(-1),
        route["route_b"].repeat(steps, 1).unsqueeze(-1),
        lenF=hbv.routing_len,
    ).permute(1, 2, 0)
    return uh_conv(q_unrouted.unsqueeze(1), unit_hydrograph).squeeze(1)


def replay(model, hbv64, tester, batch: dict) -> dict:
    """Float64 historical year + intervention, mirroring vendor/reference_helpers/author_dhbv_interventions.py."""
    fixed = list(FIXED_PARAMETER_INDICES)
    with torch.no_grad():
        raw = model._raw_forcing(batch)  # [time, basin, 3]
        historical_parameters = model.parameterizer(model._parameter_inputs(batch))  # [time, basin, 15]
        hbv64.device = raw.device
        hbv64.cache_states = False
        hbv64.states = None
        hbv64.routing_state = None
        flux = hbv64({"x_phy": raw}, historical_parameters)
        historical_states = tuple(state.clone() for state in hbv64._state_cache)
        q_history_routed_author = flux["streamflow"][:, :, 0].transpose(0, 1)
        q_history_unrouted = flux["streamflow_no_rout"].mean(dim=2).transpose(0, 1)
        et_history = flux["AET_hydro"][:, :, 0].transpose(0, 1)

        extended = extend_batch(batch, tester)
        extended_parameters = model.parameterizer(model._parameter_inputs(extended))
        extended_parameters[-1, :, fixed] = historical_parameters[-1, :, fixed]

        hbv64.cache_states = True
        hbv64.states = tuple(state.clone() for state in historical_states)
        hbv64.routing_state = None
        future_raw = model._raw_forcing(extended)[HISTORY_DAYS:]
        q_future_unrouted, et_future, storage_after, process_residual = [], [], [], []
        state_series = [torch.stack([s[:, 0] for s in historical_states], dim=-1)]  # [basin, 5]
        previous_storage = torch.stack(historical_states, dim=0).sum(dim=(0, 2))
        state_minimum = torch.tensor(float("inf"), dtype=torch.float64)
        for day in range(FUTURE_DAYS):
            parameters = extended_parameters[HISTORY_DAYS + day:HISTORY_DAYS + day + 1].clone()
            parameters[:, :, fixed] = historical_parameters[-1:, :, fixed]
            day_flux = hbv64({"x_phy": future_raw[day:day + 1]}, parameters)
            states = tuple(state.clone() for state in hbv64._state_cache)
            storage = torch.stack(states, dim=0).sum(dim=(0, 2))
            q_day = day_flux["streamflow_no_rout"][0, :, 0]
            et_day = day_flux["AET_hydro"][0, :, 0]
            process_residual.append(previous_storage - q_day - et_day - storage)
            q_future_unrouted.append(q_day)
            et_future.append(et_day)
            storage_after.append(storage)
            state_series.append(torch.stack([s[:, 0] for s in states], dim=-1))
            state_minimum = torch.minimum(state_minimum, torch.stack(states).min().cpu())
            previous_storage = storage
        hbv64.cache_states = False
        hbv64.states = None
        hbv64.routing_state = None

        q_future_unrouted = torch.stack(q_future_unrouted, dim=1)
        et_future = torch.stack(et_future, dim=1)
        storage_after = torch.stack(storage_after, dim=1)
        process_residual = torch.stack(process_residual, dim=1)
        state_series = torch.stack(state_series, dim=1)  # [basin, 92, 5]

        q_all_unrouted = torch.cat([q_history_unrouted, q_future_unrouted], dim=1)
        q_all_routed = route_with_author_uh(hbv64, q_all_unrouted, historical_parameters)
        q_history_routed = q_all_routed[:, :HISTORY_DAYS]
        q_future_routed = q_all_routed[:, HISTORY_DAYS:]
        historical_route_difference = (q_history_routed - q_history_routed_author).abs().max()

        routing_buffer_after = q_all_unrouted.cumsum(dim=1) - q_all_routed.cumsum(dim=1)
        routing_before = routing_buffer_after[:, HISTORY_DAYS - 1]
        future_routing_after = routing_buffer_after[:, HISTORY_DAYS:]
        initial_physical = torch.stack(historical_states, dim=0).sum(dim=(0, 2))
        total_before = torch.cat(
            [(initial_physical + routing_before).unsqueeze(1),
             storage_after[:, :-1] + future_routing_after[:, :-1]], dim=1)
        total_after = storage_after + future_routing_after
        total_daily_residual = total_before - et_future - q_future_routed - total_after

        initial_hbv_storage = len(hbv64.state_names) * float(hbv64.initial_state_value)
        historical_cumulative_residual = (
            initial_hbv_storage
            + raw[:, :, 0].transpose(0, 1).sum(dim=1)
            - et_history.sum(dim=1)
            - q_history_routed_author.sum(dim=1)
            - (initial_physical + routing_before)
        )
        routing_series = torch.cat([routing_before.unsqueeze(1), future_routing_after], dim=1)  # [basin, 92]
    return {
        "q": q_future_routed,
        "q_unrouted": q_future_unrouted,
        "et": et_future,
        "storage": total_after,
        "states": state_series,
        "routing_buffer": routing_series,
        "process_daily_residual": process_residual,
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
    r7[valid] = curves[valid, 7] / q0[valid]
    return {
        "valid_anchors": int(valid.sum()),
        "median_r7": float(np.nanmedian(r7[valid])) if valid.any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="debug: only the first N gauges")
    args = parser.parse_args()

    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    info = selection["architectures"][MODEL]
    epoch = int(info["selected_epoch"])
    run_dir = ROOT / info["runs"][str(args.seed)]["run_dir"]
    cfg = Config(run_dir / "config.yml")
    tester = get_tester(cfg, run_dir, "test", True)
    tester._load_weights(epoch)
    model = tester.model.eval()
    _, author_source_sha, float32_literals = build_float64_hbv_class()
    hbv64 = float64_hbv_from(model)
    model.double().eval()
    torch.set_default_dtype(torch.float64)  # routing helpers allocate with the default dtype

    anchors = np.load(ANCHORS)
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} gauges, found {len(gauges)}")
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Anchor order mismatch")
    frozen = np.load(ROOT / f"outputs/frozen/{MODEL}_seed{args.seed}_intervention_curves.npz")
    frozen_q = frozen["q_mm_day"]
    n_gauges = TARGET_GAUGES if args.limit is None else min(args.limit, TARGET_GAUGES)

    q_curve = np.full((TARGET_GAUGES, 12, FUTURE_DAYS), np.nan)
    storage_curve = np.full_like(q_curve, np.nan)
    et_curve = np.full_like(q_curve, np.nan)
    state_curve = np.full((TARGET_GAUGES, 12, FUTURE_DAYS + 1, 5), np.nan, dtype=np.float32)
    routing_curve = np.full((TARGET_GAUGES, 12, FUTURE_DAYS + 1), np.nan, dtype=np.float32)
    maxima = {k: 0.0 for k in ("process_daily", "total_daily", "total_cumulative", "historical_cumulative", "route_difference")}
    worst_gauge = {k: "" for k in maxima}
    minimum_state = np.inf
    minimum_routing_buffer = np.inf
    max_q_difference_vs_float32 = 0.0
    records = []
    for basin_index, gauge in enumerate(gauges[:n_gauges]):
        indices = anchors["eval_time_index"][basin_index].astype(int)
        dataset = tester._get_dataset(gauge)
        batch = double_batch(dataset.collate_fn([dataset[int(index)] for index in indices]))
        result = replay(model, hbv64, tester, batch)
        q_curve[basin_index] = result["q"].cpu().numpy()
        storage_curve[basin_index] = result["storage"].cpu().numpy()
        et_curve[basin_index] = result["et"].cpu().numpy()
        state_curve[basin_index] = result["states"].cpu().numpy().astype(np.float32)
        routing_curve[basin_index] = result["routing_buffer"].cpu().numpy().astype(np.float32)
        local = {
            "process_daily": float(result["process_daily_residual"].abs().max()),
            "total_daily": float(result["total_daily_residual"].abs().max()),
            "total_cumulative": float(result["total_daily_residual"].cumsum(dim=1).abs().max()),
            "historical_cumulative": float(result["historical_cumulative_residual"].abs().max()),
            "route_difference": float(result["historical_route_difference"]),
        }
        for key, value in local.items():
            if value > maxima[key]:
                maxima[key] = value
                worst_gauge[key] = gauge
        minimum_state = min(minimum_state, float(result["minimum_state"]))
        minimum_routing_buffer = min(minimum_routing_buffer, float(result["minimum_routing_buffer"]))
        max_q_difference_vs_float32 = max(
            max_q_difference_vs_float32,
            float(np.nanmax(np.abs(q_curve[basin_index] - frozen_q[basin_index]))),
        )
        records.append({"GAGE_ID": gauge, "model": MODEL, "seed": args.seed, **summarize(q_curve[basin_index])})
        if (basin_index + 1) % 100 == 0 or basin_index + 1 == n_gauges:
            print(f"{MODEL} float64 seed {args.seed}: {basin_index + 1}/{n_gauges} gauges; "
                  f"max historical cumulative residual so far {maxima['historical_cumulative']:.3e} mm", flush=True)

    output_dir = ROOT / "outputs/frozen"
    tag = "" if args.limit is None else f"_limit{n_gauges}"
    curve_path = output_dir / f"{MODEL}_seed{args.seed}_float64_replay_curves{tag}.npz"
    np.savez_compressed(
        curve_path,
        gauge_ids=np.asarray(gauges),
        quantiles=anchors["quantiles"],
        antecedent_time_index=anchors["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
        et_mm=et_curve,
        states_mm=state_curve,
        state_names=np.asarray(["SNOWPACK", "MELTWATER", "SM", "SUZ", "SLZ"]),
        routing_buffer_mm=routing_curve,
        state_day_index_note=np.asarray("index 0 = state at the anchor date before the intervention; index t = after intervention day t-1"),
    )
    fingerprint_path = output_dir / f"{MODEL}_seed{args.seed}_float64_replay_fingerprints{tag}.csv"
    pd.DataFrame(records).to_csv(fingerprint_path, index=False)
    frozen_fp = pd.read_csv(output_dir / f"{MODEL}_seed{args.seed}_fingerprints.csv", dtype={"GAGE_ID": str})
    r7_float32 = frozen_fp.median_r7.to_numpy(float)[:n_gauges]
    r7_float64 = pd.DataFrame(records).median_r7.to_numpy(float)
    receipt = {
        "model": MODEL,
        "seed": args.seed,
        "selected_epoch": epoch,
        "audit": "post-opening float64 replay of the author HBV 1.0 core; equations unchanged, dtype literals rewritten",
        "author_hbv_source_sha256": author_source_sha,
        "float32_literals_rewritten": float32_literals,
        "hydrodl2_nearzero_floor_mm": float(model.hbv.nearzero),
        "gauges_replayed": n_gauges,
        "anchors_per_gauge": 12,
        "historical_window_days": HISTORY_DAYS,
        "intervention_days": FUTURE_DAYS,
        "prespecified_dhbv_tolerance_mm": PRESPECIFIED_TOLERANCE_MM,
        "native_adapter_tolerance_mm": STRICT_TOLERANCE_MM,
        "maximum_historical_cumulative_residual_mm": maxima["historical_cumulative"],
        "worst_historical_gauge": worst_gauge["historical_cumulative"],
        "maximum_intervention_process_daily_residual_mm": maxima["process_daily"],
        "maximum_intervention_total_daily_residual_mm": maxima["total_daily"],
        "maximum_intervention_total_cumulative_drift_mm": maxima["total_cumulative"],
        "maximum_external_vs_author_historical_routing_difference_mm_day": maxima["route_difference"],
        "minimum_future_physical_state_mm": minimum_state,
        "minimum_routing_buffer_mm": minimum_routing_buffer,
        "maximum_q_difference_float64_vs_frozen_float32_mm_day": max_q_difference_vs_float32,
        "median_absolute_r7_difference_float64_vs_float32": float(np.nanmedian(np.abs(r7_float64 - r7_float32))),
        "maximum_absolute_r7_difference_float64_vs_float32": float(np.nanmax(np.abs(r7_float64 - r7_float32))),
        "passes_prespecified_tolerance_in_float64": bool(
            max(maxima.values()) <= PRESPECIFIED_TOLERANCE_MM
            and minimum_state >= -PRESPECIFIED_TOLERANCE_MM
            and minimum_routing_buffer >= -PRESPECIFIED_TOLERANCE_MM
        ),
        "passes_native_tolerance_in_float64": bool(max(maxima.values()) <= STRICT_TOLERANCE_MM),
        "curves": str(curve_path.relative_to(ROOT)),
        "curves_sha256": sha256(curve_path),
        "fingerprints": str(fingerprint_path.relative_to(ROOT)),
        "fingerprints_sha256": sha256(fingerprint_path),
        "frozen_float32_archive_unchanged": True,
    }
    receipt_path = output_dir / f"{MODEL}_seed{args.seed}_float64_replay_receipt{tag}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
