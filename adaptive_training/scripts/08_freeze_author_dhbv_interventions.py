#!/usr/bin/env python3
"""Freeze national author-source dHBV interventions and routing-aware closure audits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
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
from _pet_climatology import (  # noqa: E402
    CLIMATOLOGY_END,
    CLIMATOLOGY_START,
    seasonal_pet_climatology,
)


REFERENCE_HELPERS = (
    ROOT / "vendor/reference_helpers/author_dhbv_interventions.py"
)
spec = importlib.util.spec_from_file_location("frozen_author_dhbv_helpers", REFERENCE_HELPERS)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load dHBV intervention helpers from {REFERENCE_HELPERS}")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

ANCHORS = ROOT / "data" / "forcing_only_anchors.npz"
MODEL = "author_dhbv"
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
FUTURE_DAYS = 91


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(curves: np.ndarray, threshold: float = 0.001) -> dict:
    q0 = curves[:, 0]
    valid = np.isfinite(q0) & (q0 >= threshold)
    r7 = np.full(len(curves), np.nan)
    r30 = np.full(len(curves), np.nan)
    r7[valid] = curves[valid, 7] / q0[valid]
    r30[valid] = curves[valid, 30] / q0[valid]
    return {
        "valid_anchors": int(valid.sum()),
        "median_q0_mm_day": float(np.nanmedian(q0[valid])) if valid.any() else np.nan,
        "median_r7": float(np.nanmedian(r7[valid])) if valid.any() else np.nan,
        "median_r30": float(np.nanmedian(r30[valid])) if valid.any() else np.nan,
    }


def extend_batch_with_pet(batch: dict, tester, pet: np.ndarray) -> dict:
    """Append a 91-day P=0, T=10 C future with supplied PET."""
    batch_size = batch["x_s"].shape[0]
    raw_values = {
        "pr_mm": np.zeros((batch_size, FUTURE_DAYS), dtype=np.float32),
        "tmean_c": np.full((batch_size, FUTURE_DAYS), 10.0, dtype=np.float32),
        "pet_mm": np.asarray(pet, dtype=np.float32),
    }
    x_d = {}
    for feature, raw in raw_values.items():
        center = float(tester.scaler["xarray_feature_center"][feature].values)
        scale = float(tester.scaler["xarray_feature_scale"][feature].values)
        normalized = (raw - center) / scale
        future_normalized = torch.as_tensor(normalized).unsqueeze(-1)
        future_raw = torch.as_tensor(raw).unsqueeze(-1)
        x_d[feature] = torch.cat(
            [batch["x_d"][feature].float(), future_normalized], dim=1
        )
        copy_name = f"{feature}_copy1"
        x_d[copy_name] = torch.cat(
            [batch["x_d"][copy_name].float(), future_raw], dim=1
        )
    return {"x_d": x_d, "x_s": batch["x_s"].float()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--epoch", type=int, default=None,
                        help="Sensitivity only: replay a non-selected checkpoint (10 or 20).")
    parser.add_argument(
        "--pet-mode", choices=("zero", "climatology"), default="zero",
        help="Intervention PET forcing; climatology uses pre-test 2003-2018 GridMET.",
    )
    args = parser.parse_args()
    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    info = selection["architectures"][MODEL]
    epoch = int(info["selected_epoch"])
    sensitivity = args.epoch is not None and int(args.epoch) != epoch
    if args.epoch is not None:
        epoch = int(args.epoch)
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
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} gauges, found {len(gauges)}")
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Anchor order mismatch")

    q_curve = np.full((TARGET_GAUGES, 12, FUTURE_DAYS), np.nan, dtype=np.float64)
    storage_curve = np.full_like(q_curve, np.nan)
    records = []
    maximum_process_daily = 0.0
    maximum_total_daily = 0.0
    maximum_total_cumulative = 0.0
    maximum_historical_cumulative = 0.0
    maximum_route_difference = 0.0
    minimum_state = np.inf
    minimum_routing_buffer = np.inf
    missing_anchor_gauges = 0

    for basin_index, gauge in enumerate(gauges):
        indices = anchors["eval_time_index"][basin_index].astype(int)
        if np.any(indices < 0):
            missing_anchor_gauges += 1
            records.append({
                "GAGE_ID": gauge,
                "model": MODEL,
                "seed": args.seed,
                **summarize(q_curve[basin_index]),
            })
            continue
        dataset = tester._get_dataset(gauge)
        samples = [dataset[int(index)] for index in indices]
        batch = dataset.collate_fn(samples)
        if args.pet_mode == "climatology":
            base_dates = np.asarray(
                [sample["date"][-1] for sample in samples]
            ).astype("datetime64[D]")
            future_dates = (
                base_dates[:, None]
                + np.arange(1, FUTURE_DAYS + 1).astype("timedelta64[D]")[None, :]
            )
            pet = seasonal_pet_climatology(gauge, future_dates)
            original_extend_batch = helpers.extend_batch
            helpers.extend_batch = lambda supplied_batch, supplied_tester: extend_batch_with_pet(
                supplied_batch, supplied_tester, pet
            )
            try:
                result = helpers.run_intervention(model, tester, batch)
            finally:
                helpers.extend_batch = original_extend_batch
        else:
            result = helpers.run_intervention(model, tester, batch)
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
        if (basin_index + 1) % 100 == 0 or basin_index + 1 == TARGET_GAUGES:
            print(
                f"{MODEL} seed {args.seed}: {basin_index + 1}/{TARGET_GAUGES} gauges",
                flush=True,
            )

    if args.pet_mode == "climatology":
        output_dir = ROOT / "outputs/sensitivity_pet_climatology"
    else:
        output_dir = ROOT / ("outputs/sensitivity" if sensitivity else "outputs/frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{MODEL}_seed{args.seed}" + (f"_epoch{epoch:03d}" if sensitivity else "")
    if args.pet_mode == "climatology":
        stem += "_pet_climatology"
    curve_path = output_dir / f"{stem}_intervention_curves.npz"
    np.savez_compressed(
        curve_path,
        gauge_ids=np.asarray(gauges),
        quantiles=anchors["quantiles"],
        antecedent_time_index=anchors["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    fingerprint_path = output_dir / f"{stem}_fingerprints.csv"
    pd.DataFrame(records).to_csv(fingerprint_path, index=False)
    tolerance = 0.005
    receipt = {
        "model": MODEL,
        "seed": args.seed,
        "selected_epoch": epoch,
        "target_discharge_accessed": False,
        "target_gauges": TARGET_GAUGES,
        "gauges_with_complete_forcing_anchors": TARGET_GAUGES - missing_anchor_gauges,
        "gauges_without_forcing_anchors": missing_anchor_gauges,
        "anchors_per_eligible_gauge": 12,
        "forcing_anchor_archive": str(ANCHORS.relative_to(ROOT)),
        "forcing_anchor_archive_sha256": sha256(ANCHORS),
        "historical_window_days": 365,
        "intervention_days": FUTURE_DAYS,
        "intervention": (
            "P=0, PET=2003-2018 calendar-day GridMET climatology, T=10 C"
            if args.pet_mode == "climatology"
            else "P=0, PET=0, T=10 C"
        ),
        "pet_mode": args.pet_mode,
        "pet_climatology_period": (
            [CLIMATOLOGY_START, CLIMATOLOGY_END]
            if args.pet_mode == "climatology" else None
        ),
        "process_equations": "original HydroDL2 HBV 1.0 called one day at a time",
        "routing_equations": "original HydroDL2 uh_gamma and uh_conv",
        "closure_helper_source": str(REFERENCE_HELPERS),
        "closure_helper_source_sha256": sha256(REFERENCE_HELPERS),
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
    receipt["sensitivity_non_selected_checkpoint"] = bool(sensitivity)
    receipt_path = output_dir / f"{stem}_intervention_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
