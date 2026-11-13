#!/usr/bin/env python3
"""Freeze national MCP-MA3 and flux-gate interventions with float64 closure audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402


ANCHORS = ROOT / "data" / "forcing_only_anchors.npz"
MODELS = ("author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
FUTURE_DAYS = 91


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def double_batch(batch: dict) -> dict:
    return {
        "x_d": {name: value.double() for name, value in batch["x_d"].items()},
        "x_s": batch["x_s"].double(),
    }


def future_batch(model_name: str, x_s: torch.Tensor, base_dates: np.ndarray) -> dict:
    batch_size = len(base_dates)
    if model_name == "author_mcp_ma3":
        zeros = torch.zeros((batch_size, FUTURE_DAYS, 1), dtype=torch.float64)
        return {
            "x_d": {"pr_mm_copy1": zeros, "pet_mm_copy1": zeros.clone()},
            "x_s": x_s.double(),
        }
    dates = (
        base_dates.astype("datetime64[D]")[:, None]
        + np.arange(1, FUTURE_DAYS + 1).astype("timedelta64[D]")[None, :]
    )
    day_of_year = pd.DatetimeIndex(
        dates.reshape(-1).astype("datetime64[ns]")
    ).dayofyear.to_numpy().reshape(batch_size, FUTURE_DAYS)
    raw = {
        "pr_mm": np.zeros((batch_size, FUTURE_DAYS)),
        "pet_mm": np.zeros((batch_size, FUTURE_DAYS)),
        "tmean_c": np.full((batch_size, FUTURE_DAYS), 10.0),
        "sin_doy": np.sin(2.0 * np.pi * day_of_year / 365.25),
        "cos_doy": np.cos(2.0 * np.pi * day_of_year / 365.25),
    }
    return {
        "x_d": {
            name: torch.as_tensor(values, dtype=torch.float64).unsqueeze(-1)
            for name, values in raw.items()
        },
        "x_s": x_s.double(),
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
        "median_q0_mm_day": float(np.nanmedian(q0[valid])) if valid.any() else np.nan,
        "median_r7": float(np.nanmedian(r7[valid])) if valid.any() else np.nan,
        "median_r30": float(np.nanmedian(r30[valid])) if valid.any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--epoch", type=int, default=None,
                        help="Sensitivity only: replay a non-selected checkpoint (10 or 20).")
    args = parser.parse_args()

    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    info = selection["architectures"][args.model]
    epoch = int(info["selected_epoch"])
    sensitivity = args.epoch is not None and int(args.epoch) != epoch
    if args.epoch is not None:
        epoch = int(args.epoch)
    run_dir = ROOT / info["runs"][str(args.seed)]["run_dir"]
    prediction_receipt = json.loads(
        (ROOT / f"outputs/frozen/{args.model}_seed{args.seed}_prediction_receipt.json").read_text()
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
    maxima = {
        "historical_daily": 0.0,
        "historical_cumulative": 0.0,
        "intervention_daily": 0.0,
        "intervention_cumulative": 0.0,
        "replay_difference": 0.0,
    }
    signed = {
        "historical_min": np.inf,
        "historical_max": -np.inf,
        "intervention_min": np.inf,
        "intervention_max": -np.inf,
    }
    negative_state_count = 0
    negative_flux_count = 0
    missing_anchor_gauges = 0

    for basin_index, gauge in enumerate(gauges):
        indices = anchors["eval_time_index"][basin_index].astype(int)
        if np.any(indices < 0):
            missing_anchor_gauges += 1
            records.append({
                "GAGE_ID": gauge,
                "model": args.model,
                "seed": args.seed,
                **summarize(q_curve[basin_index]),
            })
            continue
        dataset = tester._get_dataset(gauge)
        samples = [dataset[int(index)] for index in indices]
        batch_float = dataset.collate_fn(samples)
        with torch.no_grad():
            stock = model(batch_float)
        stock_q = stock["y_hat"][:, :, 0].cpu().numpy()

        model.double().eval()
        batch = double_batch(batch_float)
        with torch.no_grad():
            historical = model(batch)
            maxima["replay_difference"] = max(
                maxima["replay_difference"],
                float(np.max(np.abs(stock_q - historical["q"].cpu().numpy()))),
            )
            residual = historical["daily_residual"]
            signed["historical_min"] = min(signed["historical_min"], float(residual.min()))
            signed["historical_max"] = max(signed["historical_max"], float(residual.max()))
            maxima["historical_daily"] = max(
                maxima["historical_daily"], float(residual.abs().max())
            )
            maxima["historical_cumulative"] = max(
                maxima["historical_cumulative"],
                float(residual.cumsum(dim=1).abs().max()),
            )
            initial_state = historical["state"][:, -1]
            base_dates = np.asarray(
                [sample["date"][-1] for sample in samples]
            ).astype("datetime64[D]")
            future = future_batch(args.model, batch["x_s"], base_dates)
            future["initial_state"] = initial_state
            intervention = model(future)
            q_curve[basin_index] = intervention["q"].cpu().numpy()
            storage_curve[basin_index] = intervention["storage"].cpu().numpy()
            residual = intervention["daily_residual"]
            signed["intervention_min"] = min(
                signed["intervention_min"], float(residual.min())
            )
            signed["intervention_max"] = max(
                signed["intervention_max"], float(residual.max())
            )
            maxima["intervention_daily"] = max(
                maxima["intervention_daily"], float(residual.abs().max())
            )
            maxima["intervention_cumulative"] = max(
                maxima["intervention_cumulative"],
                float(residual.cumsum(dim=1).abs().max()),
            )
            negative_state_count += int((historical["state"] < -1e-10).sum())
            negative_state_count += int((intervention["state"] < -1e-10).sum())
            for output in (historical, intervention):
                for flux in ("q", "et", "loss"):
                    negative_flux_count += int((output[flux] < -1e-10).sum())
        model.float().eval()
        records.append({
            "GAGE_ID": gauge,
            "model": args.model,
            "seed": args.seed,
            **summarize(q_curve[basin_index]),
        })
        if (basin_index + 1) % 100 == 0 or basin_index + 1 == TARGET_GAUGES:
            print(
                f"{args.model} seed {args.seed}: {basin_index + 1}/{TARGET_GAUGES} gauges",
                flush=True,
            )

    output_dir = ROOT / ("outputs/sensitivity" if sensitivity else "outputs/frozen")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_seed{args.seed}" + (f"_epoch{epoch:03d}" if sensitivity else "")
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
    tolerance = 1e-9
    receipt = {
        "model": args.model,
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
        "intervention": "P=0, PET=0, T=10 C; seasonal sine/cosine retained where used",
        "closure_tolerance_mm": tolerance,
        "maximum_historical_daily_residual_mm": maxima["historical_daily"],
        "minimum_historical_signed_daily_residual_mm": signed["historical_min"],
        "maximum_historical_signed_daily_residual_mm": signed["historical_max"],
        "maximum_historical_cumulative_drift_mm": maxima["historical_cumulative"],
        "maximum_intervention_daily_residual_mm": maxima["intervention_daily"],
        "minimum_intervention_signed_daily_residual_mm": signed["intervention_min"],
        "maximum_intervention_signed_daily_residual_mm": signed["intervention_max"],
        "maximum_intervention_cumulative_drift_mm": maxima["intervention_cumulative"],
        "maximum_float64_replay_vs_stock_float32_q_difference_mm_day": maxima["replay_difference"],
        "negative_state_count_below_minus_1e_10": negative_state_count,
        "negative_flux_count_below_minus_1e_10": negative_flux_count,
        "passes_closure": bool(
            max(
                maxima["historical_daily"],
                maxima["historical_cumulative"],
                maxima["intervention_daily"],
                maxima["intervention_cumulative"],
            ) <= tolerance
            and negative_state_count == 0
            and negative_flux_count == 0
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
