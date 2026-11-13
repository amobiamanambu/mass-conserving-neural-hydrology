#!/usr/bin/env python3
"""Freeze national native NeuralHydrology interventions and closure receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


REFERENCE_HELPERS = (
    ROOT / "vendor/reference_helpers/native_interventions.py"
)
spec = importlib.util.spec_from_file_location("frozen_native_intervention_helpers", REFERENCE_HELPERS)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load intervention helpers from {REFERENCE_HELPERS}")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

ANCHORS = ROOT / "data" / "forcing_only_anchors.npz"
MODELS = ("hybrid_shm", "mclstm")
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
        raise RuntimeError("Prediction freeze receipt does not preserve target blinding")

    cfg = Config(run_dir / "config.yml")
    tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
    tester._load_weights(epoch=epoch)
    model = tester.model.eval()
    anchors = np.load(ANCHORS)
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} gauges, found {len(gauges)}")
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Forcing-anchor order does not match the target list")

    q_curve = np.full((TARGET_GAUGES, 12, FUTURE_DAYS), np.nan, dtype=np.float64)
    storage_curve = np.full_like(q_curve, np.nan)
    records = []
    max_historical_daily = 0.0
    max_historical_cumulative = 0.0
    max_intervention_daily = 0.0
    max_intervention_cumulative = 0.0
    min_historical_signed = np.inf
    max_historical_signed = -np.inf
    min_intervention_signed = np.inf
    max_intervention_signed = -np.inf
    max_float64_vs_stock_q = 0.0
    negative_state_count = 0
    negative_flux_count = 0
    missing_anchor_gauges = 0

    for basin_index, gauge in enumerate(gauges):
        sample_indices = anchors["eval_time_index"][basin_index].astype(int)
        if np.any(sample_indices < 0):
            missing_anchor_gauges += 1
            records.append({
                "GAGE_ID": gauge,
                "model": args.model,
                "seed": args.seed,
                **summarize(q_curve[basin_index]),
            })
            continue
        dataset = tester._get_dataset(gauge)
        samples = [dataset[int(index)] for index in sample_indices]
        batch_float = dataset.collate_fn(samples)
        with torch.no_grad():
            stock = model(batch_float)
        stock_q = stock["y_hat"].detach().cpu().numpy()[:, :, 0]

        model.double().eval()
        batch = helpers.as_double_batch(batch_float)
        with torch.no_grad():
            if args.model == "hybrid_shm":
                sequence = model.embedding_net(batch)
                lstm_output, (hidden, cell) = model.lstm(sequence)
                logits = model.linear(lstm_output.transpose(0, 1))
                parameters = helpers.mapped_shm_parameters(model, logits)
                conceptual = torch.cat(
                    [batch["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                )
                initial = conceptual.new_tensor([0.0, 1.0, 5.0, 10.0, 15.0]).repeat(12, 1)
                historical = helpers.run_shm(initial, parameters, conceptual)
                initial_state = historical["states"][:, -1]
            else:
                initial = batch["x_s"].new_zeros((12, cfg.hidden_size))
                historical = helpers.run_mclstm(model, initial, batch)
                initial_state = historical["states"][:, -1]
                hidden = cell = None

            max_float64_vs_stock_q = max(
                max_float64_vs_stock_q,
                float(np.max(np.abs(historical["q"].cpu().numpy() - stock_q))),
            )
            historical_residual = historical["daily_residual"]
            max_historical_daily = max(
                max_historical_daily, float(historical_residual.abs().max())
            )
            min_historical_signed = min(
                min_historical_signed, float(historical_residual.min())
            )
            max_historical_signed = max(
                max_historical_signed, float(historical_residual.max())
            )
            max_historical_cumulative = max(
                max_historical_cumulative,
                float(historical_residual.cumsum(dim=1).abs().max()),
            )
            negative_state_count += int((historical["states"] < -1e-10).sum())
            negative_flux_count += int((historical["q"] < -1e-10).sum())

            base_dates = np.asarray(
                [sample["date"][-1] for sample in samples]
            ).astype("datetime64[D]")
            future = helpers.normalized_future_batch(
                tester, batch, base_dates, days=FUTURE_DAYS
            )
            if args.model == "hybrid_shm":
                future_sequence = model.embedding_net(future)
                future_lstm, _ = model.lstm(future_sequence, (hidden, cell))
                future_logits = model.linear(future_lstm.transpose(0, 1))
                future_parameters = helpers.mapped_shm_parameters(model, future_logits)
                future_conceptual = torch.cat(
                    [future["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                )
                intervention = helpers.run_shm(
                    initial_state, future_parameters, future_conceptual
                )
            else:
                intervention = helpers.run_mclstm(model, initial_state, future)
            q_curve[basin_index] = intervention["q"].cpu().numpy()
            storage_curve[basin_index] = intervention["storage"].cpu().numpy()
            intervention_residual = intervention["daily_residual"]
            max_intervention_daily = max(
                max_intervention_daily, float(intervention_residual.abs().max())
            )
            min_intervention_signed = min(
                min_intervention_signed, float(intervention_residual.min())
            )
            max_intervention_signed = max(
                max_intervention_signed, float(intervention_residual.max())
            )
            max_intervention_cumulative = max(
                max_intervention_cumulative,
                float(intervention_residual.cumsum(dim=1).abs().max()),
            )
            negative_state_count += int((intervention["states"] < -1e-10).sum())
            negative_flux_count += int((intervention["q"] < -1e-10).sum())
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
    archive = output_dir / f"{stem}_intervention_curves.npz"
    np.savez_compressed(
        archive,
        gauge_ids=np.asarray(gauges),
        quantiles=anchors["quantiles"],
        antecedent_time_index=anchors["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    fingerprints = output_dir / f"{stem}_fingerprints.csv"
    pd.DataFrame(records).to_csv(fingerprints, index=False)
    closure_tolerance_mm = 1e-9
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
        "intervention": "P=0, PET=0, T=10 C; seasonal sine/cosine retained",
        "closure_helper_source": str(REFERENCE_HELPERS),
        "closure_helper_source_sha256": sha256(REFERENCE_HELPERS),
        "closure_tolerance_mm": closure_tolerance_mm,
        "maximum_historical_daily_residual_mm": max_historical_daily,
        "minimum_historical_signed_daily_residual_mm": min_historical_signed,
        "maximum_historical_signed_daily_residual_mm": max_historical_signed,
        "maximum_historical_cumulative_drift_mm": max_historical_cumulative,
        "maximum_intervention_daily_residual_mm": max_intervention_daily,
        "minimum_intervention_signed_daily_residual_mm": min_intervention_signed,
        "maximum_intervention_signed_daily_residual_mm": max_intervention_signed,
        "maximum_intervention_cumulative_drift_mm": max_intervention_cumulative,
        "negative_state_count_below_minus_1e_10": negative_state_count,
        "negative_predicted_flux_count_below_minus_1e_10": negative_flux_count,
        "maximum_float64_replay_vs_stock_float32_q_difference_mm_day": max_float64_vs_stock_q,
        "passes_closure": bool(
            max_historical_daily <= closure_tolerance_mm
            and max_historical_cumulative <= closure_tolerance_mm
            and max_intervention_daily <= closure_tolerance_mm
            and max_intervention_cumulative <= closure_tolerance_mm
            and negative_state_count == 0
            and negative_flux_count == 0
        ),
        "curves": str(archive.relative_to(ROOT)),
        "curves_sha256": sha256(archive),
        "fingerprints": str(fingerprints.relative_to(ROOT)),
        "fingerprints_sha256": sha256(fingerprints),
    }
    receipt["sensitivity_non_selected_checkpoint"] = bool(sensitivity)
    receipt_path = output_dir / f"{stem}_intervention_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
