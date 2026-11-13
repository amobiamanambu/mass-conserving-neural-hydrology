#!/usr/bin/env python3
"""Freeze official-model drainage fingerprints and numerical-closure audits.

This script never loads observed target discharge. It uses the pinned
NeuralHydrology model, its saved scaler and checkpoint, and the same 365-day
test windows used by the stock evaluator. Model arithmetic is replayed in
float64 for the closure audit and zero-input branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ANCHORS = ROOT / "data/source_inputs/reference_only/common_forcing_anchors.npz"
MODELS = ("hybrid_shm", "mclstm")
SEEDS = (11, 29, 47)
STATE_NAMES = ("ss", "sf", "su", "si", "sb")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_double_batch(batch: dict) -> dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = {name: tensor.double() for name, tensor in value.items()}
        elif torch.is_tensor(value) and value.is_floating_point():
            result[key] = value.double()
        else:
            result[key] = value
    return result


def mapped_shm_parameters(model, logits: torch.Tensor) -> dict[str, torch.Tensor]:
    parameters = {}
    for index, (name, bounds) in enumerate(model.conceptual_model.parameter_ranges.items()):
        lower, upper = [logits.new_tensor(value) for value in bounds]
        parameters[name] = lower + torch.sigmoid(logits[:, :, index]) * (upper - lower)
    return parameters


def run_shm(initial_state: torch.Tensor, parameters: dict[str, torch.Tensor],
            conceptual: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the pinned SHM equations from supplied states in float64."""
    ss, sf, su, si, sb = [initial_state[:, index] for index in range(5)]
    q_values, et_values, storage_values, state_values, residual_values = [], [], [], [], []
    for day in range(conceptual.shape[1]):
        before = ss + sf + su + si + sb
        p = torch.clamp(conceptual[:, day, 0], min=0.0)
        pet = torch.clamp(conceptual[:, day, 1], min=0.0)
        tmean = 0.5 * (conceptual[:, day, 2] + conceptual[:, day, 3])
        snow = torch.where(tmean < 0.0, p, torch.zeros_like(p))
        liquid = torch.where(tmean < 0.0, torch.zeros_like(p), p)
        melt = torch.clamp(tmean * parameters["dd"][:, day], min=0.0)
        qs_out = torch.minimum(ss, melt)
        ss = ss - qs_out + snow
        qsp_out = qs_out + liquid

        qf_in = torch.clamp(qsp_out - parameters["f_thr"][:, day], min=0.0)
        qu_in = torch.minimum(qsp_out, parameters["f_thr"][:, day])
        sf = sf + qf_in
        qf_out = sf / parameters["kf"][:, day]
        sf = sf - qf_out

        psi = (su / parameters["sumax"][:, day]) ** parameters["beta"][:, day]
        su_temp = su + qu_in * (1.0 - psi)
        su = torch.minimum(su_temp, parameters["sumax"][:, day])
        qu_out = qu_in * psi + torch.clamp(
            su_temp - parameters["sumax"][:, day], min=0.0
        )
        pwp = 0.8 * parameters["sumax"][:, day]
        ktheta = torch.where(
            su <= pwp, su / parameters["sumax"][:, day], torch.ones_like(su)
        )
        demanded_et = pet * 0.90 * ktheta
        actual_et = torch.minimum(su, demanded_et)
        su = su - actual_et

        qi_in = qu_out * parameters["perc"][:, day]
        si = si + qi_in
        qi_out = si / parameters["ki"][:, day]
        si = si - qi_out
        qb_in = qu_out * (1.0 - parameters["perc"][:, day])
        sb = sb + qb_in
        qb_out = sb / parameters["kb"][:, day]
        sb = sb - qb_out

        q = qf_out + qi_out + qb_out
        states = torch.stack([ss, sf, su, si, sb], dim=-1)
        after = states.sum(dim=-1)
        q_values.append(q)
        et_values.append(actual_et)
        storage_values.append(after)
        state_values.append(states)
        residual_values.append(before + p - actual_et - q - after)
    return {
        "q": torch.stack(q_values, dim=1),
        "et": torch.stack(et_values, dim=1),
        "storage": torch.stack(storage_values, dim=1),
        "states": torch.stack(state_values, dim=1),
        "daily_residual": torch.stack(residual_values, dim=1),
    }


def run_mclstm(model, initial_state: torch.Tensor, batch: dict) -> dict[str, torch.Tensor]:
    auxiliary = model.embedding_net(batch, concatenate_output=True)
    mass = torch.cat([batch["x_d"][key] for key in model.cfg.mass_inputs], dim=-1).transpose(0, 1)
    state = initial_state
    q_values, total_out_values, storage_values, state_values, residual_values = [], [], [], [], []
    for daily_mass, daily_auxiliary in zip(mass, auxiliary):
        before = state.sum(dim=-1)
        mass_out, state = model.mclstm._step(daily_mass, daily_auxiliary, state)
        total_out = mass_out.sum(dim=-1)
        after = state.sum(dim=-1)
        q_values.append(mass_out[:, 1:].sum(dim=-1))
        total_out_values.append(total_out)
        storage_values.append(after)
        state_values.append(state)
        residual_values.append(before + daily_mass[:, 0] - total_out - after)
    return {
        "q": torch.stack(q_values, dim=1),
        "total_out": torch.stack(total_out_values, dim=1),
        "storage": torch.stack(storage_values, dim=1),
        "states": torch.stack(state_values, dim=1),
        "daily_residual": torch.stack(residual_values, dim=1),
    }


def normalized_future_batch(tester, template: dict, base_dates: np.ndarray,
                            days: int = 91) -> dict:
    future_dates = (
        base_dates.astype("datetime64[D]")[:, None]
        + np.arange(1, days + 1).astype("timedelta64[D]")[None, :]
    )
    day_of_year = pd.DatetimeIndex(
        future_dates.reshape(-1).astype("datetime64[ns]")
    ).dayofyear.to_numpy().reshape(len(base_dates), days)
    raw = {
        "pr_mm": np.zeros((len(base_dates), days)),
        "pet_mm": np.zeros((len(base_dates), days)),
        "tmean_c": np.full((len(base_dates), days), 10.0),
        "sin_doy": np.sin(2.0 * np.pi * day_of_year / 365.25),
        "cos_doy": np.cos(2.0 * np.pi * day_of_year / 365.25),
    }
    center = tester.scaler["xarray_feature_center"]
    scale = tester.scaler["xarray_feature_scale"]
    x_d = {}
    for name in template["x_d"]:
        base = name.split("_copy", maxsplit=1)[0]
        values = raw[base]
        x_d[name] = torch.as_tensor(
            (values - float(center[name])) / float(scale[name]), dtype=torch.float64
        ).unsqueeze(-1)
    return {"x_d": x_d, "x_s": template["x_s"].double()}


def summarize(q_curve: np.ndarray, threshold: float = 0.001) -> dict[str, float | int]:
    q0 = q_curve[:, 0]
    valid = np.isfinite(q0) & (q0 >= threshold)
    r7 = np.full(len(q_curve), np.nan)
    r30 = np.full(len(q_curve), np.nan)
    r7[valid] = q_curve[valid, 7] / q0[valid]
    r30[valid] = q_curve[valid, 30] / q0[valid]
    return {
        "valid_anchors": int(valid.sum()),
        "median_q0_mm_day": float(np.nanmedian(np.where(valid, q0, np.nan))),
        "median_r7": float(np.nanmedian(r7)),
        "median_r30": float(np.nanmedian(r30)),
    }


def main() -> None:
    args = arguments()
    selection = json.loads((ROOT / "provenance" / "development_epoch_selection.json").read_text())
    architecture = selection["architectures"][args.model]
    epoch = int(architecture["selected_epoch"])
    run_dir = ROOT / architecture["runs"][str(args.seed)]["run_dir"]
    prediction_receipt = json.loads(
        (ROOT / "outputs" / "frozen" / f"{args.model}_seed{args.seed}_prediction_receipt.json").read_text()
    )
    if prediction_receipt.get("target_discharge_accessed") is not False:
        raise RuntimeError("Prediction freeze receipt does not preserve target blinding")

    cfg = Config(run_dir / "config.yml")
    tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
    tester._load_weights(epoch=epoch)
    model = tester.model.eval()
    anchors = np.load(SOURCE_ANCHORS)
    gauges = (ROOT / "data" / "basin_lists" / "test_413.txt").read_text().split()
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Forcing-anchor order does not match the target list")

    q_curve = np.full((len(gauges), 12, 91), np.nan, dtype=np.float64)
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

    for basin_index, gauge in enumerate(gauges):
        dataset = tester._get_dataset(gauge)
        sample_indices = anchors["eval_time_index"][basin_index].astype(int)
        samples = [dataset[int(index)] for index in sample_indices]
        batch_float = dataset.collate_fn(samples)
        with torch.no_grad():
            stock = model(batch_float)
        stock_q = stock["y_hat"].detach().cpu().numpy()[:, :, 0]

        model.double().eval()
        batch = as_double_batch(batch_float)
        with torch.no_grad():
            if args.model == "hybrid_shm":
                sequence = model.embedding_net(batch)
                lstm_output, (hidden, cell) = model.lstm(sequence)
                logits = model.linear(lstm_output.transpose(0, 1))
                parameters = mapped_shm_parameters(model, logits)
                conceptual = torch.cat(
                    [batch["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                )
                initial = conceptual.new_tensor([0.0, 1.0, 5.0, 10.0, 15.0]).repeat(12, 1)
                historical = run_shm(initial, parameters, conceptual)
                initial_state = historical["states"][:, -1]
            else:
                initial = batch["x_s"].new_zeros((12, cfg.hidden_size))
                historical = run_mclstm(model, initial, batch)
                initial_state = historical["states"][:, -1]
                hidden = cell = None

            max_float64_vs_stock_q = max(
                max_float64_vs_stock_q,
                float(np.max(np.abs(historical["q"].cpu().numpy() - stock_q))),
            )
            historical_residual = historical["daily_residual"]
            max_historical_daily = max(max_historical_daily, float(historical_residual.abs().max()))
            min_historical_signed = min(min_historical_signed, float(historical_residual.min()))
            max_historical_signed = max(max_historical_signed, float(historical_residual.max()))
            max_historical_cumulative = max(
                max_historical_cumulative,
                float(historical_residual.cumsum(dim=1).abs().max()),
            )
            negative_state_count += int((historical["states"] < -1e-10).sum())
            negative_flux_count += int((historical["q"] < -1e-10).sum())

            base_dates = np.asarray([sample["date"][-1] for sample in samples]).astype("datetime64[D]")
            future = normalized_future_batch(tester, batch, base_dates)
            if args.model == "hybrid_shm":
                future_sequence = model.embedding_net(future)
                future_lstm, _ = model.lstm(future_sequence, (hidden, cell))
                future_logits = model.linear(future_lstm.transpose(0, 1))
                future_parameters = mapped_shm_parameters(model, future_logits)
                future_conceptual = torch.cat(
                    [future["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                )
                intervention = run_shm(initial_state, future_parameters, future_conceptual)
            else:
                intervention = run_mclstm(model, initial_state, future)
            local_q = intervention["q"].cpu().numpy()
            local_storage = intervention["storage"].cpu().numpy()
            q_curve[basin_index] = local_q
            storage_curve[basin_index] = local_storage
            intervention_residual = intervention["daily_residual"]
            max_intervention_daily = max(max_intervention_daily, float(intervention_residual.abs().max()))
            min_intervention_signed = min(min_intervention_signed, float(intervention_residual.min()))
            max_intervention_signed = max(max_intervention_signed, float(intervention_residual.max()))
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
        if (basin_index + 1) % 25 == 0 or basin_index + 1 == len(gauges):
            print(f"{args.model} seed {args.seed}: {basin_index + 1}/{len(gauges)} gauges", flush=True)

    output_dir = ROOT / "outputs" / "frozen"
    archive = output_dir / f"{args.model}_seed{args.seed}_intervention_curves.npz"
    np.savez_compressed(
        archive,
        gauge_ids=np.asarray(gauges),
        quantiles=anchors["quantiles"],
        antecedent_time_index=anchors["eval_time_index"],
        q_mm_day=q_curve,
        storage_mm=storage_curve,
    )
    fingerprints = output_dir / f"{args.model}_seed{args.seed}_fingerprints.csv"
    pd.DataFrame(records).to_csv(fingerprints, index=False)
    closure_tolerance_mm = 1e-9
    receipt = {
        "model": args.model,
        "seed": args.seed,
        "selected_epoch": epoch,
        "target_discharge_accessed": False,
        "target_gauges": len(gauges),
        "anchors_per_gauge": 12,
        "historical_window_days": 365,
        "intervention_days": 91,
        "intervention": "P=0, PET=0, T=10 C; seasonal sine/cosine retained",
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
    receipt_path = output_dir / f"{args.model}_seed{args.seed}_intervention_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
