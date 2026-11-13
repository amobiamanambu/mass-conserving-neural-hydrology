#!/usr/bin/env python3
"""Diagnose internal zero-input dynamics for the three closure-eligible models.

This post-evaluation diagnostic uses the fixed KGE >= 0.60 intersection of
Hybrid-SHM, MCP-MA3, and the verified flux-gate port. It does not alter any
checkpoint, prediction, intervention fingerprint, or closure receipt. The
script replays the archived forcing-only antecedent states and summarizes
response-day 0--30 discharge, storage, water fate, and state retention.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402


HELPERS_PATH = ROOT / "vendor/reference_helpers/native_interventions.py"
SPEC = importlib.util.spec_from_file_location("native_helpers", HELPERS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load intervention helpers from {HELPERS_PATH}")
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)

MODELS = ("hybrid_shm", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
THRESHOLD = 0.60
DAYS = 91
DISPLAY_DAYS = 31
STATE_NAMES = {
    "hybrid_shm": ("Snow", "Fast", "Unsaturated", "Intermediate", "Base"),
    "author_mcp_ma3": ("Main", "Routing"),
    "ported_fluxgate": ("Snow", "Soil", "Fast", "Slow", "Routing"),
}


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    return tuple(float(x) for x in np.quantile(values, [0.25, 0.50, 0.75]))


def future_adapter_batch(model_name: str, x_s: torch.Tensor, dates: np.ndarray) -> dict:
    if model_name == "author_mcp_ma3":
        zeros = torch.zeros((len(dates), DAYS, 1), dtype=torch.float64)
        return {
            "x_d": {"pr_mm_copy1": zeros, "pet_mm_copy1": zeros.clone()},
            "x_s": x_s.double(),
        }
    future_dates = (
        dates.astype("datetime64[D]")[:, None]
        + np.arange(1, DAYS + 1).astype("timedelta64[D]")[None, :]
    )
    doy = pd.DatetimeIndex(future_dates.reshape(-1).astype("datetime64[ns]")).dayofyear
    doy = doy.to_numpy().reshape(len(dates), DAYS)
    raw = {
        "pr_mm": np.zeros((len(dates), DAYS)),
        "pet_mm": np.zeros((len(dates), DAYS)),
        "tmean_c": np.full((len(dates), DAYS), 10.0),
        "sin_doy": np.sin(2.0 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2.0 * np.pi * doy / 365.25),
    }
    return {
        "x_d": {
            key: torch.as_tensor(value, dtype=torch.float64).unsqueeze(-1)
            for key, value in raw.items()
        },
        "x_s": x_s.double(),
    }


def run_shm_diagnostic(initial_state, parameters, conceptual):
    ss, sf, su, si, sb = [initial_state[:, index] for index in range(5)]
    states, q_values, et_values = [], [], []
    q_fast_values, q_intermediate_values, q_base_values = [], [], []
    for day in range(conceptual.shape[1]):
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
        q_fast = sf / parameters["kf"][:, day]
        sf = sf - q_fast

        psi = (su / parameters["sumax"][:, day]) ** parameters["beta"][:, day]
        su_temp = su + qu_in * (1.0 - psi)
        su = torch.minimum(su_temp, parameters["sumax"][:, day])
        qu_out = qu_in * psi + torch.clamp(su_temp - parameters["sumax"][:, day], min=0.0)
        pwp = 0.8 * parameters["sumax"][:, day]
        ktheta = torch.where(su <= pwp, su / parameters["sumax"][:, day], torch.ones_like(su))
        et = torch.minimum(su, pet * 0.90 * ktheta)
        su = su - et

        qi_in = qu_out * parameters["perc"][:, day]
        si = si + qi_in
        q_intermediate = si / parameters["ki"][:, day]
        si = si - q_intermediate
        qb_in = qu_out * (1.0 - parameters["perc"][:, day])
        sb = sb + qb_in
        q_base = sb / parameters["kb"][:, day]
        sb = sb - q_base

        q = q_fast + q_intermediate + q_base
        states.append(torch.stack([ss, sf, su, si, sb], dim=-1))
        q_values.append(q)
        et_values.append(et)
        q_fast_values.append(q_fast)
        q_intermediate_values.append(q_intermediate)
        q_base_values.append(q_base)
    return {
        "q": torch.stack(q_values, dim=1),
        "loss": torch.zeros_like(torch.stack(q_values, dim=1)),
        "state": torch.stack(states, dim=1),
        "process": {
            "Fast release": torch.stack(q_fast_values, dim=1),
            "Intermediate release": torch.stack(q_intermediate_values, dim=1),
            "Base release": torch.stack(q_base_values, dim=1),
        },
    }


def run_fluxgate_diagnostic(model, data):
    forcing = torch.cat(
        [data["x_d"][name] for name in ("pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy")],
        dim=-1,
    )
    embedding, params = model.static_terms(data["x_s"])
    snow, soil, fast, slow, route = data["initial_state"].unbind(dim=1)
    states, q_values, loss_values = [], [], []
    recharge_values, q_fast_values, q_slow_values = [], [], []
    for day in range(forcing.shape[1]):
        p = torch.clamp(forcing[:, day, 0], min=0.0)
        pet = torch.clamp(forcing[:, day, 1], min=0.0)
        temperature = forcing[:, day, 2]
        snow_fraction = torch.sigmoid((params["t0"] - temperature) * 2.0)
        snowfall = p * snow_fraction
        rain = p - snowfall
        available_snow = snow + snowfall
        melt = torch.minimum(
            available_snow, F.relu(temperature - params["t0"]) * params["degree_day"]
        )
        snow = available_snow - melt
        liquid = rain + melt
        gate_input = torch.cat([
            (torch.log1p(p) / 4.0)[:, None],
            (torch.log1p(pet) / 2.0)[:, None],
            (temperature / 20.0)[:, None],
            forcing[:, day, 3:5],
            torch.clamp(soil / params["capacity"], 0.0, 2.0)[:, None],
            (torch.log1p(fast) / 5.0)[:, None],
            (torch.log1p(slow) / 5.0)[:, None],
            (torch.log1p(route) / 5.0)[:, None],
            embedding,
        ], dim=1)
        gate = model.gates(gate_input)
        saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
        direct = liquid * torch.sigmoid(gate[:, 0]) * torch.sqrt(saturation + 1e-6)
        soil = soil + liquid - direct
        evap_saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
        evap = torch.minimum(
            soil,
            pet * (0.25 + 1.35 * torch.sigmoid(gate[:, 1]))
            * torch.sqrt(evap_saturation + 1e-6),
        )
        soil = soil - evap
        recharge = soil * (0.00005 + 0.08 * torch.sigmoid(gate[:, 2]))
        soil = soil - recharge
        deep_loss = soil * (0.000001 + 0.002 * torch.sigmoid(gate[:, 3]))
        soil = soil - deep_loss
        overflow = F.relu(soil - params["capacity"])
        soil = torch.minimum(soil, params["capacity"])
        fast_share = torch.sigmoid(gate[:, 4])
        fast = fast + direct + overflow + recharge * fast_share
        slow = slow + recharge * (1.0 - fast_share)
        q_fast = fast * (0.005 + 0.895 * torch.sigmoid(gate[:, 5]))
        q_slow = slow * (0.0001 + 0.12 * torch.sigmoid(gate[:, 6]))
        fast = fast - q_fast
        slow = slow - q_slow
        route = route + q_fast + q_slow
        q = route * (0.01 + 0.89 * torch.sigmoid(gate[:, 7]))
        route = route - q
        states.append(torch.stack([snow, soil, fast, slow, route], dim=1))
        q_values.append(q)
        loss_values.append(deep_loss)
        recharge_values.append(recharge)
        q_fast_values.append(q_fast)
        q_slow_values.append(q_slow)
    return {
        "q": torch.stack(q_values, dim=1),
        "loss": torch.stack(loss_values, dim=1),
        "state": torch.stack(states, dim=1),
        "process": {
            "Soil recharge": torch.stack(recharge_values, dim=1),
            "Fast release to routing": torch.stack(q_fast_values, dim=1),
            "Slow release to routing": torch.stack(q_slow_values, dim=1),
        },
    }


def append_summaries(model_name, seed, gauge, q, loss, state, processes, trajectory, states, process_rows):
    q = q.detach().cpu().numpy()
    loss = loss.detach().cpu().numpy()
    state = state.detach().cpu().numpy()
    storage = state.sum(axis=-1)
    q0 = q[:, 0]
    s0 = storage[:, 0]
    valid = np.isfinite(q0) & (q0 >= 0.001) & np.isfinite(s0) & (s0 > 1e-9)
    for anchor in np.flatnonzero(valid):
        for day in range(DISPLAY_DAYS):
            trajectory.append({
                "model": model_name,
                "seed": seed,
                "GAGE_ID": gauge,
                "anchor": int(anchor + 1),
                "day": day,
                "q_ratio": q[anchor, day] / q0[anchor],
                "storage_ratio": storage[anchor, day] / s0[anchor],
                "cumulative_q_fraction": q[anchor, 1:day + 1].sum() / s0[anchor] if day else 0.0,
                "cumulative_loss_fraction": loss[anchor, 1:day + 1].sum() / s0[anchor] if day else 0.0,
            })
        for index, state_name in enumerate(STATE_NAMES[model_name]):
            start = state[anchor, 0, index]
            if np.isfinite(start) and start > 1e-9:
                states.append({
                    "model": model_name,
                    "seed": seed,
                    "GAGE_ID": gauge,
                    "anchor": int(anchor + 1),
                    "state": state_name,
                    "day30_retention": state[anchor, 30, index] / start,
                    "day0_fraction_total": start / s0[anchor],
                })
        for process_name, values in processes.items():
            arr = values.detach().cpu().numpy()
            process_rows.append({
                "model": model_name,
                "seed": seed,
                "GAGE_ID": gauge,
                "anchor": int(anchor + 1),
                "process": process_name,
                "day1_30_fraction_initial_storage": arr[anchor, 1:31].sum() / s0[anchor],
            })


def main() -> None:
    metrics = pd.read_csv(
        ROOT / "outputs/evaluation/national_gauge_metrics.csv.gz", dtype={"GAGE_ID": str}
    )
    common = metrics[
        metrics[[f"{model}_kge" for model in MODELS]].ge(THRESHOLD).all(axis=1)
    ].copy()
    selected = set(common.GAGE_ID.astype(str).str.zfill(8))
    if len(selected) != 474:
        raise RuntimeError(f"Expected 474 common high-skill gauges, found {len(selected)}")

    anchors = np.load(ROOT / "data/forcing_only_anchors.npz")
    all_gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    trajectory, state_rows, process_rows = [], [], []
    parity_max = 0.0

    for model_name in MODELS:
        for seed in SEEDS:
            info = selection["architectures"][model_name]
            epoch = int(info["selected_epoch"])
            run_dir = ROOT / info["runs"][str(seed)]["run_dir"]
            cfg = Config(run_dir / "config.yml")
            tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
            tester._load_weights(epoch=epoch)
            model = tester.model.double().eval()
            for index, gauge in enumerate(all_gauges):
                if gauge not in selected:
                    continue
                sample_indices = anchors["eval_time_index"][index].astype(int)
                dataset = tester._get_dataset(gauge)
                samples = [dataset[int(item)] for item in sample_indices]
                batch = HELPERS.as_double_batch(dataset.collate_fn(samples))
                base_dates = np.asarray([sample["date"][-1] for sample in samples]).astype("datetime64[D]")
                with torch.no_grad():
                    if model_name == "hybrid_shm":
                        sequence = model.embedding_net(batch)
                        lstm_output, (hidden, cell) = model.lstm(sequence)
                        parameters = HELPERS.mapped_shm_parameters(
                            model, model.linear(lstm_output.transpose(0, 1))
                        )
                        conceptual = torch.cat(
                            [batch["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                        )
                        initial = conceptual.new_tensor([0.0, 1.0, 5.0, 10.0, 15.0]).repeat(12, 1)
                        historical = HELPERS.run_shm(initial, parameters, conceptual)
                        initial_state = historical["states"][:, -1]
                        future = HELPERS.normalized_future_batch(tester, batch, base_dates, days=DAYS)
                        future_sequence = model.embedding_net(future)
                        future_lstm, _ = model.lstm(future_sequence, (hidden, cell))
                        future_parameters = HELPERS.mapped_shm_parameters(
                            model, model.linear(future_lstm.transpose(0, 1))
                        )
                        future_conceptual = torch.cat(
                            [future["x_d"][key] for key in cfg.dynamic_conceptual_inputs], dim=-1
                        )
                        result = run_shm_diagnostic(initial_state, future_parameters, future_conceptual)
                    else:
                        historical = model(batch)
                        initial_state = historical["state"][:, -1]
                        future = future_adapter_batch(model_name, batch["x_s"], base_dates)
                        future["initial_state"] = initial_state
                        stock = model(future)
                        if model_name == "author_mcp_ma3":
                            result = {
                                "q": stock["q"],
                                "loss": stock["loss"],
                                "state": stock["state"],
                                "process": {"Main-to-routing transfer": stock["main_q"]},
                            }
                        else:
                            result = run_fluxgate_diagnostic(model, future)
                            parity_max = max(
                                parity_max,
                                float((result["q"] - stock["q"]).abs().max()),
                                float((result["state"] - stock["state"]).abs().max()),
                                float((result["loss"] - stock["loss"]).abs().max()),
                            )
                append_summaries(
                    model_name, seed, gauge, result["q"], result["loss"], result["state"],
                    result["process"], trajectory, state_rows, process_rows,
                )
            print(f"completed {model_name} seed {seed}", flush=True)

    if parity_max > 1e-10:
        raise RuntimeError(f"Flux-gate diagnostic replay parity failed: {parity_max}")
    output = ROOT / "outputs/evaluation"
    trajectory_frame = pd.DataFrame(trajectory)
    state_frame = pd.DataFrame(state_rows)
    process_frame = pd.DataFrame(process_rows)
    trajectory_frame.to_csv(output / "internal_process_trajectory_gauge.csv.gz", index=False)
    state_frame.to_csv(output / "internal_state_retention_gauge.csv.gz", index=False)
    process_frame.to_csv(output / "internal_process_flux_gauge.csv.gz", index=False)

    summary_rows = []
    for (model_name, day), group in trajectory_frame.groupby(["model", "day"]):
        for metric in ("q_ratio", "storage_ratio", "cumulative_q_fraction", "cumulative_loss_fraction"):
            q25, median, q75 = quantiles(group[metric].to_numpy())
            summary_rows.append({
                "model": model_name, "day": day, "metric": metric,
                "q25": q25, "median": median, "q75": q75,
                "n": int(group[metric].notna().sum()),
            })
    pd.DataFrame(summary_rows).to_csv(output / "internal_process_trajectory_summary.csv", index=False)

    state_summary = []
    for (model_name, state_name), group in state_frame.groupby(["model", "state"], sort=False):
        q25, median, q75 = quantiles(group.day30_retention.to_numpy())
        fq25, fmedian, fq75 = quantiles(group.day0_fraction_total.to_numpy())
        state_summary.append({
            "model": model_name, "state": state_name, "q25": q25, "median": median,
            "q75": q75, "day0_fraction_q25": fq25, "day0_fraction_median": fmedian,
            "day0_fraction_q75": fq75, "n": int(len(group)),
        })
    pd.DataFrame(state_summary).to_csv(output / "internal_state_retention_summary.csv", index=False)

    process_summary = []
    for (model_name, process_name), group in process_frame.groupby(["model", "process"], sort=False):
        q25, median, q75 = quantiles(group.day1_30_fraction_initial_storage.to_numpy())
        process_summary.append({
            "model": model_name, "process": process_name, "q25": q25,
            "median": median, "q75": q75, "n": int(len(group)),
        })
    pd.DataFrame(process_summary).to_csv(output / "internal_process_flux_summary.csv", index=False)
    receipt = {
        "common_high_skill_gauges": len(selected),
        "skill_threshold_for_all_three_models": THRESHOLD,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "forcing_anchors_per_gauge": 12,
        "display_days": "response days 0 through 30",
        "diagnostic_fluxgate_replay_max_absolute_difference": parity_max,
        "target_discharge_used_only_for_post_evaluation_skill_conditioning": True,
        "checkpoints_predictions_and_fingerprints_modified": False,
    }
    (output / "internal_process_diagnostic_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
