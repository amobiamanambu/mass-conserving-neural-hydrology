#!/usr/bin/env python3
"""Select, refit, predict, and freeze one reference-only conserved model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parents[1]
PRIMARY_MODELS = ("dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr")
MODEL_SETTINGS = {
    "dhbv": {"epochs": 18, "learning_rate": 0.0020},
    "fluxgate": {"epochs": 24, "learning_rate": 0.0015},
    "mcp_ma3": {"epochs": 72, "learning_rate": 0.0015},
    "mc_node": {"epochs": 48, "learning_rate": 0.0010},
    "mclstm_dr": {"epochs": 48, "learning_rate": 0.0010},
}
FAMILIES = {
    "dhbv": "neural-parameterized dHBV",
    "fluxgate": "five-store neural flux gate",
    "mcp_ma3": "mass-conserving perceptron MA3",
    "mc_node": "conservative neural ODE",
    "mclstm_dr": "mass-conserving LSTM with dynamic redistribution",
}


sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "multimodel_neural_conservation_stress_test_2026-10-03/scripts"))
import hydro_models  # noqa: E402
import conserved_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["select", "refit"], required=True)
    parser.add_argument("--model", choices=PRIMARY_MODELS, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def make_model(name: str, n_static: int) -> torch.nn.Module:
    if name in {"dhbv", "fluxgate"}:
        return hydro_models.make_model(name, n_static)
    return conserved_models.make_model(name, n_static)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_moments(values: torch.Tensor, mask: torch.Tensor):
    count = mask.sum(dim=1).clamp(min=2)
    mean = (values * mask).sum(dim=1) / count
    centered = (values - mean[:, None]) * mask
    variance = centered.square().sum(dim=1) / (count - 1)
    return mean, torch.sqrt(variance + 1e-6), count


def discharge_loss(prediction: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(observation).to(prediction.dtype)
    observed = torch.nan_to_num(observation, nan=0.0)
    pred_mean, pred_std, count = masked_moments(prediction, mask)
    obs_mean, obs_std, _ = masked_moments(observed, mask)

    def normalized_error(predicted, observed_value):
        observed_mean = (observed_value * mask).sum(dim=1) / count
        error = ((predicted - observed_value).square() * mask).sum(dim=1)
        variance = (((observed_value - observed_mean[:, None]) * mask).square()).sum(dim=1)
        return error / (variance + 0.05 * count)

    raw_error = normalized_error(prediction, observed)
    sqrt_error = normalized_error(torch.sqrt(prediction + 0.01), torch.sqrt(observed + 0.01))
    log_error = normalized_error(torch.log1p(prediction), torch.log1p(observed))
    bias = ((pred_mean - obs_mean) / (obs_mean + 0.10)).square()
    spread = torch.log((pred_std + 0.05) / (obs_std + 0.05)).square()
    return (
        0.35 * raw_error
        + 0.30 * sqrt_error
        + 0.10 * log_error
        + 0.10 * bias
        + 0.15 * spread
    ).mean()


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o, p = observed[valid], predicted[valid]
    if np.std(o) <= 0 or np.mean(o) <= 0:
        return np.nan
    r = np.corrcoef(o, p)[0, 1]
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(p) / np.mean(o)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def load_et(stations: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    table = pd.read_csv(EXPERIMENT / "data/donor_ssebop_et_2003_2018.csv.gz", dtype={"GAGE_ID": str})
    years = sorted(table.year.unique())
    donors = stations.loc[stations.role.eq("donor"), "GAGE_ID"].tolist()
    pivot = table.pivot(index="GAGE_ID", columns="year", values="ssebop_apr_oct_et_mm").reindex(index=donors, columns=years)
    if pivot.isna().any().any():
        raise RuntimeError("Reference donor ET matrix contains missing values")
    return pivot.to_numpy(np.float32), {int(year): index for index, year in enumerate(years)}


def evaluate_development(model, forcing, static, q, stations, dates, batch_size):
    development = stations.loc[stations.role.eq("development")]
    rows = development.row.to_numpy(int)
    q_rows = development.q_row.to_numpy(int)
    stop = int(np.searchsorted(dates, np.datetime64("2018-12-31"), side="right"))
    start = int(np.searchsorted(dates, np.datetime64("2015-01-01")))
    predictions = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(rows), batch_size):
            chosen = rows[first : first + batch_size]
            output = model(
                torch.from_numpy(np.asarray(forcing[chosen, :stop])).float(),
                torch.from_numpy(np.asarray(static[chosen])).float(),
            )
            predictions.append(output["q"][:, start:stop].cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    observation = np.asarray(q[np.ix_(q_rows, np.arange(start, stop))])
    records = [
        {"GAGE_ID": row.GAGE_ID, "kge": kge(observation[index], prediction[index])}
        for index, row in enumerate(development.itertuples(index=False))
    ]
    return pd.DataFrame(records), prediction


def train_epoch(model, optimizer, forcing, static, q, dates, donor_rows, donor_q_rows, et, year_to_et, years, batch_size):
    model.train()
    shuffled = np.random.permutation(len(donor_rows))
    losses, q_losses, et_losses = [], [], []
    for first in range(0, len(shuffled), batch_size):
        local = shuffled[first : first + batch_size]
        rows = donor_rows[local]
        q_rows = donor_q_rows[local]
        year = int(np.random.choice(years))
        start = int(np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01")))
        stop = int(np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right"))
        target_start = int(np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01")))
        x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
        s = torch.from_numpy(np.asarray(static[rows])).float()
        observed = torch.from_numpy(np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])).float()
        result = model(x, s)
        q_loss = discharge_loss(result["q"][:, target_start:], observed)
        season = (dates[start:stop] >= np.datetime64(f"{year}-04-01")) & (dates[start:stop] <= np.datetime64(f"{year}-10-31"))
        predicted_et = result["et"][:, torch.from_numpy(season)].sum(dim=1)
        target_et = torch.from_numpy(et[local, year_to_et[year]]).float()
        scaled = (predicted_et - target_et) / (0.20 * target_et + 50.0)
        et_loss = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled))
        loss = q_loss + 0.20 * et_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        q_losses.append(float(q_loss.detach()))
        et_losses.append(float(et_loss.detach()))
    return float(np.mean(losses)), float(np.mean(q_losses)), float(np.mean(et_losses))


def mass_audit(model, forcing, static, donor_rows) -> dict[str, float]:
    dtype = next(model.parameters()).dtype
    model = model.double().eval()
    with torch.no_grad():
        result = model(
            torch.from_numpy(np.asarray(forcing[donor_rows[:4], :730])).double(),
            torch.from_numpy(np.asarray(static[donor_rows[:4]])).double(),
            return_states=True,
        )
    audit = {
        "daily_max_abs_mm": float(result["daily_residual"].abs().max()),
        "cumulative_max_abs_mm": float(result["cumulative_residual"].abs().max()),
    }
    model.to(dtype=dtype)
    return audit


def select_storage_anchors(state: np.ndarray, forcing: np.ndarray, eval_dates: np.ndarray):
    quantiles = np.linspace(0.05, 0.95, 12)
    selected_state = np.empty((len(state), len(quantiles), state.shape[2]), dtype=np.float32)
    selected_time = np.empty((len(state), len(quantiles)), dtype=np.int16)
    month = pd.DatetimeIndex(eval_dates.astype("datetime64[ns]")).month.to_numpy()
    for basin in range(len(state)):
        storage = state[basin].sum(axis=1)
        candidate = np.flatnonzero((month >= 6) & (month <= 9) & (forcing[basin, :, 2] > 5.0) & np.isfinite(storage))
        targets = np.quantile(storage[candidate], quantiles)
        used: set[int] = set()
        chosen: list[int] = []
        for value in targets:
            order = candidate[np.argsort(np.abs(storage[candidate] - value), kind="stable")]
            pick = next((int(item) for item in order if int(item) not in used), int(order[0]))
            used.add(pick)
            chosen.append(pick)
        selected_time[basin] = chosen
        selected_state[basin] = state[basin, chosen]
    return selected_state, selected_time


def run_selection(args, model, forcing, static, q, stations, dates, et, year_to_et):
    if args.seed != 11:
        raise ValueError("Stopping-epoch selection is frozen from seed 11 only")
    setting = MODEL_SETTINGS[args.model]
    run_dir = EXPERIMENT / f"runs/development/{args.model}_qet_seed11"
    run_dir.mkdir(parents=True, exist_ok=True)
    donor = stations.loc[stations.role.eq("donor")]
    donor_rows, donor_q_rows = donor.row.to_numpy(int), donor.q_row.to_numpy(int)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=setting["epochs"], eta_min=setting["learning_rate"] / 20.0
    )
    history, best_score, best_epoch = [], -np.inf, None
    for epoch in range(1, setting["epochs"] + 1):
        loss, q_loss, et_loss = train_epoch(
            model, optimizer, forcing, static, q, dates, donor_rows, donor_q_rows,
            et, year_to_et, np.arange(2005, 2015), args.batch_size,
        )
        scheduler.step()
        median_kge = np.nan
        if epoch % 3 == 0 or epoch == setting["epochs"]:
            metrics, prediction = evaluate_development(model, forcing, static, q, stations, dates, args.batch_size)
            median_kge = float(metrics.kge.median())
            if median_kge > best_score:
                best_score, best_epoch = median_kge, epoch
                torch.save({"model_state": model.state_dict(), "model": args.model, "epoch": epoch}, run_dir / "best_model.pt")
                metrics.to_csv(run_dir / "development_metrics.csv", index=False)
                np.savez_compressed(run_dir / "development_predictions.npz", predicted=prediction)
        record = {
            "epoch": epoch, "training_loss": loss, "training_q_loss": q_loss,
            "training_et_loss": et_loss, "development_median_kge": median_kge,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps({"run": f"{args.model}_select", **record}), flush=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    summary = {
        "model": args.model, "family": FAMILIES[args.model], "best_epoch": int(best_epoch),
        "best_development_median_kge": float(best_score), "maximum_epochs": setting["epochs"],
        "donor_gauges": 400, "development_gauges": 100, "nonreference_gauges": 0,
        "mass_balance": mass_audit(model, forcing, static, donor_rows),
        "target_discharge_accessed": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def run_refit(args, model, forcing, static, q, stations, dates, et, year_to_et):
    summary_path = EXPERIMENT / f"runs/development/{args.model}_qet_seed11/summary.json"
    selected = json.loads(summary_path.read_text())
    epochs = int(selected["best_epoch"])
    setting = MODEL_SETTINGS[args.model]
    run_name = f"{args.model}_qet_seed{args.seed}"
    run_dir = EXPERIMENT / f"runs/frozen_refit/{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    donor = stations.loc[stations.role.eq("donor")]
    target = stations.loc[stations.role.eq("target")].reset_index(drop=True)
    donor_rows, donor_q_rows = donor.row.to_numpy(int), donor.q_row.to_numpy(int)
    target_rows = target.row.to_numpy(int)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=setting["learning_rate"] / 20.0
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, q_loss, et_loss = train_epoch(
            model, optimizer, forcing, static, q, dates, donor_rows, donor_q_rows,
            et, year_to_et, np.arange(2005, 2019), args.batch_size,
        )
        scheduler.step()
        record = {
            "epoch": epoch, "training_loss": loss, "training_q_loss": q_loss,
            "training_et_loss": et_loss, "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps({"run": run_name, **record}), flush=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {"model_state": model.state_dict(), "model": args.model, "family": FAMILIES[args.model],
         "seed": args.seed, "epochs": epochs, "n_static": int(static.shape[1])},
        checkpoint_path,
    )
    audit = mass_audit(model, forcing, static, donor_rows)
    model.float().eval()

    prediction_dir = EXPERIMENT / "outputs/predictions"
    state_dir = EXPERIMENT / "outputs/antecedent_states"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    eval_start = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    eval_end = int(np.searchsorted(dates, np.datetime64("2024-12-31"), side="right"))
    eval_dates = np.asarray(dates[eval_start:eval_end])
    prediction_path = prediction_dir / f"{run_name}_target_q_2019_2024.npy"
    prediction = np.lib.format.open_memmap(
        prediction_path, mode="w+", dtype=np.float32, shape=(len(target), len(eval_dates))
    )
    common = np.load(EXPERIMENT / "data/common_forcing_anchors.npz")
    if not np.array_equal(common["gage_ids"].astype(str), target.GAGE_ID.to_numpy(str)):
        raise RuntimeError("Common anchor target order mismatch")
    common_time = common["eval_time_index"].astype(np.int16)
    common_state = np.empty((len(target), 12, model.state_size), dtype=np.float32)
    specific_state = np.empty_like(common_state)
    specific_time = np.empty((len(target), 12), dtype=np.int16)
    with torch.no_grad():
        for first in range(0, len(target), args.batch_size):
            stop = min(first + args.batch_size, len(target))
            rows = target_rows[first:stop]
            x = np.asarray(forcing[rows, :eval_end])
            result = model(
                torch.from_numpy(x).float(), torch.from_numpy(np.asarray(static[rows])).float(),
                return_states=True,
            )
            prediction[first:stop] = result["q"][:, eval_start:eval_end].cpu().numpy()
            state = result["state"][:, eval_start:eval_end].cpu().numpy()
            for local in range(stop - first):
                common_state[first + local] = state[local, common_time[first + local]]
            chosen_state, chosen_time = select_storage_anchors(state, x[:, eval_start:eval_end], eval_dates)
            specific_state[first:stop] = chosen_state
            specific_time[first:stop] = chosen_time
            print(f"{run_name}: predictions and states {stop}/{len(target)}", flush=True)
    prediction.flush()
    common_path = state_dir / f"{run_name}_common_date.npz"
    specific_path = state_dir / f"{run_name}_model_specific.npz"
    np.savez_compressed(
        common_path, gage_ids=target.GAGE_ID.to_numpy(str), quantiles=common["quantiles"],
        state=common_state, eval_time_index=common_time,
    )
    np.savez_compressed(
        specific_path, gage_ids=target.GAGE_ID.to_numpy(str), quantiles=common["quantiles"],
        state=specific_state, eval_time_index=specific_time,
    )
    receipt = {
        "run": run_name, "model": args.model, "family": FAMILIES[args.model], "seed": args.seed,
        "epochs": epochs, "donor_gauges": 400, "target_gauges": 413,
        "nonreference_gauges": 0, "mass_balance": audit,
        "target_discharge_accessed": False,
        "checkpoint": str(checkpoint_path.relative_to(EXPERIMENT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction": str(prediction_path.relative_to(EXPERIMENT)),
        "prediction_sha256": sha256(prediction_path),
        "common_date_states": str(common_path.relative_to(EXPERIMENT)),
        "common_date_states_sha256": sha256(common_path),
        "model_specific_states": str(specific_path.relative_to(EXPERIMENT)),
        "model_specific_states_sha256": sha256(specific_path),
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = EXPERIMENT / "data"
    stations = pd.read_csv(data / "station_index.csv", dtype={"GAGE_ID": str})
    dates = np.load(data / "dates.npy", mmap_mode="r")
    forcing = np.load(data / "forcing.npy", mmap_mode="r")
    static = np.load(data / "static.npy", mmap_mode="r")
    q = np.load(data / "q_donor_development.npy", mmap_mode="r")
    et, year_to_et = load_et(stations)
    model = make_model(args.model, static.shape[1])
    if args.mode == "select":
        run_selection(args, model, forcing, static, q, stations, dates, et, year_to_et)
    else:
        run_refit(args, model, forcing, static, q, stations, dates, et, year_to_et)


if __name__ == "__main__":
    main()
