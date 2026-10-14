#!/usr/bin/env python3
"""Train on 500 donors and select stopping epochs on 200 development gauges."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model, parameter_count  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["ealstm", "dhbv", "mclstm", "fluxgate"], required=True)
    parser.add_argument("--rung", choices=["q", "qet"], required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def masked_moments(values: torch.Tensor, mask: torch.Tensor):
    count = mask.sum(dim=1).clamp(min=2)
    mean = (values * mask).sum(dim=1) / count
    centered = (values - mean[:, None]) * mask
    variance = centered.square().sum(dim=1) / (count - 1)
    return mean, torch.sqrt(variance + 1e-6), centered, count


def discharge_loss(prediction: torch.Tensor, observation: torch.Tensor):
    mask = torch.isfinite(observation).to(prediction.dtype)
    observed = torch.nan_to_num(observation, nan=0.0)
    pred_mean, pred_std, _, count = masked_moments(prediction, mask)
    obs_mean, obs_std, _, _ = masked_moments(observed, mask)

    def normalized_error(predicted_transform, observed_transform):
        observed_mean = (observed_transform * mask).sum(dim=1) / count
        error = ((predicted_transform - observed_transform).square() * mask).sum(dim=1)
        variance = (((observed_transform - observed_mean[:, None]) * mask).square()).sum(dim=1)
        return error / (variance + 0.05 * count)

    raw_error = normalized_error(prediction, observed)
    sqrt_error = normalized_error(torch.sqrt(prediction + 0.01), torch.sqrt(observed + 0.01))
    log_error = normalized_error(torch.log1p(prediction), torch.log1p(observed))
    bias = ((pred_mean - obs_mean) / (obs_mean + 0.10)).square()
    spread = torch.log((pred_std + 0.05) / (obs_std + 0.05)).square()
    return (0.35 * raw_error + 0.30 * sqrt_error + 0.10 * log_error + 0.10 * bias + 0.15 * spread).mean()


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o, p = observed[valid], predicted[valid]
    if np.std(o) <= 0 or np.mean(o) <= 0:
        return np.nan
    correlation = np.corrcoef(o, p)[0, 1]
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(p) / np.mean(o)
    return float(1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    o, p = observed[valid], predicted[valid]
    denominator = np.sum((o - np.mean(o)) ** 2)
    return float(1.0 - np.sum((o - p) ** 2) / denominator) if denominator > 0 else np.nan


def load_et(stations: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    table = pd.read_parquet(PROJECT / "data/processed/donor_ssebop_et_2003_2018.parquet")
    table.GAGE_ID = table.GAGE_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    years = sorted(table.year.unique())
    donor_ids = stations.loc[stations.role.eq("donor"), "GAGE_ID"].tolist()
    pivot = table.pivot(index="GAGE_ID", columns="year", values="ssebop_apr_oct_et_mm").reindex(index=donor_ids, columns=years)
    if pivot.isna().any().any():
        raise RuntimeError("Missing donor SSEBop values")
    return pivot.to_numpy(np.float32), {year: position for position, year in enumerate(years)}


def mass_audit(model: torch.nn.Module, forcing: np.ndarray, static: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    if not getattr(model, "conserves_mass", False):
        return {"daily_max_abs_mm": np.nan, "cumulative_max_abs_mm": np.nan}
    original_dtype = next(model.parameters()).dtype
    model = model.double().eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(np.asarray(forcing[rows, :730])).double(),
            torch.from_numpy(np.asarray(static[rows])).double(),
            return_states=True,
        )
    result = {
        "daily_max_abs_mm": float(output["daily_residual"].abs().max()),
        "cumulative_max_abs_mm": float(output["cumulative_residual"].abs().max()),
    }
    model.to(dtype=original_dtype)
    return result


def evaluate(model, forcing, static, q, stations, dates, batch_size):
    development = stations[stations.role.eq("development")]
    rows = development.row.to_numpy(int)
    q_rows = development.q_row.to_numpy(int)
    end = np.searchsorted(dates, np.datetime64("2018-12-31"), side="right")
    start = np.searchsorted(dates[:end], np.datetime64("2015-01-01"))
    predictions = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(rows), batch_size):
            chosen = rows[first:first + batch_size]
            output = model(
                torch.from_numpy(np.asarray(forcing[chosen, :end])).float(),
                torch.from_numpy(np.asarray(static[chosen])).float(),
            )
            predictions.append(output["q"][:, start:end].cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    observation = np.asarray(q[np.ix_(q_rows, np.arange(start, end))])
    records = []
    for i, station in enumerate(development.itertuples(index=False)):
        records.append({
            "GAGE_ID": station.GAGE_ID,
            "row": int(station.row),
            "kge": kge(observation[i], prediction[i]),
            "nse": nse(observation[i], prediction[i]),
            "bias_fraction": float((np.nanmean(prediction[i]) - np.nanmean(observation[i])) / (np.nanmean(observation[i]) + 1e-6)),
        })
    return pd.DataFrame(records), prediction


def main() -> None:
    args = parse_args()
    if args.architecture == "ealstm" and args.rung != "q":
        raise ValueError("The predictive LSTM benchmark is Q-only")
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    processed = PROJECT / "data/processed"
    stations = pd.read_csv(processed / "station_index.csv", dtype={"GAGE_ID": str})
    dates = np.load(processed / "dates.npy", mmap_mode="r")
    forcing = np.load(processed / "forcing.npy", mmap_mode="r")
    static = np.load(processed / "static.npy", mmap_mode="r")
    q = np.load(processed / "q_donor_development.npy", mmap_mode="r")
    et, year_to_et = load_et(stations)
    donor = stations[stations.role.eq("donor")]
    donor_rows = donor.row.to_numpy(int)
    donor_q_rows = donor.q_row.to_numpy(int)

    defaults = {
        "ealstm": {"epochs": 30, "learning_rate": 0.0010, "weight_decay": 1e-5, "eval_every": 3},
        "dhbv": {"epochs": 18, "learning_rate": 0.0020, "weight_decay": 1e-5, "eval_every": 3},
        "mclstm": {"epochs": 18, "learning_rate": 0.0015, "weight_decay": 1e-5, "eval_every": 3},
        "fluxgate": {"epochs": 24, "learning_rate": 0.0015, "weight_decay": 1e-5, "eval_every": 3},
    }[args.architecture]
    epochs = args.epochs or defaults["epochs"]
    model = make_model(args.architecture, static.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=defaults["learning_rate"], weight_decay=defaults["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=defaults["learning_rate"] / 20.0)

    run_name = f"{args.architecture}_{args.rung}_seed{args.seed}"
    output_dir = PROJECT / "runs/development" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch = [], -np.inf, None
    training_years = np.arange(2005, 2015)

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.permutation(len(donor_rows))
        losses, q_losses, et_losses = [], [], []
        for first in range(0, len(shuffled), args.batch_size):
            local = shuffled[first:first + args.batch_size]
            rows = donor_rows[local]
            q_rows = donor_q_rows[local]
            year = int(np.random.choice(training_years))
            start = np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01"))
            stop = np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right")
            target_start = np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01"))
            x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            observed = torch.from_numpy(np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])).float()
            result = model(x, s)
            q_loss = discharge_loss(result["q"][:, target_start:], observed)
            et_loss = torch.zeros((), dtype=q_loss.dtype)
            if args.rung == "qet":
                season = (dates[start:stop] >= np.datetime64(f"{year}-04-01")) & (dates[start:stop] <= np.datetime64(f"{year}-10-31"))
                predicted_et = result["et"][:, torch.from_numpy(season)].sum(dim=1)
                target_et = torch.from_numpy(et[local, year_to_et[year]]).float()
                scaled = (predicted_et - target_et) / (0.20 * target_et + 50.0)
                et_loss = torch.nn.functional.smooth_l1_loss(scaled, torch.zeros_like(scaled))
            loss = q_loss + (0.20 * et_loss if args.rung == "qet" else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach())); q_losses.append(float(q_loss.detach())); et_losses.append(float(et_loss.detach()))
        scheduler.step()

        should_evaluate = epoch % defaults["eval_every"] == 0 or epoch == epochs
        median_kge = q25_kge = np.nan
        if should_evaluate:
            metrics, predictions = evaluate(model, forcing, static, q, stations, dates, args.batch_size)
            median_kge = float(metrics.kge.median())
            q25_kge = float(metrics.kge.quantile(0.25))
            if median_kge > best_score:
                best_score, best_epoch = median_kge, epoch
                torch.save({
                    "model_state": model.state_dict(), "architecture": args.architecture,
                    "rung": args.rung, "seed": args.seed, "n_static": static.shape[1],
                    "epoch": epoch,
                }, output_dir / "best_model.pt")
                metrics.to_csv(output_dir / "development_metrics.csv", index=False)
                np.savez_compressed(
                    output_dir / "development_predictions.npz",
                    dates=np.asarray(dates[(dates >= np.datetime64('2015-01-01')) & (dates <= np.datetime64('2018-12-31'))]),
                    gage_ids=stations.loc[stations.role.eq("development"), "GAGE_ID"].to_numpy(str),
                    predicted=predictions,
                )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "training_q_loss": float(np.mean(q_losses)),
            "training_et_loss": float(np.mean(et_losses)),
            "development_median_kge": median_kge,
            "development_q25_kge": q25_kge,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps({"run": run_name, **record}), flush=True)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    audit = mass_audit(model, forcing, static, donor_rows[:4])
    summary = {
        "run": run_name,
        "architecture": args.architecture,
        "rung": args.rung,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "donor_gauges": len(donor_rows),
        "development_gauges": int(stations.role.eq("development").sum()),
        "training_years": [2005, 2014],
        "development_period": ["2015-01-01", "2018-12-31"],
        "best_epoch": int(best_epoch),
        "best_development_median_kge": float(best_score),
        "mass_balance": audit,
        "target_discharge_accessed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
