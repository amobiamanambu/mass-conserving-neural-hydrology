#!/usr/bin/env python3
"""Select one stopping epoch per architecture without reading target discharge."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conserved_models import MODEL_NAMES, make_model, parameter_count


EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[2]
LEARNING_RATES = {
    "mclstm_dr": 0.0010,
    "mc_node": 0.0010,
    "mcp_ma3": 0.0015,
    "mcp_ma4": 0.0015,
    "mcp_ma5": 0.0015,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


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
    sqrt_error = normalized_error(
        torch.sqrt(prediction + 0.01), torch.sqrt(observed + 0.01)
    )
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
    correlation = np.corrcoef(o, p)[0, 1]
    alpha = np.std(p, ddof=1) / np.std(o, ddof=1)
    beta = np.mean(p) / np.mean(o)
    return float(
        1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    )


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
    pivot = table.pivot(
        index="GAGE_ID", columns="year", values="ssebop_apr_oct_et_mm"
    ).reindex(index=donor_ids, columns=years)
    if pivot.isna().any().any():
        raise RuntimeError("Missing donor SSEBop values")
    return pivot.to_numpy(np.float32), {int(year): i for i, year in enumerate(years)}


def evaluate(model, forcing, static, q, stations, dates, batch_size):
    development = stations.loc[stations.role.eq("development")]
    rows = development.row.to_numpy(int)
    q_rows = development.q_row.to_numpy(int)
    end = int(np.searchsorted(dates, np.datetime64("2018-12-31"), side="right"))
    start = int(np.searchsorted(dates[:end], np.datetime64("2015-01-01")))
    predictions = []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(rows), batch_size):
            chosen = rows[first : first + batch_size]
            result = model(
                torch.from_numpy(np.asarray(forcing[chosen, :end])).float(),
                torch.from_numpy(np.asarray(static[chosen])).float(),
            )
            predictions.append(result["q"][:, start:end].cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    observation = np.asarray(q[np.ix_(q_rows, np.arange(start, end))])
    records = []
    for index, station in enumerate(development.itertuples(index=False)):
        records.append(
            {
                "GAGE_ID": station.GAGE_ID,
                "row": int(station.row),
                "kge": kge(observation[index], prediction[index]),
                "nse": nse(observation[index], prediction[index]),
                "bias_fraction": float(
                    (np.nanmean(prediction[index]) - np.nanmean(observation[index]))
                    / (np.nanmean(observation[index]) + 1e-6)
                ),
            }
        )
    return pd.DataFrame(records), prediction


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed = PROJECT / "data/processed"
    stations = pd.read_csv(processed / "station_index.csv", dtype={"GAGE_ID": str})
    dates = np.load(processed / "dates.npy", mmap_mode="r")
    forcing = np.load(processed / "forcing.npy", mmap_mode="r")
    static = np.load(processed / "static.npy", mmap_mode="r")
    q = np.load(processed / "q_donor_development.npy", mmap_mode="r")
    et, year_to_et = load_et(stations)
    donor = stations.loc[stations.role.eq("donor")]
    donor_rows = donor.row.to_numpy(int)
    donor_q_rows = donor.q_row.to_numpy(int)

    model = make_model(args.model, static.shape[1])
    learning_rate = LEARNING_RATES[args.model]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=learning_rate / 20.0
    )

    run_name = f"{args.model}_qet_seed{args.seed}"
    output_dir = EXPERIMENT / "runs/development" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -np.inf
    best_epoch = None
    training_years = np.arange(2005, 2015)

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = np.random.permutation(len(donor_rows))
        losses, q_losses, et_losses = [], [], []
        for first in range(0, len(shuffled), args.batch_size):
            local = shuffled[first : first + args.batch_size]
            rows = donor_rows[local]
            q_rows = donor_q_rows[local]
            year = int(np.random.choice(training_years))
            start = int(np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01")))
            stop = int(np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right"))
            target_start = int(np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01")))
            x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            observed = torch.from_numpy(
                np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])
            ).float()
            result = model(x, s)
            q_loss = discharge_loss(result["q"][:, target_start:], observed)
            season = (dates[start:stop] >= np.datetime64(f"{year}-04-01")) & (
                dates[start:stop] <= np.datetime64(f"{year}-10-31")
            )
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
        scheduler.step()

        median_kge = q25_kge = np.nan
        if epoch % 3 == 0 or epoch == args.epochs:
            development_metrics, predictions = evaluate(
                model, forcing, static, q, stations, dates, args.batch_size
            )
            median_kge = float(development_metrics.kge.median())
            q25_kge = float(development_metrics.kge.quantile(0.25))
            if median_kge > best_score:
                best_score = median_kge
                best_epoch = epoch
                checkpoint_path = output_dir / "best_model.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "architecture": args.model,
                        "family": model.family,
                        "rung": "qet",
                        "seed": args.seed,
                        "n_static": int(static.shape[1]),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                development_metrics.to_csv(output_dir / "development_metrics.csv", index=False)
                np.savez_compressed(
                    output_dir / "development_predictions.npz",
                    dates=np.asarray(
                        dates[
                            (dates >= np.datetime64("2015-01-01"))
                            & (dates <= np.datetime64("2018-12-31"))
                        ]
                    ),
                    gage_ids=stations.loc[
                        stations.role.eq("development"), "GAGE_ID"
                    ].to_numpy(str),
                    predicted=predictions,
                )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "training_q_loss": float(np.mean(q_losses)),
            "training_et_loss": float(np.mean(et_losses)),
            "development_median_kge": median_kge,
            "development_q25_kge": q25_kge,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        print(json.dumps({"run": run_name, **record}), flush=True)

    if best_epoch is None:
        raise RuntimeError("No finite development epoch was selected")
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint_path = output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.double().eval()
    with torch.no_grad():
        audit = model(
            torch.from_numpy(np.asarray(forcing[donor_rows[:4], :730])).double(),
            torch.from_numpy(np.asarray(static[donor_rows[:4]])).double(),
            return_states=True,
        )
    summary = {
        "run": run_name,
        "architecture": args.model,
        "family": model.family,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "best_development_median_kge": float(best_score),
        "maximum_epochs": args.epochs,
        "learning_rate": learning_rate,
        "parameter_count": parameter_count(model),
        "daily_mass_residual_max_abs_mm": float(audit["daily_residual"].abs().max()),
        "cumulative_mass_residual_max_abs_mm": float(
            audit["cumulative_residual"].abs().max()
        ),
        "checkpoint": str(checkpoint_path.relative_to(EXPERIMENT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "target_discharge_accessed_by_script": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
