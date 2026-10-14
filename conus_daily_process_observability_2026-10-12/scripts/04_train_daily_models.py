#!/usr/bin/env python3
"""Refit frozen daily models with dynamic process-observation targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model, parameter_count  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
SOURCE = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
RUNS = PROJECT / "runs" / "frozen_refit"
RUNG_TARGETS = {
    "q": (), "q_sm": ("sm",), "q_et": ("et",), "q_sm_et": ("sm", "et"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["dhbv", "fluxgate"], required=True)
    parser.add_argument("--rung", choices=list(RUNG_TARGETS), required=True)
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true")
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


def discharge_loss(prediction: torch.Tensor, observation: torch.Tensor):
    mask = torch.isfinite(observation).to(prediction.dtype)
    observed = torch.nan_to_num(observation, nan=0.0)
    pred_mean, pred_std, count = masked_moments(prediction, mask)
    obs_mean, obs_std, _ = masked_moments(observed, mask)

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


def masked_smooth_l1(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(observed)
    if not torch.any(mask):
        return predicted.new_tensor(0.0)
    return torch.nn.functional.smooth_l1_loss(predicted[mask], observed[mask])


def model_soil_fraction(model: torch.nn.Module, state: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
    if model.architecture == "dhbv":
        capacity = model.parameters_from_static(static)["capacity"]
    elif model.architecture == "fluxgate":
        _, terms = model.static_terms(static)
        capacity = terms["capacity"]
    else:
        raise ValueError(model.architecture)
    return state[..., 1] / capacity[:, None]


def copy_q_baseline(architecture: str, seed: int, destination: Path) -> None:
    source = SOURCE / "runs" / "frozen_refit" / f"{architecture}_q_seed{seed}"
    if not (source / "model.pt").exists():
        raise FileNotFoundError(source / "model.pt")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "model.pt", destination / "model.pt")
    for optional in ["training_history.csv", "receipt.json"]:
        if (source / optional).exists():
            shutil.copy2(source / optional, destination / f"source_{optional}")
    receipt = {
        "run": destination.name,
        "source": str(source.relative_to(ROOT)),
        "model_sha256": sha256(destination / "model.pt"),
        "copied_without_retraining": True,
        "reason": "exact frozen Q-only comparator from the source experiment",
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def load_data():
    base = SOURCE / "data" / "processed"
    stations = pd.read_csv(base / "station_index.csv", dtype={"GAGE_ID": str})
    stations["GAGE_ID"] = stations.GAGE_ID.str.zfill(8)
    dates = np.load(base / "dates.npy", mmap_mode="r")
    forcing = np.load(base / "forcing.npy", mmap_mode="r")
    static = np.load(base / "static.npy", mmap_mode="r")
    q = np.load(base / "q_donor_development.npy", mmap_mode="r")
    sm = np.load(PROJECT / "data/model_ready/nldas_rootzone_saturation.npy", mmap_mode="r")
    et_file = np.load(PROJECT / "data/model_ready/ssebop_monthly.npz")
    et_ids = et_file["gage_ids"].astype(str)
    et = et_file["eta_month_mm"]
    block_start = et_file["block_start"]
    block_stop = et_file["block_stop"]
    donor = stations[stations.role.eq("donor")]
    et_lookup = {gage: i for i, gage in enumerate(et_ids)}
    donor_et_rows = np.asarray([et_lookup[gage] for gage in donor.GAGE_ID], int)
    return stations, dates, forcing, static, q, sm, et, block_start, block_stop, donor_et_rows


def main() -> None:
    args = arguments()
    run_name = f"{args.architecture}_{args.rung}_seed{args.seed}"
    run_dir = RUNS / run_name
    if (run_dir / "model.pt").exists() and not args.force:
        print(f"{run_name}: already complete")
        return
    if args.rung == "q":
        copy_q_baseline(args.architecture, args.seed, run_dir)
        return

    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    (
        stations, dates, forcing, static, q, sm, et, block_start, block_stop, donor_et_rows,
    ) = load_data()
    donor = stations[stations.role.eq("donor")]
    donor_rows = donor.row.to_numpy(int)
    donor_q_rows = donor.q_row.to_numpy(int)
    epochs = 15 if args.architecture == "dhbv" else 24
    learning_rate = 0.0020 if args.architecture == "dhbv" else 0.0015
    model = make_model(args.architecture, static.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 20.0
    )
    targets = RUNG_TARGETS[args.rung]
    years = np.arange(2005, 2019)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = np.random.permutation(len(donor_rows))
        records = {"loss": [], "q": [], "sm": [], "et": []}
        for first in range(0, len(shuffled), args.batch_size):
            local = shuffled[first:first + args.batch_size]
            rows = donor_rows[local]
            q_rows = donor_q_rows[local]
            year = int(np.random.choice(years))
            start = int(np.searchsorted(dates, np.datetime64(f"{year - 1}-07-01")))
            stop = int(np.searchsorted(dates, np.datetime64(f"{year}-12-31"), side="right"))
            target_start = int(np.searchsorted(dates[start:stop], np.datetime64(f"{year}-01-01")))
            x = torch.from_numpy(np.asarray(forcing[rows, start:stop])).float()
            s = torch.from_numpy(np.asarray(static[rows])).float()
            observed_q = torch.from_numpy(
                np.asarray(q[np.ix_(q_rows, np.arange(start + target_start, stop))])
            ).float()
            output = model(x, s, return_states="sm" in targets)
            q_loss = discharge_loss(output["q"][:, target_start:], observed_q)
            auxiliary = []
            sm_loss = q_loss.new_tensor(0.0)
            et_loss = q_loss.new_tensor(0.0)
            if "sm" in targets:
                soil = model_soil_fraction(model, output["state"][:, target_start:], s)
                predicted_sm = torch.clamp(soil, 0.0, 1.0)
                observed_sm = torch.from_numpy(np.asarray(sm[rows, start + target_start:stop])).float()
                sm_loss = masked_smooth_l1(predicted_sm, observed_sm)
                auxiliary.append(sm_loss)
            if "et" in targets:
                blocks = np.flatnonzero((block_start >= start + target_start) & (block_stop <= stop))
                if len(blocks):
                    predicted_blocks = []
                    for block in blocks:
                        predicted_blocks.append(output["et"][:, block_start[block] - start:block_stop[block] - start].sum(dim=1))
                    predicted_et = torch.stack(predicted_blocks, dim=1)
                    observed_et = torch.from_numpy(et[np.ix_(donor_et_rows[local], blocks)]).float()
                    et_loss = masked_smooth_l1(
                        torch.log1p(torch.clamp(predicted_et, min=0.0)) / 4.0,
                        torch.log1p(torch.clamp(observed_et, min=0.0)) / 4.0,
                    )
                    if torch.isfinite(observed_et).any():
                        auxiliary.append(et_loss)
            loss = q_loss + 0.20 * torch.stack(auxiliary).mean() if auxiliary else q_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            records["loss"].append(float(loss.detach()))
            records["q"].append(float(q_loss.detach()))
            records["sm"].append(float(sm_loss.detach()))
            records["et"].append(float(et_loss.detach()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(records["loss"])),
            "training_q_loss": float(np.mean(records["q"])),
            "training_sm_loss": float(np.mean(records["sm"])),
            "training_et_loss": float(np.mean(records["et"])),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(json.dumps({"run": run_name, **row}), flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    checkpoint = run_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(), "architecture": args.architecture,
        "rung": args.rung, "seed": args.seed, "n_static": int(static.shape[1]),
        "epochs": epochs, "refit_end": "2018-12-31", "auxiliary_weight": 0.20,
        "targets": targets,
    }, checkpoint)
    receipt = {
        "run": run_name, "architecture": args.architecture, "rung": args.rung,
        "seed": args.seed, "epochs": epochs, "parameter_count": parameter_count(model),
        "donor_gauges": len(donor_rows), "training_years": [2005, 2018],
        "auxiliary_weight": 0.20, "targets": targets,
        "training_seconds": time.time() - started,
        "model_sha256": sha256(checkpoint), "target_discharge_accessed": False,
        "model_source_sha256": sha256(Path(__file__).resolve().parent / "hydro_models.py"),
        "training_target_sha256": {
            **({"nldas_rootzone_saturation.npy": sha256(PROJECT / "data/model_ready/nldas_rootzone_saturation.npy")} if "sm" in targets else {}),
            **({"ssebop_monthly.npz": sha256(PROJECT / "data/model_ready/ssebop_monthly.npz")} if "et" in targets else {}),
        },
    }
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
