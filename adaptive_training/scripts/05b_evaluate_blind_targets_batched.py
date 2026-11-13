#!/usr/bin/env python3
"""Freeze blind national predictions with a parity-validated NeuralHydrology evaluator.

The stock NeuralHydrology tester creates one DataLoader per gauge. This script uses
the same Config, GenericDataset, collate function, model class, checkpoint, pre-model
hook, target subsetting, and framework scaler, but batches samples across small groups
of gauges. The changed execution schedule is validated against the stock CPU tester
before a national archive can be written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import MethodType

import numpy as np
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import xarray as xr
from torch.utils.data import DataLoader

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
TARGET_DAYS = 2192
GROUP_SIZE = 32
ACCELERATOR_BATCH_SIZE = 8192
PARITY_GAUGES = 2
IMPLEMENTATION_ATOL = 2e-6
IMPLEMENTATION_RTOL = 2e-6
ACCELERATOR_MAX_ABS_MM_DAY = 0.02
ACCELERATOR_MEDIAN_ABS_MM_DAY = 0.001
ACCELERATOR_MAX_RELATIVE = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_blind_receipt(gauges: list[str]) -> dict:
    receipt_path = ROOT / "provenance/blind_target_verification.json"
    receipt = json.loads(receipt_path.read_text())
    inventory_path = ROOT / receipt["inventory"]
    if sha256(inventory_path) != receipt["inventory_sha256"]:
        raise RuntimeError("Blind-target inventory hash mismatch")
    if receipt.get("target_discharge_accessed") is not False:
        raise RuntimeError("Blind-target verification did not preserve the target blind")
    if receipt.get("all_target_values_are_dummy_zero") is not True:
        raise RuntimeError("Blind-target verification did not pass")
    if receipt.get("target_gauges") != len(gauges):
        raise RuntimeError("Blind-target receipt gauge count mismatch")
    return receipt


def stock_cpu_predictions(run_dir: Path, epoch: int, gauges: list[str]) -> np.ndarray:
    """Run the unmodified gauge-wise framework evaluator for parity evidence."""
    cfg = Config(run_dir / "config.yml")
    cfg.device = "cpu"
    cfg._cfg["batch_size"] = 256
    cfg._cfg["verbose"] = 0
    tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
    tester.basins = list(gauges)
    results = tester.evaluate(epoch=epoch, save_results=False, save_all_output=False, metrics=[])
    values = []
    for gauge in gauges:
        frame = results[gauge]["1D"]["xr"]
        values.append(np.asarray(frame["q_mm_day_sim"][:, 0].values, dtype=np.float64))
    return np.stack(values)


def _install_grouped_mclstm_step(model: torch.nn.Module, group_count: int) -> None:
    """Vectorize independent stock-evaluator batches without mixing their L1 norms.

    NeuralHydrology 1.13.0 normalizes the MC-LSTM state by one scalar L1 norm
    over the complete evaluator batch. We therefore preserve each gauge's stock
    batch as an independent group inside the larger accelerator batch.
    """
    cell = model.mclstm

    def grouped_step(self, xt_m, xt_a, c):
        grouped = c.reshape(group_count, -1, self._hidden_size)
        denominator = grouped.norm(p=1, dim=(1, 2), keepdim=True) + 1e-5
        normalized = (grouped / denominator).reshape_as(c)
        features = torch.cat([xt_m, xt_a, normalized], dim=-1)
        i = self.input_gate(features)
        r = self.redistribution(features)
        o = self.output_gate(features)
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        return o * m_new, (1 - o) * m_new

    cell._step = MethodType(grouped_step, cell)


def _move_batch(data: dict, device: torch.device) -> dict:
    for key in data:
        if key.startswith("x_d"):
            data[key] = {name: value.to(device) for name, value in data[key].items()}
        elif not key.startswith("date"):
            data[key] = data[key].to(device)
    return data


def _grouped_mclstm_predictions(
    tester, dataset, gauges: list[str], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce gauge-wise stock batches while evaluating them in parallel."""
    stock_microbatch = 256
    if stock_microbatch * len(gauges) > batch_size:
        raise RuntimeError("MC-LSTM grouped evaluation requires the stock batch size of 256")
    _install_grouped_mclstm_step(tester.model, len(gauges))
    predictions = np.empty((len(gauges), TARGET_DAYS), dtype=np.float64)
    dates = np.empty((len(gauges), TARGET_DAYS), dtype="datetime64[D]")
    with torch.no_grad():
        for start in range(0, TARGET_DAYS, stock_microbatch):
            stop = min(start + stock_microbatch, TARGET_DAYS)
            width = stop - start
            indices = [
                gauge_index * TARGET_DAYS + time_index
                for gauge_index in range(len(gauges))
                for time_index in range(start, stop)
            ]
            data = dataset.collate_fn([dataset[index] for index in indices])
            batch_dates = np.asarray(data["date"][:, -1]).astype("datetime64[D]")
            data = _move_batch(data, tester.device)
            data = tester.model.pre_model_hook(data, is_train=False)
            output = tester.model(data)["y_hat"][:, -1, 0].detach().cpu().numpy()
            predictions[:, start:stop] = output.reshape(len(gauges), width)
            dates[:, start:stop] = batch_dates.reshape(len(gauges), width)
    return predictions, dates


def run_group(
    run_dir: Path,
    epoch: int,
    gauges: list[str],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one gauge group through framework-native data/model operations."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as stream:
        stream.write("\n".join(gauges) + "\n")
        group_file = Path(stream.name)
    try:
        cfg = Config(run_dir / "config.yml")
        cfg.device = device
        cfg._cfg["batch_size"] = batch_size
        cfg._cfg["verbose"] = 0
        cfg._cfg["test_basin_file"] = group_file
        tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
        tester._load_weights(epoch)
        tester.model.eval()
        dataset = get_dataset(
            cfg=cfg,
            is_train=False,
            period="test",
            basin=None,
            additional_features=tester.additional_features,
            id_to_int=tester.id_to_int,
            scaler=tester.scaler,
        )
        expected_samples = len(gauges) * TARGET_DAYS
        if len(dataset) != expected_samples:
            raise RuntimeError(
                f"Group has {len(dataset)} framework samples; expected {expected_samples}"
            )
        lookup_gauges = [dataset.lookup_table[index][0] for index in range(len(dataset))]
        expected_order = [gauge for gauge in gauges for _ in range(TARGET_DAYS)]
        if lookup_gauges != expected_order:
            raise RuntimeError("Framework lookup order is not gauge-major and complete")

        if cfg.model.lower() == "mclstm":
            normalized_2d, date_values = _grouped_mclstm_predictions(
                tester, dataset, gauges, batch_size
            )
            normalized = normalized_2d[:, :, None]
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
            raw_predictions, _, raw_dates, _, _ = tester._evaluate(
                tester.model, loader, dataset.frequencies, save_all_output=False
            )
            normalized = raw_predictions["1D"].reshape(len(gauges), TARGET_DAYS, 1)
            date_values = np.asarray(raw_dates["1D"][:, -1]).astype("datetime64[D]").reshape(
                len(gauges), TARGET_DAYS
            )
        feature_scale = (
            tester.scaler["xarray_feature_scale"][cfg.target_variables]
            .to_array().values
        )
        feature_center = (
            tester.scaler["xarray_feature_center"][cfg.target_variables]
            .to_array().values
        )
        physical = normalized * feature_scale + feature_center
        physical = np.asarray(physical[:, :, 0], dtype=np.float64)
        if not np.all(date_values == date_values[0]):
            raise RuntimeError("Framework dates differ across gauges")
        if not np.all(np.isfinite(physical)):
            raise RuntimeError("Nonfinite national predictions cannot be frozen")
        return physical, date_values[0]
    finally:
        group_file.unlink(missing_ok=True)
        gc.collect()
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.model == "author_dhbv":
        os.environ.setdefault("NGEN_SILENT", "1")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run outside the filesystem sandbox")

    selection_path = ROOT / "provenance/development_epoch_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("target_discharge_accessed") is not False:
        raise RuntimeError("Development selection receipt does not preserve blinding")
    model_info = selection["architectures"][args.model]
    epoch = int(model_info["selected_epoch"])
    run_dir = ROOT / model_info["runs"][str(args.seed)]["run_dir"]
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} target gauges, found {len(gauges)}")
    blind_receipt = verify_blind_receipt(gauges)

    parity_gauges = gauges[:PARITY_GAUGES]
    parity_start = time.time()
    stock = stock_cpu_predictions(run_dir, epoch, parity_gauges)
    batched_cpu, _ = run_group(
        run_dir, epoch, parity_gauges, "cpu", ACCELERATOR_BATCH_SIZE
    )
    accelerated, parity_dates = run_group(
        run_dir, epoch, parity_gauges, args.device, ACCELERATOR_BATCH_SIZE
    )
    implementation_absolute = np.abs(stock - batched_cpu)
    implementation_passes = bool(np.allclose(
        stock,
        batched_cpu,
        atol=IMPLEMENTATION_ATOL,
        rtol=IMPLEMENTATION_RTOL,
        equal_nan=False,
    ))
    accelerator_absolute = np.abs(batched_cpu - accelerated)
    denominator = np.maximum(np.abs(batched_cpu), ACCELERATOR_MAX_ABS_MM_DAY)
    accelerator_relative = accelerator_absolute / denominator
    accelerator_passes = bool(
        np.max(accelerator_absolute) <= ACCELERATOR_MAX_ABS_MM_DAY
        and np.median(accelerator_absolute) <= ACCELERATOR_MEDIAN_ABS_MM_DAY
        and np.max(accelerator_relative) <= ACCELERATOR_MAX_RELATIVE
    )
    parity_passes = implementation_passes and accelerator_passes
    parity = {
        "model": args.model,
        "seed": args.seed,
        "gauges": parity_gauges,
        "stock_device": "cpu",
        "batched_device": args.device,
        "stock_batch_size": 256,
        "batched_batch_size": ACCELERATOR_BATCH_SIZE,
        "implementation_cpu_maximum_absolute_difference_mm_day": float(np.max(implementation_absolute)),
        "implementation_cpu_median_absolute_difference_mm_day": float(np.median(implementation_absolute)),
        "implementation_atol": IMPLEMENTATION_ATOL,
        "implementation_rtol": IMPLEMENTATION_RTOL,
        "implementation_passes": implementation_passes,
        "accelerator_maximum_absolute_difference_mm_day": float(np.max(accelerator_absolute)),
        "accelerator_median_absolute_difference_mm_day": float(np.median(accelerator_absolute)),
        "accelerator_maximum_relative_difference": float(np.max(accelerator_relative)),
        "accelerator_maximum_absolute_tolerance_mm_day": ACCELERATOR_MAX_ABS_MM_DAY,
        "accelerator_median_absolute_tolerance_mm_day": ACCELERATOR_MEDIAN_ABS_MM_DAY,
        "accelerator_maximum_relative_tolerance": ACCELERATOR_MAX_RELATIVE,
        "accelerator_passes": accelerator_passes,
        "passes": parity_passes,
        "elapsed_seconds": time.time() - parity_start,
        "target_discharge_accessed": False,
    }
    parity_dir = ROOT / "provenance/evaluator_parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    parity_path = parity_dir / f"{args.model}_seed{args.seed}.json"
    parity_path.write_text(json.dumps(parity, indent=2) + "\n")
    if not parity_passes:
        raise RuntimeError(f"Batched evaluator failed stock parity: {parity}")
    print(json.dumps(parity, indent=2), flush=True)
    if args.parity_only:
        return

    all_predictions = np.empty((TARGET_GAUGES, TARGET_DAYS), dtype=np.float64)
    common_dates = None
    evaluation_start = time.time()
    for start in range(0, TARGET_GAUGES, GROUP_SIZE):
        stop = min(start + GROUP_SIZE, TARGET_GAUGES)
        group_predictions, group_dates = run_group(
            run_dir,
            epoch,
            gauges[start:stop],
            args.device,
            ACCELERATOR_BATCH_SIZE,
        )
        all_predictions[start:stop] = group_predictions
        if common_dates is None:
            common_dates = group_dates
        elif not np.array_equal(common_dates, group_dates):
            raise RuntimeError("Date mismatch between national gauge groups")
        print(
            f"{args.model} seed {args.seed}: {stop}/{TARGET_GAUGES} gauges "
            f"({time.time() - evaluation_start:.1f} s)",
            flush=True,
        )

    if all_predictions.shape != (TARGET_GAUGES, TARGET_DAYS):
        raise RuntimeError(f"Unexpected prediction shape {all_predictions.shape}")
    if common_dates is None or len(common_dates) != TARGET_DAYS:
        raise RuntimeError("Missing or incomplete prediction dates")

    frozen_dir = ROOT / "outputs/frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    archive = frozen_dir / f"{args.model}_seed{args.seed}_stock_predictions.npz"
    np.savez_compressed(
        archive,
        gauge_ids=np.asarray(gauges),
        dates=common_dates,
        q_sim_mm_day=all_predictions,
    )
    receipt = {
        "model": args.model,
        "seed": args.seed,
        "selected_epoch": epoch,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "evaluator": "NeuralHydrology framework operations with cross-gauge batching",
        "execution_device": args.device,
        "group_size": GROUP_SIZE,
        "batch_size": ACCELERATOR_BATCH_SIZE,
        "stock_parity_receipt": str(parity_path.relative_to(ROOT)),
        "stock_parity_receipt_sha256": sha256(parity_path),
        "stock_parity_passes": True,
        "blind_target_verification": "provenance/blind_target_verification.json",
        "blind_target_inventory_sha256": blind_receipt["inventory_sha256"],
        "framework_evaluator_result": str(archive.relative_to(ROOT)),
        "framework_evaluator_result_sha256": sha256(archive),
        "frozen_prediction_archive": str(archive.relative_to(ROOT)),
        "frozen_prediction_archive_sha256": sha256(archive),
        "target_gauges": TARGET_GAUGES,
        "target_days": TARGET_DAYS,
        "elapsed_seconds": time.time() - evaluation_start,
        "target_discharge_accessed": False,
        "dummy_target_metrics_used": False,
    }
    receipt_path = frozen_dir / f"{args.model}_seed{args.seed}_prediction_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
