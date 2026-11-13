#!/usr/bin/env python3
"""Run the stock evaluator against blinded national targets and freeze predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
ADAPTER = ROOT / "vendor" / "neuralhydrology_literature_adapter"
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
TARGET_DAYS = 2192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_blind_targets(gauges: list[str]) -> None:
    for index, gauge in enumerate(gauges):
        path = ROOT / "data/generic_blind/time_series" / f"{gauge}.nc"
        with xr.open_dataset(path) as dataset:
            test = dataset["q_mm_day"].sel(date=slice("2019-01-01", "2024-12-31"))
            if not np.all(test.values == 0):
                raise RuntimeError(f"Non-dummy target discharge found in {path}")
            if int(dataset.attrs.get("target_discharge_blinded", 0)) != 1:
                raise RuntimeError(f"Missing blinded-target marker in {path}")
        if (index + 1) % 500 == 0:
            print(f"Verified {index + 1}/{len(gauges)} blinded target files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args()

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
    verify_blind_targets(gauges)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ADAPTER)
    environment["MPLCONFIGDIR"] = "/private/tmp/nh_mpl_cache"
    if args.model == "author_dhbv":
        environment["NGEN_SILENT"] = "1"
    subprocess.run(
        [
            str(PYTHON), "-m", "neuralhydrology.nh_run", "evaluate",
            "--run-dir", str(run_dir), "--period", "test", "--epoch", str(epoch),
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )

    result_dir = run_dir / "test" / f"model_epoch{epoch:03d}"
    result_path = result_dir / "test_results.p"
    results = pickle.loads(result_path.read_bytes())
    if set(results) != set(gauges):
        raise RuntimeError("Stock evaluator result gauges do not match frozen target list")
    predictions = []
    common_dates = None
    for gauge in gauges:
        frame = results[gauge]["1D"]["xr"]
        values = np.asarray(frame["q_mm_day_sim"][:, 0].values, dtype=np.float64)
        dates = np.asarray(frame["date"].values).astype("datetime64[D]")
        if common_dates is None:
            common_dates = dates
        elif not np.array_equal(common_dates, dates):
            raise RuntimeError(f"Date mismatch for target gauge {gauge}")
        predictions.append(values)
    predictions_array = np.stack(predictions)
    if predictions_array.shape != (TARGET_GAUGES, TARGET_DAYS):
        raise RuntimeError(f"Unexpected prediction shape {predictions_array.shape}")
    if not np.all(np.isfinite(predictions_array)):
        raise RuntimeError("Nonfinite target predictions cannot be frozen")

    frozen_dir = ROOT / "outputs/frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    archive = frozen_dir / f"{args.model}_seed{args.seed}_stock_predictions.npz"
    np.savez_compressed(
        archive,
        gauge_ids=np.asarray(gauges),
        dates=common_dates,
        q_sim_mm_day=predictions_array,
    )
    receipt = {
        "model": args.model,
        "seed": args.seed,
        "selected_epoch": epoch,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "framework_evaluator_result": str(result_path.relative_to(ROOT)),
        "framework_evaluator_result_sha256": sha256(result_path),
        "frozen_prediction_archive": str(archive.relative_to(ROOT)),
        "frozen_prediction_archive_sha256": sha256(archive),
        "target_gauges": TARGET_GAUGES,
        "target_days": TARGET_DAYS,
        "target_discharge_accessed": False,
        "dummy_target_metrics_used": False,
    }
    receipt_path = frozen_dir / f"{args.model}_seed{args.seed}_prediction_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
