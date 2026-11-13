#!/usr/bin/env python3
"""Quantify the batch-composition dependence of the NeuralHydrology 1.13 MC-LSTM.

NeuralHydrology 1.13.0 normalizes the MC-LSTM cell state by one L1 norm over the
complete batch tensor (``c / (c.norm(1) + 1e-5)`` in ``_MCLSTMCell._step``), so the
gate inputs depend on which samples share a batch.  Training used random
256-sample batches drawn from 500 gauges; the stock evaluator used 256 consecutive
days of one gauge; the frozen closure replay used the 12 anchor samples of one gauge.
This post-opening diagnostic evaluates the same checkpoint on a fixed set of target
gauges with the stock evaluator at batch sizes 256, 64, 12 and 1 and reports how
much the hydrographs move.  Nothing frozen is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

MODEL = "mclstm"
BATCH_SIZES = (256, 64, 12)
SINGLE_SAMPLE_GAUGES = 4
STRIDE = 250  # every 250th target gauge -> 16 gauges


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if valid.sum() < 365:
        return np.nan
    obs, sim = observed[valid], predicted[valid]
    if np.std(obs, ddof=1) <= 0 or np.mean(obs) <= 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def stock_predictions(run_dir: Path, epoch: int, gauges: list[str], batch_size: int) -> np.ndarray:
    cfg = Config(run_dir / "config.yml")
    cfg.device = "cpu"
    cfg._cfg["batch_size"] = int(batch_size)
    cfg._cfg["verbose"] = 0
    tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
    tester.basins = list(gauges)
    results = tester.evaluate(epoch=epoch, save_results=False, save_all_output=False, metrics=[])
    return np.stack([
        np.asarray(results[g]["1D"]["xr"]["q_mm_day_sim"][:, 0].values, dtype=np.float64) for g in gauges
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    info = selection["architectures"][MODEL]
    epoch = int(info["selected_epoch"])
    run_dir = ROOT / info["runs"][str(args.seed)]["run_dir"]
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    chosen = gauges[::STRIDE]
    target = np.load(ROOT / "data/source_inputs/evaluation/q_target_2019_2024.npy")
    index = {g: i for i, g in enumerate(gauges)}

    predictions = {}
    for batch_size in BATCH_SIZES:
        print(f"batch size {batch_size}: {len(chosen)} gauges", flush=True)
        predictions[batch_size] = stock_predictions(run_dir, epoch, chosen, batch_size)
    print(f"batch size 1: {SINGLE_SAMPLE_GAUGES} gauges", flush=True)
    single = stock_predictions(run_dir, epoch, chosen[:SINGLE_SAMPLE_GAUGES], 1)

    rows = []
    reference = predictions[256]
    for j, gauge in enumerate(chosen):
        obs = target[index[gauge]].astype(np.float64)
        row = {"GAGE_ID": gauge, "kge_batch256": kge(obs, reference[j])}
        for batch_size in BATCH_SIZES[1:]:
            alt = predictions[batch_size][j]
            row[f"kge_batch{batch_size}"] = kge(obs, alt)
            row[f"max_abs_diff_batch{batch_size}_vs_256_mm_day"] = float(np.nanmax(np.abs(alt - reference[j])))
            row[f"median_rel_diff_batch{batch_size}_vs_256"] = float(
                np.nanmedian(np.abs(alt - reference[j]) / np.maximum(np.abs(reference[j]), 0.01)))
            row[f"corr_batch{batch_size}_vs_256"] = float(np.corrcoef(alt, reference[j])[0, 1])
        if j < SINGLE_SAMPLE_GAUGES:
            alt = single[j]
            row["kge_batch1"] = kge(obs, alt)
            row["max_abs_diff_batch1_vs_256_mm_day"] = float(np.nanmax(np.abs(alt - reference[j])))
            row["median_rel_diff_batch1_vs_256"] = float(
                np.nanmedian(np.abs(alt - reference[j]) / np.maximum(np.abs(reference[j]), 0.01)))
            row["corr_batch1_vs_256"] = float(np.corrcoef(alt, reference[j])[0, 1])
        rows.append(row)
    frame = pd.DataFrame(rows)
    out_dir = ROOT / "outputs/sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"mclstm_seed{args.seed}_batch_sensitivity.csv", index=False)
    summary = {
        "model": MODEL,
        "seed": args.seed,
        "epoch": epoch,
        "gauges": chosen,
        "batch_sizes": list(BATCH_SIZES) + [1],
        "note": "NeuralHydrology 1.13.0 MC-LSTM normalizes the cell state by the L1 norm of the whole batch",
        "median_kge_batch256": float(np.nanmedian(frame.kge_batch256)),
        **{f"median_kge_batch{b}": float(np.nanmedian(frame[f"kge_batch{b}"])) for b in BATCH_SIZES[1:]},
        **{f"median_of_max_abs_diff_batch{b}_vs_256_mm_day": float(np.nanmedian(frame[f"max_abs_diff_batch{b}_vs_256_mm_day"])) for b in BATCH_SIZES[1:]},
        **{f"median_rel_diff_batch{b}_vs_256": float(np.nanmedian(frame[f"median_rel_diff_batch{b}_vs_256"])) for b in BATCH_SIZES[1:]},
        "median_kge_batch1_first_gauges": float(np.nanmedian(frame.kge_batch1.dropna())) if "kge_batch1" in frame else None,
        "median_rel_diff_batch1_vs_256": float(np.nanmedian(frame.median_rel_diff_batch1_vs_256.dropna())) if "median_rel_diff_batch1_vs_256" in frame else None,
    }
    (out_dir / f"mclstm_seed{args.seed}_batch_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
