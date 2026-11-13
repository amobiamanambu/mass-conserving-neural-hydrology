#!/usr/bin/env python3
"""Audit author-source dHBV historical closure using float64 reductions.

The trained model and its float32 predictions are not changed.  This audit
only repeats the annual water-balance sums in float64 to distinguish
accumulated float32 rounding from structural mass leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))

from neuralhydrology.evaluation import get_tester  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402


MODEL = "author_dhbv"
SEEDS = (11, 29, 47)
TARGET_GAUGES = 3982
TOLERANCE_MM = 0.005
ANCHORS = ROOT / "data/forcing_only_anchors.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    args = parser.parse_args()

    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    info = selection["architectures"][MODEL]
    epoch = int(info["selected_epoch"])
    run_dir = ROOT / info["runs"][str(args.seed)]["run_dir"]
    cfg = Config(run_dir / "config.yml")
    tester = get_tester(cfg, run_dir, "test", True)
    tester._load_weights(epoch)
    model = tester.model.eval()

    anchors = np.load(ANCHORS)
    gauges = (ROOT / "data/basin_lists/test_3982.txt").read_text().split()
    if len(gauges) != TARGET_GAUGES:
        raise RuntimeError(f"Expected {TARGET_GAUGES} gauges, found {len(gauges)}")
    if not np.array_equal(anchors["gage_ids"].astype(str), np.asarray(gauges)):
        raise RuntimeError("Forcing-anchor order mismatch")

    maximum_float32_residual = 0.0
    maximum_float64_residual = 0.0
    worst_float32_gauge = ""
    worst_float64_gauge = ""

    with torch.no_grad():
        for basin_index, gauge in enumerate(gauges):
            indices = anchors["eval_time_index"][basin_index].astype(int)
            if np.any(indices < 0):
                raise RuntimeError(f"Missing forcing anchor for {gauge}")
            dataset = tester._get_dataset(gauge)
            batch = dataset.collate_fn([dataset[int(index)] for index in indices])
            historical = model(batch)

            states = tuple(state.clone() for state in model.hbv._state_cache)
            physical32 = torch.stack(states, dim=0).sum(dim=(0, 2))
            q_unrouted32 = historical["q_unrouted"]
            q_routed32 = historical["q"]
            routing32 = q_unrouted32.sum(dim=1) - q_routed32.sum(dim=1)
            precipitation32 = model._raw_forcing(batch)[:, :, 0].transpose(0, 1).sum(dim=1)
            initial32 = physical32.new_full(
                physical32.shape, 5.0 * float(model.hbv.initial_state_value)
            )
            residual32 = (
                initial32
                + precipitation32
                - historical["et"].sum(dim=1)
                - q_routed32.sum(dim=1)
                - physical32
                - routing32
            )

            physical64 = torch.stack([state.double() for state in states], dim=0).sum(dim=(0, 2))
            q_unrouted64 = q_unrouted32.double()
            q_routed64 = q_routed32.double()
            routing64 = q_unrouted64.sum(dim=1) - q_routed64.sum(dim=1)
            precipitation64 = (
                model._raw_forcing(batch)[:, :, 0].transpose(0, 1).double().sum(dim=1)
            )
            initial64 = physical64.new_full(
                physical64.shape, 5.0 * float(model.hbv.initial_state_value)
            )
            residual64 = (
                initial64
                + precipitation64
                - historical["et"].double().sum(dim=1)
                - q_routed64.sum(dim=1)
                - physical64
                - routing64
            )

            local32 = float(residual32.abs().max())
            local64 = float(residual64.abs().max())
            if local32 > maximum_float32_residual:
                maximum_float32_residual = local32
                worst_float32_gauge = gauge
            if local64 > maximum_float64_residual:
                maximum_float64_residual = local64
                worst_float64_gauge = gauge

            if (basin_index + 1) % 200 == 0 or basin_index + 1 == TARGET_GAUGES:
                print(f"{MODEL} seed {args.seed}: {basin_index + 1}/{TARGET_GAUGES}", flush=True)

    receipt = {
        "model": MODEL,
        "seed": args.seed,
        "selected_epoch": epoch,
        "target_discharge_accessed": False,
        "target_gauges": TARGET_GAUGES,
        "anchors_per_gauge": 12,
        "forcing_anchor_archive": str(ANCHORS.relative_to(ROOT)),
        "forcing_anchor_archive_sha256": sha256(ANCHORS),
        "audit": "identical float32 model outputs; annual accounting reductions repeated in float64",
        "closure_tolerance_mm": TOLERANCE_MM,
        "maximum_float32_accumulated_residual_mm": maximum_float32_residual,
        "worst_float32_gauge": worst_float32_gauge,
        "maximum_float64_accumulated_residual_mm": maximum_float64_residual,
        "worst_float64_gauge": worst_float64_gauge,
        "passes_float64_accumulation_audit": bool(maximum_float64_residual <= TOLERANCE_MM),
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    path = ROOT / f"outputs/frozen/{MODEL}_seed{args.seed}_float64_closure_audit.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
