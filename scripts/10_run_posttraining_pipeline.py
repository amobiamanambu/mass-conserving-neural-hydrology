#!/usr/bin/env python3
"""Resume the locked national pipeline after all 15 training runs finish."""

from __future__ import annotations

import hashlib
import json
import tempfile
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def completed_run(model: str, seed: int) -> Path | None:
    candidates = [
        path for path in sorted((ROOT / "runs").glob(
            f"national_literature_{model}_seed{seed}_*"
        ))
        if (path / "model_epoch030.pt").exists()
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple completed runs for {model} seed {seed}: {candidates}")
    return candidates[0] if candidates else None


def wait_for_training() -> None:
    while True:
        missing = [
            f"{model}/seed{seed}"
            for model in MODELS
            for seed in SEEDS
            if completed_run(model, seed) is None
        ]
        if not missing:
            print("All 15 epoch-30 training runs are complete.", flush=True)
            return
        print(f"Waiting for {len(missing)} training runs: {', '.join(missing)}", flush=True)
        time.sleep(30)


def run(*arguments: str) -> None:
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    if free_gib < 8:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free before {' '.join(arguments)}")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "nh_mpl_cache")
    environment["PYTHONPATH"] = str(ROOT / "vendor/neuralhydrology_literature_adapter")
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"RUN {' '.join(arguments)}; free space {free_gib:.1f} GiB", flush=True)
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, env=environment, check=True)


def valid_receipt(receipt_path: Path, file_key: str, hash_key: str) -> bool:
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    path = ROOT / receipt[file_key]
    return path.exists() and sha256(path) == receipt[hash_key]


def freeze_intervention(model: str, seed: int) -> None:
    """Run one independent model-seed intervention and closure audit."""
    if model in {"hybrid_shm", "mclstm"}:
        run("scripts/07_freeze_native_interventions.py", "--model", model, "--seed", str(seed))
    elif model == "author_dhbv":
        run("scripts/08_freeze_author_dhbv_interventions.py", "--seed", str(seed))
    else:
        run("scripts/06_freeze_adapter_interventions.py", "--model", model, "--seed", str(seed))


def main() -> None:
    wait_for_training()
    run("scripts/04_select_development_epoch.py")
    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    if selection.get("all_three_seeds_complete") is not True:
        raise RuntimeError("Final development-selection receipt is incomplete")

    blind_receipt = ROOT / "provenance/blind_target_verification.json"
    if not blind_receipt.exists():
        run("scripts/05a_verify_blind_target_dataset.py")

    frozen = ROOT / "outputs/frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for seed in SEEDS:
            receipt = frozen / f"{model}_seed{seed}_prediction_receipt.json"
            if valid_receipt(receipt, "frozen_prediction_archive", "frozen_prediction_archive_sha256"):
                print(f"SKIP frozen prediction {model} seed {seed}", flush=True)
            else:
                run(
                    "scripts/05b_evaluate_blind_targets_batched.py",
                    "--model", model,
                    "--seed", str(seed),
                    "--device", "mps",
                )

    pending_interventions: list[tuple[str, int]] = []
    for model in MODELS:
        for seed in SEEDS:
            receipt = frozen / f"{model}_seed{seed}_intervention_receipt.json"
            if valid_receipt(receipt, "curves", "curves_sha256"):
                print(f"SKIP frozen intervention {model} seed {seed}", flush=True)
            else:
                pending_interventions.append((model, seed))

    # Model-seed interventions are scientifically independent and write to
    # distinct archives. Parallel execution changes only wall-clock time; each
    # worker retains the original per-gauge 12-anchor batch semantics.
    with ThreadPoolExecutor(max_workers=min(3, len(pending_interventions))) as pool:
        futures = {
            pool.submit(freeze_intervention, model, seed): (model, seed)
            for model, seed in pending_interventions
        }
        for future in as_completed(futures):
            model, seed = futures[future]
            future.result()
            receipt = frozen / f"{model}_seed{seed}_intervention_receipt.json"
            if not valid_receipt(receipt, "curves", "curves_sha256"):
                raise RuntimeError(f"Invalid intervention receipt after {model} seed {seed}")
            print(f"COMPLETE frozen intervention {model} seed {seed}", flush=True)

    run("scripts/09_open_national_target_and_analyze.py")
    run("scripts/12_independent_result_audit.py")


if __name__ == "__main__":
    main()
