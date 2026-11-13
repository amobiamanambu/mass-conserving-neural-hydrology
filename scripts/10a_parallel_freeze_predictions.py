#!/usr/bin/env python3
"""Run independent blinded stock-evaluator jobs concurrently.

This changes only the execution schedule. Each model--seed job still executes
``05_evaluate_blind_targets.py``, which invokes the unmodified NeuralHydrology
test evaluator and writes its own hash-verified prediction receipt.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import tempfile
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)
MAX_WORKERS = int(os.environ.get("NH_PREDICTION_WORKERS", "8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_is_valid(model: str, seed: int) -> bool:
    receipt_path = ROOT / "outputs/frozen" / f"{model}_seed{seed}_prediction_receipt.json"
    if not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    archive = ROOT / receipt["frozen_prediction_archive"]
    result = ROOT / receipt["framework_evaluator_result"]
    return (
        receipt.get("target_discharge_accessed") is False
        and archive.exists()
        and result.exists()
        and sha256(archive) == receipt["frozen_prediction_archive_sha256"]
        and sha256(result) == receipt["framework_evaluator_result_sha256"]
    )


def evaluate(job: tuple[str, int]) -> dict:
    model, seed = job
    if receipt_is_valid(model, seed):
        return {"model": model, "seed": seed, "status": "preexisting_valid_receipt"}

    log_dir = ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"prediction_{model}_seed{seed}.log"
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "nh_mpl_cache"),
    })
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                str(PYTHON),
                "scripts/05_evaluate_blind_targets.py",
                "--model", model,
                "--seed", str(seed),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Prediction failed for {model} seed {seed}; inspect {log_path}")
    if not receipt_is_valid(model, seed):
        raise RuntimeError(f"Prediction receipt failed validation for {model} seed {seed}")
    return {
        "model": model,
        "seed": seed,
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    selection = json.loads(
        (ROOT / "provenance/development_epoch_selection.json").read_text()
    )
    if selection.get("target_discharge_accessed") is not False:
        raise RuntimeError("Development selection did not preserve target blinding")
    if selection.get("all_three_seeds_complete") is not True:
        raise RuntimeError("Training is incomplete")
    if not 1 <= MAX_WORKERS <= 12:
        raise RuntimeError("NH_PREDICTION_WORKERS must be between 1 and 12")

    jobs = [(model, seed) for model in MODELS for seed in SEEDS]
    started = time.time()
    records = []
    print(f"Starting {len(jobs)} blinded stock-evaluator jobs with {MAX_WORKERS} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"PREDICTION {record['status']}: {record['model']} seed {record['seed']} "
                f"({len(records)}/{len(jobs)})",
                flush=True,
            )

    records.sort(key=lambda item: (MODELS.index(item["model"]), SEEDS.index(item["seed"])))
    receipt = {
        "purpose": "parallel execution of independent unmodified stock-evaluator jobs",
        "scientific_calculation_changed": False,
        "target_discharge_accessed": False,
        "workers": MAX_WORKERS,
        "elapsed_seconds": time.time() - started,
        "jobs": records,
    }
    path = ROOT / "provenance/parallel_prediction_execution.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
