#!/usr/bin/env python3
"""Run the complete audited analysis chain after adaptive training and selection."""

from __future__ import annotations

import hashlib
import json
import tempfile
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
import sys
PYTHON = Path(sys.executable)
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
SEEDS = (11, 29, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*arguments: str) -> None:
    free_gib = shutil.disk_usage(HERE).free / 1024**3
    if free_gib < 8:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free before {' '.join(arguments)}")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "nh_mpl_cache")
    environment["PYTHONPATH"] = str(HERE / "vendor/neuralhydrology_literature_adapter")
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    environment["NGEN_SILENT"] = "1"
    print(f"RUN {' '.join(arguments)}; free {free_gib:.1f} GiB", flush=True)
    subprocess.run([str(PYTHON), *arguments], cwd=HERE, env=environment, check=True)


def valid_receipt(path: Path, file_key: str, hash_key: str) -> bool:
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    file_path = HERE / receipt[file_key]
    return file_path.exists() and sha256(file_path) == receipt[hash_key]


def intervention(model: str, seed: int, pet_mode: str = "zero") -> None:
    pet_arguments = ["--pet-mode", pet_mode]
    if model == "hybrid_shm":
        run(
            "scripts/07_freeze_native_interventions.py", "--model", model,
            "--seed", str(seed), *pet_arguments,
        )
    elif model == "author_dhbv":
        if pet_mode == "climatology":
            # The float64 audit compares against the float32 author-core run, so make that first.
            curves = HERE / "outputs/sensitivity_pet_climatology" / f"author_dhbv_seed{seed}_pet_climatology_intervention_curves.npz"
            if not curves.exists():
                run("scripts/08_freeze_author_dhbv_interventions.py", "--seed", str(seed), *pet_arguments)
            run("scripts/08b_audit_dhbv_float64_replay.py", "--seed", str(seed), *pet_arguments)
        else:
            run("scripts/08_freeze_author_dhbv_interventions.py", "--seed", str(seed), *pet_arguments)
    else:
        run(
            "scripts/06_freeze_adapter_interventions.py", "--model", model,
            "--seed", str(seed), *pet_arguments,
        )


def main() -> None:
    selection = HERE / "provenance/development_epoch_selection.json"
    if not selection.exists():
        raise FileNotFoundError("Run 03_build_selection_receipt.py first")

    blind = HERE / "provenance/blind_target_verification.json"
    if not blind.exists():
        run("scripts/05a_verify_blind_target_dataset.py")

    frozen = HERE / "outputs/frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    # National prediction uses the parity-checked batched evaluator. MPS is
    # preferred when this script is launched outside the filesystem sandbox;
    # pass CONVERGENCE_EVAL_DEVICE=cpu when MPS is unavailable.
    device = os.environ.get("ADAPTIVE_EVAL_DEVICE", "mps")
    for model in MODELS:
        for seed in SEEDS:
            receipt = frozen / f"{model}_seed{seed}_prediction_receipt.json"
            if not valid_receipt(receipt, "frozen_prediction_archive", "frozen_prediction_archive_sha256"):
                run(
                    "scripts/05b_evaluate_blind_targets_batched.py",
                    "--model", model, "--seed", str(seed), "--device", device,
                )

    pending = []
    for model in MODELS:
        for seed in SEEDS:
            receipt = frozen / f"{model}_seed{seed}_intervention_receipt.json"
            if not valid_receipt(receipt, "curves", "curves_sha256"):
                pending.append((model, seed))
    if pending:
        with ThreadPoolExecutor(max_workers=min(3, len(pending))) as pool:
            futures = {
                pool.submit(intervention, model, seed, "zero"): (model, seed)
                for model, seed in pending
            }
            for future in as_completed(futures):
                future.result()
                print(f"COMPLETE intervention {futures[future]}", flush=True)

    # Build the prespecified three-model result, audit it independently, then
    # perform the float64 dHBV amendment and four-model analysis.
    evaluation = HERE / "outputs/evaluation"
    if not (evaluation / "national_framework_discovery.json").exists():
        run("scripts/09_open_national_target_and_analyze.py")
    if not (evaluation / "national_scientific_audit.json").exists():
        run("scripts/11_additional_scientific_diagnostics.py")
    if not (evaluation / "independent_final_audit.json").exists():
        run("scripts/12_independent_result_audit.py")
    if not (evaluation / "internal_process_diagnostic_receipt.json").exists():
        run("scripts/13_internal_process_diagnostics.py")
    for seed in SEEDS:
        receipt = frozen / f"author_dhbv_seed{seed}_float64_replay_receipt.json"
        if not valid_receipt(receipt, "curves", "curves_sha256"):
            run("scripts/08b_audit_dhbv_float64_replay.py", "--seed", str(seed))
    amended = HERE / "outputs/evaluation_amended/amended_discovery.json"
    if not amended.exists():
        run("scripts/16_post_opening_amended_analysis.py")

    # Outcome-blind forcing sensitivity: preserve every plateau-selected
    # checkpoint and antecedent state, but retain seasonal evaporative demand.
    pet_archive = HERE / "derived/pet_climatology_2003_2018.npz"
    pet_receipt = pet_archive.with_suffix(".receipt.json")
    if not valid_receipt(pet_receipt, "archive", "archive_sha256"):
        run("scripts/00c_prepare_pet_climatology.py")
    pet_dir = HERE / "outputs/sensitivity_pet_climatology"
    pet_dir.mkdir(parents=True, exist_ok=True)
    pet_pending = []
    for model in MODELS:
        for seed in SEEDS:
            receipt = pet_dir / (
                f"{model}_seed{seed}_pet_climatology_float64_replay_receipt.json"
                if model == "author_dhbv"
                else f"{model}_seed{seed}_pet_climatology_intervention_receipt.json"
            )
            valid = valid_receipt(receipt, "curves", "curves_sha256")
            if model == "author_dhbv" and valid:
                valid = bool(json.loads(receipt.read_text()).get("passes_prespecified_tolerance_in_float64"))
            if not valid:
                pet_pending.append((model, seed))
    if pet_pending:
        with ThreadPoolExecutor(max_workers=min(3, len(pet_pending))) as pool:
            futures = {
                pool.submit(intervention, model, seed, "climatology"): (model, seed)
                for model, seed in pet_pending
            }
            for future in as_completed(futures):
                future.result()
                print(f"COMPLETE climatological-PET intervention {futures[future]}", flush=True)
    run("scripts/17_compare_pet_climatology.py")
    run("scripts/18_independent_amended_result_audit.py")
    run("scripts/20_dhbv_floor_check.py")


if __name__ == "__main__":
    main()
