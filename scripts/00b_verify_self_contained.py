#!/usr/bin/env python3
"""Verify that the national NeuralHydrology experiment is self-contained."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "provenance/self_containment_audit.json"

EXPECTED_SHA256 = {
    "data/source_inputs/processed/dates.npy": "d814e728f276866470109f57db7d94becc07aa9ca24408a45b28a775be3cfe4d",
    "data/source_inputs/processed/forcing.npy": "2156194bc092ca45453a464839339aa738a92b8fc84ad196f9e1fef8fd35899e",
    "data/source_inputs/processed/q_donor_development.npy": "8c55b70e8dc74950cf897c8985121b1971e9c85ff3855321c6e751183043fb64",
    "data/source_inputs/processed/station_index.csv": "6e92bac4d8f82d9c48a6cffadbe536160b38ec615c054eddf7c4c94add8ae367",
    "data/source_inputs/processed/static_normalization.csv": "56eb80cf334992f1e3249aa62d37daeed3f11cd140f2902719e67fd39d493625",
    "data/source_inputs/frozen_split/station_split.csv": "2045e2b2e323a3e73f09ca625b45bc21d315c1e2bda9623b7a6f9d36f11187f1",
    "data/source_inputs/evaluation/q_target_2019_2024.npy": "37e021117b11a5dbcac395694f350783601794db6fb3948548466aae009f4f95",
    "data/source_inputs/observed_recessions/recession_summary.parquet": "c289e959236e8e248ce6ba5e396f4e968aa009841029cfc9a62afd20f5ed9e9a",
    "data/source_inputs/nldi/river_network_edges.csv": "b3d5e772264caf89a6b787c889e76a2453055c738675d0fe4f00b37e6a445fc9",
    "data/source_inputs/reference_only/combined_architecture_summary.csv": "989613be0165cf2ec47b05b5bdda37fa0eb0d0f6496e502fde2db3793486b350",
    "data/source_inputs/reference_only/combined_pairwise_skill_conditioned_summary.csv": "803f1c1b2030f29215603b8432f29204643897ce2f39e4c95bb4767d22474412",
    "data/source_inputs/reference_only/common_forcing_anchors.npz": "b67388aaa65760d7dd6b4821afb642300bae99eb574601f7e4d03fdfc1d36313",
    "provenance/source_parity/author_source_parity.json": "198b74aba3eff83e58c024fcc077c77ad02434929be7416b4476eb37ed27c86b",
    "provenance/source_parity/ported_model_equivalence.json": "ca57463871ef41e5cf1617caa65779145ff8e54055b768b3dcf5474fee4f7803",
    "vendor/mcp_author_source/MCPBRNN_lib_tools/HyMOD_Shared_Cases.py": "4bfc798c4b8afeabdc4b666f09087b8b7bab9fdbd6ef17bd1bc388cbcdade379",
    "vendor/reference_helpers/native_interventions.py": "d5e9cb88654133d89994b5dea30d3af7ad945cd569c2557b30713f0b3fe8f222",
    "vendor/reference_helpers/author_dhbv_interventions.py": "7eda3751d31bb415173bcc7de527b40e809fe5e757880094e3cc1d26e13c8819",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    verified: dict[str, str] = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
            continue
        actual = sha256(path)
        verified[relative] = actual
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")

    selection = json.loads((ROOT / "provenance/development_epoch_selection.json").read_text())
    checkpoint_count = 0
    for model, model_record in selection["architectures"].items():
        for seed, run_record in model_record["runs"].items():
            checkpoint = ROOT / run_record["checkpoint"]
            if not checkpoint.is_file():
                errors.append(f"missing checkpoint: {model} seed {seed}")
                continue
            checkpoint_count += 1
            if sha256(checkpoint) != run_record["checkpoint_sha256"]:
                errors.append(f"checkpoint hash mismatch: {model} seed {seed}")

    config_count = len(list((ROOT / "configs").glob("full_*_seed*.yml")))
    if config_count != 15:
        errors.append(f"expected 15 frozen configurations, found {config_count}")

    # Relative links are allowed when they resolve to files inside this archive.
    # Absolute links, broken links, and links leaving ROOT would make a copied or
    # cloned archive non-reproducible and therefore fail the audit.
    internal_symlinks: list[str] = []
    invalid_symlinks: list[str] = []
    root_resolved = ROOT.resolve()
    for path in ROOT.rglob("*"):
        if not path.is_symlink():
            continue
        relative = str(path.relative_to(ROOT))
        target = path.readlink()
        resolved = path.resolve(strict=False)
        if target.is_absolute() or not resolved.exists() or not resolved.is_relative_to(root_resolved):
            invalid_symlinks.append(relative)
        else:
            internal_symlinks.append(relative)
    if invalid_symlinks:
        errors.append(f"non-portable symbolic links: {invalid_symlinks}")

    retired_names = (
        "conus_" + "zero_shot_law_transfer_2026-09-30",
        "conus_" + "zero_shot_law_transfer_V2_2026-10-12",
        "conus_" + "identifiable_drainage_hypothesis_network_2026-10-09",
        "conus_" + "ai_law_falsification_2026-09-30",
        "conus_" + "hydrologic_law_identifiability_2026-09-30",
        "hydrologic_" + "identifiable_invariants_redesign_test_2026-10-04",
        "hydrologic_" + "identifiability_theorem_test_2026-10-04",
    )
    external_references = []
    scan_roots = [ROOT / "scripts", ROOT / "manuscript_revision_2026-11-06"]
    for scan_root in scan_roots:
        for path in scan_root.rglob("*.py"):
            text = path.read_text(errors="replace")
            if any(name in text for name in retired_names):
                external_references.append(str(path.relative_to(ROOT)))
    if external_references:
        errors.append(f"runnable code still references retired projects: {external_references}")

    frozen = ROOT / "outputs/frozen"
    frozen_receipts = len(list(frozen.glob("*_prediction_receipt.json")))
    intervention_receipts = len(list(frozen.glob("*_intervention_receipt.json")))
    if frozen_receipts != 15 or intervention_receipts != 15:
        errors.append(
            "expected 15 prediction and 15 intervention receipts, found "
            f"{frozen_receipts} and {intervention_receipts}"
        )

    report = {
        "status": "pass" if not errors else "fail",
        "project_root": str(ROOT),
        "verified_source_files": verified,
        "frozen_configurations": config_count,
        "verified_selected_checkpoints": checkpoint_count,
        "prediction_receipts": frozen_receipts,
        "intervention_receipts": intervention_receipts,
        "portable_internal_symbolic_links": internal_symlinks,
        "invalid_symbolic_links": invalid_symlinks,
        "runnable_code_references_to_retired_projects": external_references,
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("Self-containment audit failed")


if __name__ == "__main__":
    main()
