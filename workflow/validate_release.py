#!/usr/bin/env python3
"""Validate scope, provenance, hashes, and headline result consistency."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "VALIDATION_REPORT.json"
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".sh", ".txt", ".yaml", ".yml"}
REQUIRED = (
    "README.md",
    "CITATION.cff",
    "environment.yml",
    "adaptive_training/PROTOCOL.md",
    "adaptive_training/provenance/development_epoch_selection.json",
    "adaptive_training/outputs/evaluation_amended/amended_architecture_summary.csv",
    "adaptive_training/outputs/evaluation_amended/amended_pairwise_summary.csv",
    "adaptive_training/outputs/evaluation_pet_climatology/pet_climatology_pairwise_summary.csv",
    "vendor/neuralhydrology_literature_adapter/LICENSE",
    "vendor/mcp_author_source/LICENSE",
)
BANNED_PARTS = {
    "manuscript",
    "manuscript_revision_2026-11-06",
    "reproducible_figures",
    "archive_mcp_failed_2026-11-10",
    "conus_zero_shot_law_transfer_2026-09-30",
    "conus_daily_process_observability_2026-10-12",
}
BANNED_SUFFIXES = {".docx", ".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".svg", ".html"}
PERSONAL_PATHS = ("/Users/" + "amobichukwu", "/opt/" + "miniconda3/envs/neuralhydrology")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def basin_count(name: str) -> int:
    path = ROOT / "data/basin_lists" / name
    return sum(bool(line.strip()) for line in path.read_text().splitlines())


def validate_manifest(failures: list[str]) -> int:
    manifest = ROOT / "MANIFEST.csv"
    check(manifest.exists(), "MANIFEST.csv is missing", failures)
    if not manifest.exists():
        return 0
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row["path"]
        check(path.is_file(), f"Manifest file missing: {row['path']}", failures)
        if path.is_file():
            check(path.stat().st_size == int(row["bytes"]), f"Size mismatch: {row['path']}", failures)
            check(sha256(path) == row["sha256"], f"Hash mismatch: {row['path']}", failures)
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and path.name not in {"MANIFEST.csv", "VALIDATION_REPORT.json"}
    }
    check({row["path"] for row in rows} == expected, "Manifest inventory is incomplete or stale", failures)
    return len(rows)


def main() -> None:
    failures: list[str] = []
    for relative in REQUIRED:
        check((ROOT / relative).is_file(), f"Required file missing: {relative}", failures)

    release_files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in release_files:
        relative = path.relative_to(ROOT)
        lowered = {part.lower() for part in relative.parts}
        check(not (lowered & BANNED_PARTS), f"Excluded project material present: {relative}", failures)
        check(path.suffix.lower() not in BANNED_SUFFIXES, f"Publication artifact present: {relative}", failures)
        check(path.stat().st_size < 50 * 1024 * 1024, f"GitHub file exceeds 50 MiB: {relative}", failures)
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            for marker in PERSONAL_PATHS:
                check(marker not in text, f"Absolute workstation path in {relative}", failures)

    check(basin_count("train_500.txt") == 500, "Training population is not 500", failures)
    check(basin_count("validation_200.txt") == 200, "Development population is not 200", failures)
    check(basin_count("test_3982.txt") == 3982, "Target population is not 3,982", failures)

    architecture_path = ROOT / "adaptive_training/outputs/evaluation_amended/amended_architecture_summary.csv"
    if architecture_path.exists():
        architecture = pd.read_csv(architecture_path)
        expected_models = {"hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate", "mclstm"}
        check(set(architecture["model"]) == expected_models, "Architecture summary model set changed", failures)
        admitted = set(architecture.loc[architecture["admitted_under_float64_audit"], "model"])
        check(admitted == expected_models - {"mclstm"}, "Closure-admitted model set changed", failures)
        expected_kge = {
            "hybrid_shm": 0.6056297422472267,
            "author_dhbv": 0.6001393539230612,
            "author_mcp_ma3": 0.4776231947574482,
            "ported_fluxgate": 0.5895487670818396,
        }
        for model, value in expected_kge.items():
            found = float(architecture.loc[architecture.model == model, "median_kge"].iloc[0])
            check(math.isclose(found, value, rel_tol=0, abs_tol=1e-12), f"Median KGE changed for {model}", failures)

    pair_path = ROOT / "adaptive_training/outputs/evaluation_amended/amended_pairwise_summary.csv"
    if pair_path.exists():
        pairs = pd.read_csv(pair_path)
        high_skill = pairs[(pairs.skill_threshold == 0.6) & pairs.pair_in_amended_analysis]
        check(len(high_skill) == 6, "Expected six amended high-skill model pairs", failures)

    pet_path = ROOT / "adaptive_training/outputs/evaluation_pet_climatology/pet_climatology_pairwise_summary.csv"
    if pet_path.exists():
        pet = pd.read_csv(pet_path)
        check(set(pet["pet_mode"]) == {"zero", "climatology"}, "PET modes changed", failures)
        check(len(pet[pet.skill_threshold == 0.6]) == 12, "Expected twelve high-skill PET rows", failures)

    registry = ROOT / "adaptive_training/provenance/preopening_freeze_registry.json"
    registry_hash = ROOT / "adaptive_training/provenance/preopening_freeze_registry.sha256"
    if registry.exists() and registry_hash.exists():
        check(sha256(registry) == registry_hash.read_text().strip(), "Normalized freeze-registry hash mismatch", failures)

    manifest_files = validate_manifest(failures)
    report = {
        "status": "pass" if not failures else "fail",
        "files_checked": len(release_files),
        "manifest_files": manifest_files,
        "population": {"train": 500, "development": 200, "target": 3982},
        "closure_admitted_models": ["hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate"],
        "failures": failures,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("\n".join(f"FAIL: {item}" for item in failures))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
