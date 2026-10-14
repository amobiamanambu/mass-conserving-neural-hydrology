#!/usr/bin/env python3
"""Validate repository scope, frozen design, and file integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "conus_zero_shot_law_transfer_2026-09-30"
PROCESS = ROOT / "conus_daily_process_observability_2026-10-12"
MANIFEST = ROOT / "MANIFEST.csv"
MAX_FILE_BYTES = 95 * 1024 * 1024
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".toml", ".yml", ".yaml"}
FORBIDDEN_PARTS = {"manuscript", "reproducible_figures", "presentation"}
FORBIDDEN_SUFFIXES = {".docx", ".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".svg", ".html"}
LOCAL_PATH = re.compile(r"/(?:Users|home)/[^/]+/")
SECRET = re.compile(
    r"(?:github_pat_|ghp_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|"
    r"(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"]+)",
    re.IGNORECASE,
)
EXPECTED_ROLES = {"donor": 500, "development": 200, "target": 3982}
EXPECTED_MODELS = {"dhbv", "fluxgate", "mcp_ma3", "mc_node", "mclstm_dr"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def repository_files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and not EXCLUDED_PARTS.intersection(path.parts)
            and path.name != ".DS_Store"
            and path.suffix not in {".pyc", ".pyo"}
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def check_scope(errors: list[str]) -> None:
    for path in repository_files():
        relative = path.relative_to(ROOT)
        if FORBIDDEN_PARTS.intersection(relative.parts):
            errors.append(f"forbidden publication path: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden rendered artifact: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 95 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if LOCAL_PATH.search(text):
            errors.append(f"local absolute path found: {relative}")
        # Do not flag this validator merely because it contains the detection
        # expressions themselves.
        if relative.as_posix() != "workflow/validate_release.py" and SECRET.search(text):
            errors.append(f"possible credential found: {relative}")


def check_frozen_design(errors: list[str], summary: dict[str, object]) -> None:
    split_path = PRIMARY / "data/frozen/station_split.csv"
    with split_path.open(newline="", encoding="utf-8") as stream:
        roles = Counter(row["role"] for row in csv.DictReader(stream))
    summary["station_roles"] = dict(roles)
    if dict(roles) != EXPECTED_ROLES:
        errors.append(f"station role counts changed: {dict(roles)}")

    models = {"dhbv", "fluxgate"}
    stress_path = PRIMARY / "multimodel_neural_conservation_stress_test_2026-10-03/outputs/evaluation/target_skill_summary.csv"
    with stress_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("architecture") in EXPECTED_MODELS:
                models.add(row["architecture"])
    summary["primary_models"] = sorted(models)
    if models != EXPECTED_MODELS:
        errors.append(f"five-model coverage changed: {sorted(models)}")

    decision = json.loads((PROCESS / "results/decision.json").read_text(encoding="utf-8"))
    summary["process_runs"] = decision.get("runs_evaluated")
    summary["process_target_gauges"] = decision.get("target_gauges")
    if decision.get("runs_evaluated") != 24 or decision.get("target_gauges") != 3982:
        errors.append("process-observation matrix is incomplete")

    stages = json.loads((ROOT / "workflow/stages.json").read_text(encoding="utf-8"))
    summary["workflow_stages"] = len(stages["stages"])
    if len(stages["stages"]) < 180:
        errors.append("workflow stage index is incomplete")


def check_manifest(errors: list[str], summary: dict[str, object]) -> None:
    if not MANIFEST.exists():
        errors.append("MANIFEST.csv is missing")
        return
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    expected = {
        path.relative_to(ROOT).as_posix(): path
        for path in repository_files()
        if path.name not in {"MANIFEST.csv", "VALIDATION_REPORT.json"}
    }
    recorded = {row["relative_path"]: row for row in records}
    missing = sorted(set(expected) - set(recorded))
    extra = sorted(set(recorded) - set(expected))
    if missing:
        errors.append(f"manifest missing {len(missing)} files; first: {missing[0]}")
    if extra:
        errors.append(f"manifest lists {len(extra)} absent files; first: {extra[0]}")
    for name in sorted(set(expected).intersection(recorded)):
        path = expected[name]
        row = recorded[name]
        if int(row["size_bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            errors.append(f"manifest mismatch: {name}")
            if len(errors) >= 25:
                break
    summary["manifest_files"] = len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    errors: list[str] = []
    summary: dict[str, object] = {}
    check_scope(errors)
    check_frozen_design(errors, summary)
    check_manifest(errors, summary)
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "summary": summary,
        "errors": errors,
    }
    if args.report:
        destination = args.report if args.report.is_absolute() else ROOT / args.report
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
