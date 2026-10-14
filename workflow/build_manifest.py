#!/usr/bin/env python3
"""Build the repository SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.csv"
EXCLUDED = {"MANIFEST.csv", "VALIDATION_REPORT.json"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
EXCLUDED_NAMES = {".DS_Store"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or path.name in EXCLUDED_NAMES or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in included_files():
            writer.writerow([
                path.relative_to(ROOT).as_posix(),
                path.stat().st_size,
                digest(path),
            ])
    print(f"Wrote {OUTPUT} with {len(included_files()):,} files")


if __name__ == "__main__":
    main()
