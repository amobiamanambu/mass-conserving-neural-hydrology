#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for the GitHub release."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.csv"
EXCLUDED = {".git", "MANIFEST.csv", "VALIDATION_REPORT.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return path.is_file() and not any(part in EXCLUDED for part in relative.parts)


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if included(path):
            rows.append((path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)))
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        writer.writerows(rows)
    print(f"Wrote {OUTPUT} with {len(rows)} files")


if __name__ == "__main__":
    main()
