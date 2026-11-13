#!/usr/bin/env python3
"""Build the companion Zenodo archives from the frozen project workspace.

The archives overlay a checkout of the GitHub release. Large inputs and model
products are divided into independently downloadable ZIP64 files. Manuscript
materials, rendered figures, failed MCP runs, and superseded experiments are
excluded by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ADAPTIVE_SOURCE = "adaptive_training_2026-11-10"
ADAPTIVE_RELEASE = "adaptive_training"
ARCHIVE_NAMES = (
    "mcnh_code_snapshot.zip",
    "mcnh_model_ready_inputs.zip",
    "mcnh_source_inputs.zip",
    "mcnh_initial_runs_and_mclstm.zip",
    "mcnh_selected_adaptive_checkpoints.zip",
    "mcnh_frozen_predictions.zip",
    "mcnh_zero_pet_interventions.zip",
    "mcnh_dhbv_float64_replays.zip",
    "mcnh_pet_climatology_interventions.zip",
    "mcnh_full_process_trajectories.zip",
)


@dataclass(frozen=True)
class Member:
    source: Path
    archive_path: Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def files_below(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def project_member(source_project: Path, path: Path) -> Member:
    relative = path.relative_to(source_project)
    if relative.parts[0] == ADAPTIVE_SOURCE:
        relative = Path(ADAPTIVE_RELEASE, *relative.parts[1:])
    return Member(path, relative)


def unique_members(members: Iterable[Member]) -> list[Member]:
    by_destination: dict[Path, Member] = {}
    for member in members:
        existing = by_destination.get(member.archive_path)
        if existing is not None and existing.source != member.source:
            raise RuntimeError(f"Archive collision: {existing} versus {member}")
        by_destination[member.archive_path] = member
    return [by_destination[key] for key in sorted(by_destination)]


def write_zip(path: Path, members: Iterable[Member]) -> list[Member]:
    selected = unique_members(members)
    if path.exists():
        path.unlink()
    print(f"Building {path.name} ({len(selected):,} files)", flush=True)
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for member in selected:
            archive.write(member.source, member.archive_path.as_posix())
    return selected


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release inputs:\n" + "\n".join(missing))


def selected_checkpoint_members(source_project: Path) -> list[Member]:
    receipt_path = source_project / ADAPTIVE_SOURCE / "provenance" / "development_epoch_selection.json"
    receipt = json.loads(receipt_path.read_text())
    members: list[Member] = []
    for architecture in ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate"):
        architecture_record = receipt["architectures"][architecture]
        for seed, run_record in architecture_record["runs"].items():
            run_dir = Path(run_record["run_dir"])
            checkpoint = Path(run_record["checkpoint"])
            selected_epoch = int(architecture_record["selected_epoch"])
            if not run_dir.is_relative_to(source_project):
                raise RuntimeError(f"Selected run is outside source project: {run_dir}")
            require_paths((run_dir, checkpoint))
            candidates = [
                checkpoint,
                run_dir / f"optimizer_state_epoch{selected_epoch:03d}.pt",
                run_dir / "config.yml",
                run_dir / "output.log",
            ]
            candidates.extend(files_below(run_dir / "train_data"))
            validation_dir = run_dir / "validation" / f"model_epoch{selected_epoch:03d}"
            if validation_dir.exists():
                candidates.extend(files_below(validation_dir))
            for candidate in candidates:
                if candidate.exists() and candidate.is_file() and not candidate.is_symlink():
                    members.append(project_member(source_project, candidate))
            print(f"  selected {architecture} seed {seed} epoch {selected_epoch}", flush=True)
    return unique_members(members)


def matching_members(
    source_project: Path,
    directory: Path,
    *,
    include: tuple[str, ...] = ("*",),
    exclude_substrings: tuple[str, ...] = (),
) -> list[Member]:
    members: list[Member] = []
    for pattern in include:
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.name
            if any(text in name for text in exclude_substrings):
                continue
            members.append(project_member(source_project, path))
    return unique_members(members)


def build_code_snapshot(repo_root: Path, output: Path) -> list[Member]:
    if output.exists():
        output.unlink()
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={output}", "HEAD"],
        cwd=repo_root,
        check=True,
    )
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo_root).split(b"\0")
    return [
        Member(repo_root / raw.decode(), Path(raw.decode()))
        for raw in tracked
        if raw and (repo_root / raw.decode()).is_file()
    ]


def write_readme(output_dir: Path, commit: str) -> None:
    text = f"""# Zenodo companion archive

This directory contains the large inputs and frozen products for the study
**Can Mass-Conserving Neural Hydrologic Models Identify Transferable Catchment
Drainage Responses?**

GitHub repository: https://github.com/amobiamanambu/mass-conserving-neural-hydrology

Archived GitHub commit: `{commit}`

Extract `mcnh_code_snapshot.zip` first. Extract the remaining archives over the
same directory when the corresponding reproduction level is needed. The ZIP
member paths already match the GitHub project layout.

`ZENODO_MANIFEST.csv` gives each archive's size and SHA-256 digest.
`ZENODO_CONTENTS.csv` gives the source size and SHA-256 digest of every member.

The package excludes manuscript text, document-building code, rendered
publication figures, superseded exploratory analyses, and failed MCP runs.
The record remains private during peer review and must not be published until
the author explicitly approves publication.
"""
    (output_dir / "README_ZENODO.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_project = args.source_project.resolve()
    output_dir = args.output.resolve()
    adaptive = source_project / ADAPTIVE_SOURCE
    frozen = adaptive / "outputs" / "frozen"
    pet = adaptive / "outputs" / "sensitivity_pet_climatology"

    require_paths(
        (
            source_project / "data" / "generic_blind",
            source_project / "data" / "source_inputs",
            source_project / "data" / "basin_lists",
            source_project / "data" / "forcing_only_anchors.npz",
            source_project / "runs",
            source_project / "outputs" / "frozen",
            frozen,
            pet,
            adaptive / "provenance" / "development_epoch_selection.json",
        )
    )
    if (adaptive / "archive_mcp_failed_2026-11-10").exists():
        print("Confirmed: failed MCP archive exists in source and will be excluded.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Member]] = {}

    groups[ARCHIVE_NAMES[0]] = build_code_snapshot(repo_root, output_dir / ARCHIVE_NAMES[0])

    model_ready_paths = (
        files_below(source_project / "data" / "generic_blind")
        + files_below(source_project / "data" / "basin_lists")
        + [source_project / "data" / "forcing_only_anchors.npz"]
    )
    groups[ARCHIVE_NAMES[1]] = write_zip(
        output_dir / ARCHIVE_NAMES[1],
        (project_member(source_project, path) for path in model_ready_paths),
    )
    groups[ARCHIVE_NAMES[2]] = write_zip(
        output_dir / ARCHIVE_NAMES[2],
        (project_member(source_project, path) for path in files_below(source_project / "data" / "source_inputs")),
    )

    initial_paths = files_below(source_project / "runs")
    initial_paths += sorted((source_project / "outputs" / "frozen").glob("mclstm*"))
    initial_paths += sorted((source_project / "outputs" / "sensitivity").glob("mclstm_seed11_batch_sensitivity.*"))
    groups[ARCHIVE_NAMES[3]] = write_zip(
        output_dir / ARCHIVE_NAMES[3],
        (project_member(source_project, path) for path in initial_paths if path.is_file() and not path.is_symlink()),
    )

    groups[ARCHIVE_NAMES[4]] = write_zip(
        output_dir / ARCHIVE_NAMES[4], selected_checkpoint_members(source_project)
    )
    groups[ARCHIVE_NAMES[5]] = write_zip(
        output_dir / ARCHIVE_NAMES[5],
        matching_members(
            source_project,
            frozen,
            include=("*_stock_predictions.npz", "*_prediction_receipt.json"),
        ),
    )
    groups[ARCHIVE_NAMES[6]] = write_zip(
        output_dir / ARCHIVE_NAMES[6],
        matching_members(
            source_project,
            frozen,
            include=("*_intervention_curves.npz", "*_fingerprints.csv", "*_intervention_receipt.json"),
            exclude_substrings=("float64_replay",),
        ),
    )

    float64_members = matching_members(
        source_project,
        frozen,
        include=("author_dhbv_*float64_replay*",),
    )
    float64_members += matching_members(
        source_project,
        pet,
        include=("author_dhbv_*float64_replay*",),
    )
    groups[ARCHIVE_NAMES[7]] = write_zip(output_dir / ARCHIVE_NAMES[7], float64_members)
    groups[ARCHIVE_NAMES[8]] = write_zip(
        output_dir / ARCHIVE_NAMES[8],
        matching_members(
            source_project,
            pet,
            exclude_substrings=("float64_replay",),
        ),
    )

    trajectory = adaptive / "outputs" / "evaluation" / "internal_process_trajectory_gauge.csv.gz"
    require_paths((trajectory,))
    groups[ARCHIVE_NAMES[9]] = write_zip(
        output_dir / ARCHIVE_NAMES[9], [project_member(source_project, trajectory)]
    )

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    descriptions = {
        ARCHIVE_NAMES[0]: f"Exact GitHub source snapshot at commit {commit}",
        ARCHIVE_NAMES[1]: "Model-ready NeuralHydrology inputs, basin lists, and forcing-only anchors",
        ARCHIVE_NAMES[2]: "Archived source tables and arrays used to construct the model-ready inputs",
        ARCHIVE_NAMES[3]: "Initial epoch-30 runs and the descriptive MC-LSTM products",
        ARCHIVE_NAMES[4]: "Selected adaptive checkpoints, optimizer states, scalers, and validation products",
        ARCHIVE_NAMES[5]: "Frozen target predictions for the four closure-admitted architectures",
        ARCHIVE_NAMES[6]: "Prespecified zero-PET intervention curves, fingerprints, and receipts",
        ARCHIVE_NAMES[7]: "Float64 dHBV replays for zero-PET and climatological-PET probes",
        ARCHIVE_NAMES[8]: "Climatological-PET sensitivity curves, fingerprints, and receipts",
        ARCHIVE_NAMES[9]: "Full gauge-level internal-process trajectory table omitted from GitHub",
    }

    content_rows: list[dict[str, object]] = []
    for archive_name, members in groups.items():
        for member in members:
            content_rows.append(
                {
                    "archive": archive_name,
                    "path": member.archive_path.as_posix(),
                    "source_bytes": member.source.stat().st_size,
                    "source_sha256": sha256(member.source),
                }
            )
    with (output_dir / "ZENODO_CONTENTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("archive", "path", "source_bytes", "source_sha256"))
        writer.writeheader()
        writer.writerows(content_rows)

    manifest_rows = []
    for archive_name in ARCHIVE_NAMES:
        archive_path = output_dir / archive_name
        manifest_rows.append(
            {
                "archive": archive_name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256(archive_path),
                "description": descriptions[archive_name],
            }
        )
    with (output_dir / "ZENODO_MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("archive", "bytes", "sha256", "description"))
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_readme(output_dir, commit)

    print(f"Built {len(ARCHIVE_NAMES)} archives in {output_dir}", flush=True)
    print(f"GitHub commit: {commit}", flush=True)


if __name__ == "__main__":
    main()
