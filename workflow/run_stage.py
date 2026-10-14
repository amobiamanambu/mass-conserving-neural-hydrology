#!/usr/bin/env python3
"""Inspect or execute the canonical audited stage index."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "workflow/stages.json"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stage", help="exact stage identifier")
    selection.add_argument("--group", help="run every stage in a workflow group")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute commands; without this flag the runner is a dry run",
    )
    return parser.parse_args()


def command_for(stage: dict[str, object], project: Path) -> list[str]:
    values = []
    for token in stage["argv"]:
        values.append(str(token).replace("{python}", sys.executable).replace("{project}", str(project)))
    return values


def main() -> None:
    args = arguments()
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    stages = index["stages"]
    if args.list:
        for stage in stages:
            print(f"{stage['stage_id']:<48} {stage['group']:<12} {stage['description']}")
        return
    if args.stage:
        chosen = [stage for stage in stages if stage["stage_id"] == args.stage]
    elif args.group:
        chosen = [stage for stage in stages if stage["group"] == args.group]
    else:
        raise SystemExit("Select --list, --stage STAGE_ID, or --group GROUP")
    if not chosen:
        raise SystemExit("No matching workflow stages")

    project = ROOT / index["project_directory"]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    for stage in chosen:
        cwd = project / stage["cwd"]
        command = command_for(stage, project)
        print(f"[{stage['stage_id']}] cwd={cwd.relative_to(ROOT)}")
        print("  " + " ".join(command))
        if args.execute:
            subprocess.run(command, cwd=cwd, env=environment, check=True)


if __name__ == "__main__":
    main()
