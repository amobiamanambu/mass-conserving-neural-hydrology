from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "conus_zero_shot_law_transfer_2026-09-30"


def test_frozen_split_counts() -> None:
    with (PRIMARY / "data/frozen/station_split.csv").open(newline="", encoding="utf-8") as stream:
        counts = Counter(row["role"] for row in csv.DictReader(stream))
    assert counts == {"donor": 500, "development": 200, "target": 3982}


def test_stage_index_has_no_publication_groups() -> None:
    index = json.loads((ROOT / "workflow/stages.json").read_text(encoding="utf-8"))
    groups = {stage["group"] for stage in index["stages"]}
    assert "manuscript" not in groups
    assert "figures" not in groups
    assert {"core", "original", "mcp12", "stress", "reference", "audit"} <= groups


def test_process_observation_matrix() -> None:
    decision = json.loads(
        (ROOT / "conus_daily_process_observability_2026-10-12/results/decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["runs_evaluated"] == 24
    assert decision["target_gauges"] == 3982
    assert {item["rung"] for item in decision["decisions"]} == {"q_sm", "q_et", "q_sm_et"}


def test_float64_audit_is_present() -> None:
    result = json.loads(
        (
            PRIMARY
            / "outputs/law_interventions_float64_audit/float64_replay_comparison.json"
        ).read_text(encoding="utf-8")
    )
    assert result
