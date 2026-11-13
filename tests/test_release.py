from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_population_split() -> None:
    counts = {}
    for filename in ("train_500.txt", "validation_200.txt", "test_3982.txt"):
        counts[filename] = sum(
            bool(line.strip())
            for line in (ROOT / "data/basin_lists" / filename).read_text().splitlines()
        )
    assert counts == {"train_500.txt": 500, "validation_200.txt": 200, "test_3982.txt": 3982}


def test_adaptive_selection_is_outcome_blind() -> None:
    selection = json.loads(
        (ROOT / "adaptive_training/provenance/development_epoch_selection.json").read_text()
    )
    assert selection["target_discharge_accessed_for_stopping_or_selection"] is False
    assert selection["all_three_seeds_complete"] is True
    selected = {
        model: selection["architectures"][model]["selected_epoch"]
        for model in ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
    }
    assert selected == {
        "hybrid_shm": 160,
        "author_dhbv": 200,
        "author_mcp_ma3": 250,
        "ported_fluxgate": 140,
    }


def test_final_model_eligibility() -> None:
    table = pd.read_csv(
        ROOT / "adaptive_training/outputs/evaluation_amended/amended_architecture_summary.csv"
    )
    admitted = set(table.loc[table["admitted_under_float64_audit"], "model"])
    assert admitted == {"hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate"}


def test_pet_sensitivity_has_both_modes() -> None:
    table = pd.read_csv(
        ROOT / "adaptive_training/outputs/evaluation_pet_climatology/pet_climatology_pairwise_summary.csv"
    )
    assert set(table["pet_mode"]) == {"zero", "climatology"}
    assert len(table[table["skill_threshold"] == 0.6]) == 12
