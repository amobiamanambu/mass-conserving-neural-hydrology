#!/usr/bin/env python3
"""Validate the overflow-safe MCP partition against the pinned author equations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/neuralhydrology_literature_adapter"))
sys.path.insert(0, str(ROOT / "vendor/mcp_author_source"))

from MCPBRNN_lib_tools.HyMOD_Shared_Cases import MA3_PETconstraint_CellShared  # noqa: E402
from neuralhydrology.modelzoo.literatureconservingmodels import AuthorMCPMA3Regional  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.manual_seed(317)
    author = MA3_PETconstraint_CellShared(
        input_size=1, batch_first=True, hidden_size=1, initial_forget_bias=0
    ).double()
    days = 127
    precipitation = torch.rand(1, days, dtype=torch.float64) * 14.0
    pet = torch.rand(1, days, dtype=torch.float64) * 7.0
    author_input = torch.stack([precipitation[0], pet[0]], dim=1).unsqueeze(1)
    c_mean, c_std = 412.9139085, 77.49943867
    route_mean, route_std = 1.673117957, 3.592955064
    expected = author(author_input, c_mean, c_std, route_mean, route_std)
    names = (
        "weight_r_yom", "weight_r_ylm", "weight_r_yfm", "bias_b0_yom",
        "weight_b1_yom", "weight_b2_yom", "bias_b0_ylm", "weight_b2_ylm",
        "weight_r_yom_sw", "weight_r_yfm_sw", "bias_b0_yom_sw",
        "weight_b1_yom_sw", "weight_b2_yom_sw",
    )
    gates = torch.stack(
        [getattr(author, name).reshape(-1)[0] for name in names], dim=0
    ).unsqueeze(0)
    actual = AuthorMCPMA3Regional._core(
        precipitation,
        pet,
        gates,
        precipitation.new_tensor([c_mean]),
        precipitation.new_tensor([c_std]),
        precipitation.new_tensor([route_mean]),
        precipitation.new_tensor([route_std]),
    )
    comparisons = {
        "system_discharge": float((expected[10].squeeze(1) - actual["q"].squeeze(0)).abs().max()),
        "main_discharge": float((expected[0].squeeze(1) - actual["main_q"].squeeze(0)).abs().max()),
        "main_storage_pre_update": float((expected[1].squeeze(1) - actual["pre_state"][0, :, 0]).abs().max()),
        "routing_storage_pre_update": float((expected[11].squeeze(1) - actual["pre_state"][0, :, 1]).abs().max()),
        "pet_limited_loss": float((expected[3].squeeze(1) - actual["et"].squeeze(0)).abs().max()),
    }

    extreme = gates.detach().repeat(4, 1).clone()
    extreme[:, :3] = torch.tensor(
        [[1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0],
         [0.0, -1000.0, 1000.0], [1000.0, 999.0, 998.0]], dtype=torch.float64
    )
    extreme[:, 8:10] = torch.tensor(
        [[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, 999.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    extreme.requires_grad_(True)
    extreme_result = AuthorMCPMA3Regional._core(
        precipitation.repeat(4, 1),
        pet.repeat(4, 1),
        extreme,
        precipitation.new_full((4,), c_mean),
        precipitation.new_full((4,), c_std),
        precipitation.new_full((4,), route_mean),
        precipitation.new_full((4,), route_std),
    )
    objective = extreme_result["q"].mean() + extreme_result["et"].mean()
    objective.backward()
    extreme_finite = all(
        torch.isfinite(extreme_result[name]).all().item()
        for name in ("q", "et", "state", "daily_residual")
    ) and torch.isfinite(extreme.grad).all().item()

    tolerance = 1e-12
    receipt = {
        "source_class": "MA3_PETconstraint_CellShared",
        "partition_change": "exp(x_i)/sum(exp(x)) evaluated with torch.softmax for overflow safety",
        "mathematical_equations_changed": False,
        "author_parity_maximum_absolute_differences": comparisons,
        "author_parity_tolerance": tolerance,
        "author_parity_passes": max(comparisons.values()) <= tolerance,
        "extreme_raw_gate_range": [-1000.0, 1000.0],
        "extreme_forward_and_gradient_finite": bool(extreme_finite),
        "passes": bool(max(comparisons.values()) <= tolerance and extreme_finite),
        "adapted_source": str(
            ROOT / "vendor/neuralhydrology_literature_adapter/neuralhydrology/modelzoo/literatureconservingmodels.py"
        ),
        "adapted_source_sha256": sha256(
            ROOT / "vendor/neuralhydrology_literature_adapter/neuralhydrology/modelzoo/literatureconservingmodels.py"
        ),
    }
    output = ROOT / "provenance/stable_mcp_partition_validation.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not receipt["passes"]:
        raise RuntimeError("Stable MCP partition validation failed")


if __name__ == "__main__":
    main()
