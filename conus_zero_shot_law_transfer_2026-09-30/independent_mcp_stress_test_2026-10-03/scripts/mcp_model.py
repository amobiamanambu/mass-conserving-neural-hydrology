#!/usr/bin/env python3
"""Regional multi-cell Mass-Conserving Perceptron for the independent test."""

from __future__ import annotations

import torch
from torch import nn


def dynamic_features(forcing: torch.Tensor) -> torch.Tensor:
    """Use the same outcome-independent forcing transforms as the paper models."""
    return torch.stack(
        [
            torch.log1p(torch.clamp(forcing[..., 0], min=0.0)) / 3.0,
            torch.log1p(torch.clamp(forcing[..., 1], min=0.0)) / 2.0,
            forcing[..., 2] / 20.0,
            forcing[..., 3],
            forcing[..., 4],
        ],
        dim=-1,
    )


class RegionalMCP(nn.Module):
    """A fully neural recurrent water-mass network with exact daily closure.

    State column 0 is snow mass. The remaining columns are parallel liquid
    MCP cells. For every liquid cell, a four-way softmax allocates available
    water to retention, discharge, evapotranspiration, or deep loss. At zero
    PET, the ET allocation is returned to retention, so the zero-input
    intervention cannot drain water through an ET artifact.
    """

    architecture = "mcp"
    conserves_mass = True

    def __init__(
        self,
        n_static: int,
        n_cells: int = 12,
        static_size: int = 48,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_cells = int(n_cells)
        self.state_size = 1 + self.n_cells
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 96),
            nn.SiLU(),
            nn.Linear(96, static_size),
            nn.Tanh(),
        )
        state_features = 2 * self.n_cells + 1
        gate_inputs = 5 + static_size + state_features
        self.snow_gate = nn.Sequential(
            nn.Linear(4 + static_size + 1, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2),
        )
        self.input_gate = nn.Sequential(
            nn.Linear(gate_inputs, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.n_cells),
        )
        self.flux_gate = nn.Sequential(
            nn.Linear(gate_inputs, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.n_cells * 4),
        )
        self.pet_scale = nn.Linear(static_size, 1)

        with torch.no_grad():
            self.snow_gate[-1].bias.copy_(torch.tensor([0.0, -3.0]))
            self.input_gate[-1].bias.zero_()
            allocation_bias = torch.tensor([2.2, -2.2, -3.0, -6.0])
            self.flux_gate[-1].bias.copy_(allocation_bias.repeat(self.n_cells))

    def _state_features(self, cells: torch.Tensor) -> torch.Tensor:
        total = cells.sum(dim=1, keepdim=True)
        fractions = cells / (total + 1e-5)
        magnitude = torch.log1p(torch.clamp(cells, min=0.0)) / 5.0
        total_magnitude = torch.log1p(torch.clamp(total, min=0.0)) / 7.0
        return torch.cat([fractions, magnitude, total_magnitude], dim=1)

    def forward(
        self,
        forcing: torch.Tensor,
        static: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = forcing.shape[0]
        if initial_state is None:
            snow = forcing.new_zeros(batch)
            cells = forcing.new_zeros((batch, self.n_cells))
        else:
            if initial_state.shape[1] != self.state_size:
                raise ValueError(f"Expected {self.state_size} state columns")
            snow = initial_state[:, 0]
            cells = initial_state[:, 1:]

        initial_total = snow + cells.sum(dim=1)
        static_embedding = self.static_encoder(static)
        pet_half_saturation = 0.25 + 7.75 * torch.sigmoid(
            self.pet_scale(static_embedding).squeeze(1)
        )
        q_values: list[torch.Tensor] = []
        et_values: list[torch.Tensor] = []
        loss_values: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []

        transformed = dynamic_features(forcing)
        for day in range(forcing.shape[1]):
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            before = snow + cells.sum(dim=1)
            dynamic = transformed[:, day]

            # Temperature enters as an explicit monotone term below. Excluding
            # it from the residual network prevents that network from learning
            # a warm-snowfall/cold-melt reversal while retaining neural control
            # by precipitation, PET, season, catchment, and antecedent snow.
            snow_features = torch.cat(
                [
                    dynamic[:, 0:2],
                    dynamic[:, 3:5],
                    static_embedding,
                    (torch.log1p(snow) / 5.0)[:, None],
                ],
                dim=1,
            )
            snow_logits = self.snow_gate(snow_features)
            temperature = forcing[:, day, 2]
            snow_fraction = torch.sigmoid(
                2.0 * torch.tanh(snow_logits[:, 0]) - 1.5 * temperature
            )
            snowfall = p * snow_fraction
            liquid_rain = p - snowfall
            snow_available = snow + snowfall
            melt_fraction = torch.sigmoid(
                2.0 * torch.tanh(snow_logits[:, 1]) + temperature
            )
            melt = snow_available * melt_fraction
            snow = snow_available - melt
            liquid = liquid_rain + melt

            recurrent = torch.cat(
                [dynamic, static_embedding, self._state_features(cells)], dim=1
            )
            input_fraction = torch.softmax(self.input_gate(recurrent), dim=1)
            available = cells + liquid[:, None] * input_fraction

            raw_fraction = torch.softmax(
                self.flux_gate(recurrent).reshape(batch, self.n_cells, 4), dim=2
            )
            retain_fraction = raw_fraction[:, :, 0]
            q_fraction = raw_fraction[:, :, 1]
            raw_et_fraction = raw_fraction[:, :, 2]
            loss_fraction = raw_fraction[:, :, 3]

            demand = pet / (pet + pet_half_saturation + 1e-6)
            et_fraction = raw_et_fraction * demand[:, None]
            retain_fraction = retain_fraction + raw_et_fraction * (1.0 - demand[:, None])

            q_cell = available * q_fraction
            et_cell = available * et_fraction
            loss_cell = available * loss_fraction
            cells = available * retain_fraction
            q = q_cell.sum(dim=1)
            et = et_cell.sum(dim=1)
            deep_loss = loss_cell.sum(dim=1)
            after = snow + cells.sum(dim=1)

            q_values.append(q)
            et_values.append(et)
            loss_values.append(deep_loss)
            if return_states:
                states.append(torch.cat([snow[:, None], cells], dim=1))
                residuals.append(before + p - q - et - deep_loss - after)

        final_state = torch.cat([snow[:, None], cells], dim=1)
        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        deep_loss = torch.stack(loss_values, dim=1)
        result = {
            "q": q,
            "et": et,
            "loss": deep_loss,
            "final_state": final_state,
        }
        if return_states:
            result["state"] = torch.stack(states, dim=1)
            result["storage"] = result["state"].sum(dim=-1)
            result["daily_residual"] = torch.stack(residuals, dim=1)
            result["cumulative_residual"] = (
                initial_total
                + forcing[..., 0].sum(dim=1)
                - q.sum(dim=1)
                - et.sum(dim=1)
                - deep_loss.sum(dim=1)
                - final_state.sum(dim=1)
            )
        return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
