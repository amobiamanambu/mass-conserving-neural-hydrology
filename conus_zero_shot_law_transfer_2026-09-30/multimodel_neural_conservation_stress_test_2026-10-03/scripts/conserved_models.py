#!/usr/bin/env python3
"""Five neural rainfall-runoff architectures with exact water accounting.

The implementations share only forcing transforms, static encoding, and a
temperature-monotone neural snow module. Their liquid-water state topology and
release operators are architecture specific.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = ("mclstm_dr", "mc_node", "mcp_ma3", "mcp_ma4", "mcp_ma5")


def dynamic_features(forcing: torch.Tensor) -> torch.Tensor:
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


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConservedHydrologyBase(nn.Module):
    """Common neural snow mass balance and exact whole-system accounting."""

    conserves_mass = True

    def __init__(self, n_static: int, liquid_states: int, static_size: int = 48) -> None:
        super().__init__()
        self.liquid_states = int(liquid_states)
        self.state_size = 1 + self.liquid_states
        self.static_size = int(static_size)
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 96),
            nn.SiLU(),
            nn.Linear(96, static_size),
            nn.Tanh(),
        )
        self.snow_gate = nn.Sequential(
            nn.Linear(4 + static_size + 1, 48),
            nn.SiLU(),
            nn.Linear(48, 2),
        )
        self.pet_scale = nn.Linear(static_size, 1)
        with torch.no_grad():
            self.snow_gate[-1].bias.copy_(torch.tensor([0.0, -3.0]))

    @staticmethod
    def _state_features(cells: torch.Tensor) -> torch.Tensor:
        total = cells.sum(dim=1, keepdim=True)
        fraction = cells / (total + 1e-5)
        magnitude = torch.log1p(torch.clamp(cells, min=0.0)) / 5.0
        total_magnitude = torch.log1p(torch.clamp(total, min=0.0)) / 7.0
        return torch.cat([fraction, magnitude, total_magnitude], dim=1)

    def _pet_demand(self, pet: torch.Tensor, static_embedding: torch.Tensor) -> torch.Tensor:
        half = 0.25 + 7.75 * torch.sigmoid(self.pet_scale(static_embedding).squeeze(1))
        return pet / (pet + half + 1e-6)

    def _snow_step(
        self,
        snow: torch.Tensor,
        p: torch.Tensor,
        temperature: torch.Tensor,
        dynamic: torch.Tensor,
        static_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        snow_features = torch.cat(
            [
                dynamic[:, 0:2],
                dynamic[:, 3:5],
                static_embedding,
                (torch.log1p(torch.clamp(snow, min=0.0)) / 5.0)[:, None],
            ],
            dim=1,
        )
        residual = self.snow_gate(snow_features)
        snow_fraction = torch.sigmoid(
            2.0 * torch.tanh(residual[:, 0]) - 1.5 * temperature
        )
        snowfall = p * snow_fraction
        rain = p - snowfall
        snow_available = snow + snowfall
        melt_fraction = torch.sigmoid(
            2.0 * torch.tanh(residual[:, 1]) + temperature
        )
        melt = snow_available * melt_fraction
        return snow_available - melt, rain + melt

    def liquid_step(
        self,
        cells: torch.Tensor,
        liquid: torch.Tensor,
        pet: torch.Tensor,
        dynamic: torch.Tensor,
        static_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
            cells = forcing.new_zeros((batch, self.liquid_states))
        else:
            if initial_state.shape[1] != self.state_size:
                raise ValueError(
                    f"{self.architecture} expects {self.state_size} state columns"
                )
            snow = initial_state[:, 0]
            cells = initial_state[:, 1:]

        initial_total = snow + cells.sum(dim=1)
        static_embedding = self.static_encoder(static)
        transformed = dynamic_features(forcing)
        q_values: list[torch.Tensor] = []
        et_values: list[torch.Tensor] = []
        loss_values: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []

        for day in range(forcing.shape[1]):
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            temperature = forcing[:, day, 2]
            before = snow + cells.sum(dim=1)
            dynamic = transformed[:, day]
            snow, liquid = self._snow_step(
                snow, p, temperature, dynamic, static_embedding
            )
            cells, q, et, loss = self.liquid_step(
                cells, liquid, pet, dynamic, static_embedding
            )
            after = snow + cells.sum(dim=1)
            q_values.append(q)
            et_values.append(et)
            loss_values.append(loss)
            if return_states:
                states.append(torch.cat([snow[:, None], cells], dim=1))
                residuals.append(before + p - q - et - loss - after)

        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        final_state = torch.cat([snow[:, None], cells], dim=1)
        result = {"q": q, "et": et, "loss": loss, "final_state": final_state}
        if return_states:
            result["state"] = torch.stack(states, dim=1)
            result["storage"] = result["state"].sum(dim=-1)
            result["daily_residual"] = torch.stack(residuals, dim=1)
            result["cumulative_residual"] = (
                initial_total
                + forcing[..., 0].sum(dim=1)
                - q.sum(dim=1)
                - et.sum(dim=1)
                - loss.sum(dim=1)
                - final_state.sum(dim=1)
            )
        return result


class MCLSTMDR(ConservedHydrologyBase):
    """Canonical MC-LSTM with time-dependent stochastic redistribution."""

    architecture = "mclstm_dr"
    family = "MC-LSTM"

    def __init__(self, n_static: int, hidden_size: int = 24) -> None:
        super().__init__(n_static, liquid_states=hidden_size)
        self.hidden_size = int(hidden_size)
        gate_input = 5 + self.static_size + 2 * hidden_size + 1
        self.input_gate = nn.Sequential(
            nn.Linear(gate_input, 64), nn.SiLU(), nn.Linear(64, hidden_size)
        )
        self.output_gate = nn.Sequential(
            nn.Linear(gate_input, 64), nn.SiLU(), nn.Linear(64, hidden_size)
        )
        self.redistribution_gate = nn.Sequential(
            nn.Linear(gate_input, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_size * hidden_size),
        )
        with torch.no_grad():
            self.output_gate[-1].bias.fill_(-2.8)
            bias = torch.full((hidden_size, hidden_size), -2.5)
            bias.fill_diagonal_(2.5)
            self.redistribution_gate[-1].bias.copy_(bias.reshape(-1))

    def liquid_step(self, cells, liquid, pet, dynamic, static_embedding):
        features = torch.cat(
            [dynamic, static_embedding, self._state_features(cells)], dim=1
        )
        input_distribution = torch.softmax(self.input_gate(features), dim=1)
        # R[dest, source] is column stochastic, matching the MC-LSTM theorem.
        redistribution = self.redistribution_gate(features).reshape(
            len(cells), self.hidden_size, self.hidden_size
        )
        redistribution = torch.softmax(redistribution, dim=1)
        redistributed = torch.bmm(redistribution, cells[:, :, None]).squeeze(2)
        available = redistributed + liquid[:, None] * input_distribution
        raw_output = torch.sigmoid(self.output_gate(features))
        demand = self._pet_demand(pet, static_embedding)
        output = available * raw_output
        et = output[:, 0] * demand
        loss = output[:, 1]
        q = output[:, 2:].sum(dim=1)
        new_cells = available - output
        # PET-limited mass remains in the typed ET cell.
        new_cells = torch.cat(
            [(new_cells[:, 0] + output[:, 0] * (1.0 - demand))[:, None], new_cells[:, 1:]],
            dim=1,
        )
        return new_cells, q, et, loss


class MCNeuralODE(ConservedHydrologyBase):
    """M100-style neural process ODE with conservative hazard integration."""

    architecture = "mc_node"
    family = "Hydrologic neural ODE"

    def __init__(self, n_static: int, substeps: int = 4) -> None:
        super().__init__(n_static, liquid_states=2)
        self.substeps = int(substeps)
        inputs = 5 + self.static_size + 5
        self.process_net = nn.Sequential(
            nn.Linear(inputs, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Linear(96, 6),
        )
        with torch.no_grad():
            self.process_net[-1].bias.copy_(
                torch.tensor([-3.0, -2.5, -3.0, -6.0, -4.0, -6.0])
            )

    @staticmethod
    def _hazard_flux(
        available: torch.Tensor, rates: torch.Tensor, dt: float
    ) -> torch.Tensor:
        total_rate = rates.sum(dim=1)
        released = available * (1.0 - torch.exp(-total_rate * dt))
        return released[:, None] * rates / (total_rate[:, None] + 1e-8)

    def liquid_step(self, cells, liquid, pet, dynamic, static_embedding):
        upper, lower = cells.unbind(dim=1)
        q_total = torch.zeros_like(upper)
        et_total = torch.zeros_like(upper)
        loss_total = torch.zeros_like(upper)
        dt = 1.0 / self.substeps
        for _ in range(self.substeps):
            state = torch.stack([upper, lower], dim=1)
            features = torch.cat(
                [dynamic, static_embedding, self._state_features(state)], dim=1
            )
            rate = F.softplus(self.process_net(features))
            demand = self._pet_demand(pet, static_embedding)
            # upper: fast Q, recharge, ET, loss; lower: slow Q, loss
            upper_rates = torch.stack(
                [rate[:, 0], rate[:, 1], rate[:, 2] * demand, rate[:, 3]], dim=1
            )
            upper_available = upper + liquid / self.substeps
            upper_flux = self._hazard_flux(upper_available, upper_rates, dt)
            upper = upper_available - upper_flux.sum(dim=1)
            recharge = upper_flux[:, 1]
            lower_rates = torch.stack([rate[:, 4], rate[:, 5]], dim=1)
            lower_available = lower + recharge
            lower_flux = self._hazard_flux(lower_available, lower_rates, dt)
            lower = lower_available - lower_flux.sum(dim=1)
            q_total = q_total + upper_flux[:, 0] + lower_flux[:, 0]
            et_total = et_total + upper_flux[:, 2]
            loss_total = loss_total + upper_flux[:, 3] + lower_flux[:, 1]
        return torch.stack([upper, lower], dim=1), q_total, et_total, loss_total


class MCPGraphBase(ConservedHydrologyBase):
    family = "MCP directed graph"

    def _features(self, cells, dynamic, static_embedding):
        return torch.cat(
            [dynamic, static_embedding, self._state_features(cells)], dim=1
        )

    @staticmethod
    def _pet_limit(fractions, et_index, demand):
        raw_et = fractions[:, et_index]
        fractions = fractions.clone()
        fractions[:, et_index] = raw_et * demand
        fractions[:, 0] = fractions[:, 0] + raw_et * (1.0 - demand)
        return fractions


class MCPMA3(MCPGraphBase):
    """Published MA3 topology: partitioning state and serial routing state."""

    architecture = "mcp_ma3"

    def __init__(self, n_static: int) -> None:
        super().__init__(n_static, liquid_states=2)
        inputs = 5 + self.static_size + 5
        self.soil_gate = nn.Sequential(
            nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 4)
        )
        self.route_gate = nn.Sequential(
            nn.Linear(inputs, 48), nn.SiLU(), nn.Linear(48, 3)
        )
        with torch.no_grad():
            self.soil_gate[-1].bias.copy_(torch.tensor([2.2, -1.8, -3.0, -6.0]))
            self.route_gate[-1].bias.copy_(torch.tensor([2.2, -2.2, -6.0]))

    def liquid_step(self, cells, liquid, pet, dynamic, static_embedding):
        soil, route = cells.unbind(dim=1)
        features = self._features(cells, dynamic, static_embedding)
        soil_fraction = torch.softmax(self.soil_gate(features), dim=1)
        soil_fraction = self._pet_limit(
            soil_fraction, 2, self._pet_demand(pet, static_embedding)
        )
        soil_available = soil + liquid
        soil_flux = soil_available[:, None] * soil_fraction
        soil = soil_flux[:, 0]
        to_route, et, loss = soil_flux[:, 1], soil_flux[:, 2], soil_flux[:, 3]
        route_available = route + to_route
        route_fraction = torch.softmax(self.route_gate(features), dim=1)
        route_flux = route_available[:, None] * route_fraction
        route = route_flux[:, 0]
        q = route_flux[:, 1]
        loss = loss + route_flux[:, 2]
        return torch.stack([soil, route], dim=1), q, et, loss


class MCPMA4(MCPGraphBase):
    """Published MA4 topology: direct fast flow plus routed groundwater."""

    architecture = "mcp_ma4"

    def __init__(self, n_static: int) -> None:
        super().__init__(n_static, liquid_states=2)
        inputs = 5 + self.static_size + 5
        self.soil_gate = nn.Sequential(
            nn.Linear(inputs, 64), nn.SiLU(), nn.Linear(64, 5)
        )
        self.groundwater_gate = nn.Sequential(
            nn.Linear(inputs, 48), nn.SiLU(), nn.Linear(48, 3)
        )
        with torch.no_grad():
            self.soil_gate[-1].bias.copy_(
                torch.tensor([2.2, -2.4, -2.2, -3.0, -6.0])
            )
            self.groundwater_gate[-1].bias.copy_(torch.tensor([2.8, -3.0, -6.0]))

    def liquid_step(self, cells, liquid, pet, dynamic, static_embedding):
        soil, groundwater = cells.unbind(dim=1)
        features = self._features(cells, dynamic, static_embedding)
        soil_fraction = torch.softmax(self.soil_gate(features), dim=1)
        soil_fraction = self._pet_limit(
            soil_fraction, 3, self._pet_demand(pet, static_embedding)
        )
        soil_available = soil + liquid
        soil_flux = soil_available[:, None] * soil_fraction
        soil = soil_flux[:, 0]
        direct_q, recharge, et, loss = (
            soil_flux[:, 1],
            soil_flux[:, 2],
            soil_flux[:, 3],
            soil_flux[:, 4],
        )
        groundwater_available = groundwater + recharge
        groundwater_fraction = torch.softmax(self.groundwater_gate(features), dim=1)
        groundwater_flux = groundwater_available[:, None] * groundwater_fraction
        groundwater = groundwater_flux[:, 0]
        q = direct_q + groundwater_flux[:, 1]
        loss = loss + groundwater_flux[:, 2]
        return torch.stack([soil, groundwater], dim=1), q, et, loss


class MCPMA5(MCPGraphBase):
    """Published HyMod-like MA5 topology with channel and groundwater paths."""

    architecture = "mcp_ma5"

    def __init__(self, n_static: int) -> None:
        super().__init__(n_static, liquid_states=3)
        inputs = 5 + self.static_size + 7
        self.soil_gate = nn.Sequential(
            nn.Linear(inputs, 72), nn.SiLU(), nn.Linear(72, 5)
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(inputs, 48), nn.SiLU(), nn.Linear(48, 3)
        )
        self.groundwater_gate = nn.Sequential(
            nn.Linear(inputs, 48), nn.SiLU(), nn.Linear(48, 3)
        )
        with torch.no_grad():
            self.soil_gate[-1].bias.copy_(
                torch.tensor([2.2, -2.0, -2.4, -3.0, -6.0])
            )
            self.channel_gate[-1].bias.copy_(torch.tensor([2.0, -2.0, -6.0]))
            self.groundwater_gate[-1].bias.copy_(torch.tensor([2.8, -3.0, -6.0]))

    def liquid_step(self, cells, liquid, pet, dynamic, static_embedding):
        soil, channel, groundwater = cells.unbind(dim=1)
        features = self._features(cells, dynamic, static_embedding)
        soil_fraction = torch.softmax(self.soil_gate(features), dim=1)
        soil_fraction = self._pet_limit(
            soil_fraction, 3, self._pet_demand(pet, static_embedding)
        )
        soil_available = soil + liquid
        soil_flux = soil_available[:, None] * soil_fraction
        soil = soil_flux[:, 0]
        to_channel, recharge, et, loss = (
            soil_flux[:, 1],
            soil_flux[:, 2],
            soil_flux[:, 3],
            soil_flux[:, 4],
        )
        channel_available = channel + to_channel
        channel_fraction = torch.softmax(self.channel_gate(features), dim=1)
        channel_flux = channel_available[:, None] * channel_fraction
        channel = channel_flux[:, 0]
        groundwater_available = groundwater + recharge
        groundwater_fraction = torch.softmax(self.groundwater_gate(features), dim=1)
        groundwater_flux = groundwater_available[:, None] * groundwater_fraction
        groundwater = groundwater_flux[:, 0]
        q = channel_flux[:, 1] + groundwater_flux[:, 1]
        loss = loss + channel_flux[:, 2] + groundwater_flux[:, 2]
        return torch.stack([soil, channel, groundwater], dim=1), q, et, loss


def make_model(name: str, n_static: int) -> ConservedHydrologyBase:
    classes = {
        "mclstm_dr": MCLSTMDR,
        "mc_node": MCNeuralODE,
        "mcp_ma3": MCPMA3,
        "mcp_ma4": MCPMA4,
        "mcp_ma5": MCPMA5,
    }
    if name not in classes:
        raise KeyError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return classes[name](n_static=n_static)
