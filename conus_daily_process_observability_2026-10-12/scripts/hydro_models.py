#!/usr/bin/env python3
"""Neural and differentiable hydrologic model families for the CONUS test."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _dynamic_features(forcing: torch.Tensor) -> torch.Tensor:
    """Scale forcing without using target-gauge outcome statistics."""
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


class EntityAwareLSTM(nn.Module):
    """Regional LSTM conditioned on static catchment attributes."""

    architecture = "ealstm"
    conserves_mass = False

    def __init__(self, n_static: int, hidden_size: int = 96, static_size: int = 48):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 96), nn.ReLU(),
            nn.Linear(96, static_size), nn.Tanh(),
        )
        self.lstm = nn.LSTM(5 + static_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, forcing: torch.Tensor, static: torch.Tensor, initial_state=None, return_states: bool = False):
        dynamic = _dynamic_features(forcing)
        catchment = self.static_encoder(static)
        catchment = catchment[:, None, :].expand(-1, forcing.shape[1], -1)
        output, final = self.lstm(torch.cat([dynamic, catchment], dim=-1), initial_state)
        q = F.softplus(self.head(output).squeeze(-1))
        result = {"q": q, "final_state": final}
        if return_states:
            result["state"] = output
        return result


class DifferentiableHBV(nn.Module):
    """Static-parameter regional HBV-like model with exact water accounting."""

    architecture = "dhbv"
    conserves_mass = True
    state_size = 5

    def __init__(self, n_static: int):
        super().__init__()
        self.parameter_net = nn.Sequential(
            nn.Linear(n_static, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def parameters_from_static(self, static: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.parameter_net(static)
        sigmoid = torch.sigmoid
        return {
            "t0": -3.0 + 6.0 * sigmoid(raw[:, 0]),
            "degree_day": 0.5 + 7.5 * sigmoid(raw[:, 1]),
            "capacity": 40.0 + 960.0 * sigmoid(raw[:, 2]),
            "beta": 0.4 + 5.6 * sigmoid(raw[:, 3]),
            "evap_shape": 0.25 + 2.75 * sigmoid(raw[:, 4]),
            "k_fast": 0.02 + 0.78 * sigmoid(raw[:, 5]),
            "percolation": 0.001 + 0.199 * sigmoid(raw[:, 6]),
            "k_slow": 0.0002 + 0.1498 * sigmoid(raw[:, 7]),
            "routing": 0.02 + 0.88 * sigmoid(raw[:, 8]),
            "deep_fraction": 0.000001 + 0.004999 * sigmoid(raw[:, 9]),
        }

    def forward(self, forcing: torch.Tensor, static: torch.Tensor, initial_state=None, return_states: bool = False):
        params = self.parameters_from_static(static)
        batch = forcing.shape[0]
        if initial_state is None:
            snow = forcing.new_zeros(batch)
            soil = forcing.new_zeros(batch)
            fast = forcing.new_zeros(batch)
            slow = forcing.new_zeros(batch)
            route = forcing.new_zeros(batch)
        else:
            snow, soil, fast, slow, route = initial_state.unbind(dim=1)
        initial_total = snow + soil + fast + slow + route
        q_values, et_values, loss_values = [], [], []
        states, residuals = [], []
        for day in range(forcing.shape[1]):
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            temperature = forcing[:, day, 2]
            before = snow + soil + fast + slow + route

            snow_fraction = torch.sigmoid(-2.0 * (temperature - params["t0"]))
            snowfall = p * snow_fraction
            rainfall = p - snowfall
            melt = torch.minimum(snow + snowfall, params["degree_day"] * F.relu(temperature - params["t0"]))
            snow = snow + snowfall - melt
            liquid = rainfall + melt

            saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
            recharge = liquid * torch.pow(saturation + 1e-6, params["beta"])
            soil = soil + liquid - recharge
            overflow = F.relu(soil - params["capacity"])
            soil = torch.minimum(soil, params["capacity"])
            evap_fraction = torch.pow(torch.clamp(soil / params["capacity"], 0.0, 1.0) + 1e-6, params["evap_shape"])
            evap = torch.minimum(soil, pet * evap_fraction)
            soil = soil - evap

            fast = fast + recharge + overflow
            percolation = fast * params["percolation"]
            fast = fast - percolation
            q_fast = fast * params["k_fast"]
            fast = fast - q_fast
            slow = slow + percolation
            deep_loss = slow * params["deep_fraction"]
            slow = slow - deep_loss
            q_slow = slow * params["k_slow"]
            slow = slow - q_slow
            route = route + q_fast + q_slow
            q = route * params["routing"]
            route = route - q

            after = snow + soil + fast + slow + route
            q_values.append(q); et_values.append(evap); loss_values.append(deep_loss)
            if return_states:
                states.append(torch.stack([snow, soil, fast, slow, route], dim=1))
                residuals.append(before + p - q - evap - deep_loss - after)

        final_state = torch.stack([snow, soil, fast, slow, route], dim=1)
        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        result = {"q": q, "et": et, "loss": loss, "final_state": final_state}
        if return_states:
            result["state"] = torch.stack(states, dim=1)
            result["storage"] = result["state"].sum(dim=-1)
            result["daily_residual"] = torch.stack(residuals, dim=1)
            result["cumulative_residual"] = (
                initial_total + forcing[..., 0].sum(dim=1)
                - q.sum(dim=1) - et.sum(dim=1) - loss.sum(dim=1) - final_state.sum(dim=1)
            )
        return result


class TypedMassConservingLSTM(nn.Module):
    """MC-LSTM with explicit ET, deep-loss, and discharge output cells.

    Precipitation is distributed over mass cells, stored mass is redistributed
    by a row-stochastic matrix, and an output gate releases a nonnegative
    fraction. Cell 0 is interpreted as evapotranspiration, cell 1 as unobserved
    deep exchange, and the remaining cells sum to streamflow.
    """

    architecture = "mclstm"
    conserves_mass = True

    def __init__(self, n_static: int, hidden_size: int = 32, static_size: int = 48):
        super().__init__()
        if hidden_size < 4:
            raise ValueError("MC-LSTM requires at least four mass cells")
        self.hidden_size = hidden_size
        self.state_size = hidden_size
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 96), nn.ReLU(), nn.Linear(96, static_size), nn.Tanh(),
        )
        gate_inputs = 1 + 4 + static_size + hidden_size
        self.input_gate = nn.Linear(gate_inputs, hidden_size)
        self.redistribution_gate = nn.Linear(gate_inputs, hidden_size * hidden_size)
        self.output_gate = nn.Linear(gate_inputs, hidden_size)
        nn.init.constant_(self.output_gate.bias, -2.0)

    def forward(self, forcing: torch.Tensor, static: torch.Tensor, initial_state=None, return_states: bool = False):
        batch = forcing.shape[0]
        cell = forcing.new_zeros((batch, self.hidden_size)) if initial_state is None else initial_state
        initial_total = cell.sum(dim=1)
        static_embedding = self.static_encoder(static)
        q_values, et_values, loss_values = [], [], []
        states, residuals = [], []
        for day in range(forcing.shape[1]):
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            auxiliary = torch.stack([
                torch.log1p(torch.clamp(forcing[:, day, 1], min=0.0)) / 2.0,
                forcing[:, day, 2] / 20.0,
                forcing[:, day, 3],
                forcing[:, day, 4],
            ], dim=1)
            normalized_cell = cell / (cell.sum(dim=1, keepdim=True) + 1e-5)
            gate_features = torch.cat([torch.log1p(p)[:, None] / 3.0, auxiliary, static_embedding, normalized_cell], dim=1)
            input_distribution = torch.softmax(self.input_gate(gate_features), dim=1)
            redistribution = self.redistribution_gate(gate_features).reshape(batch, self.hidden_size, self.hidden_size)
            redistribution = torch.softmax(redistribution, dim=2)
            output_fraction = torch.sigmoid(self.output_gate(gate_features))
            incoming = p[:, None] * input_distribution
            redistributed = torch.bmm(cell[:, None, :], redistribution).squeeze(1)
            available = incoming + redistributed
            outgoing = output_fraction * available
            new_cell = available - outgoing
            et = outgoing[:, 0]
            deep_loss = outgoing[:, 1]
            q = outgoing[:, 2:].sum(dim=1)
            if return_states:
                residuals.append(cell.sum(dim=1) + p - outgoing.sum(dim=1) - new_cell.sum(dim=1))
                states.append(new_cell)
            cell = new_cell
            q_values.append(q); et_values.append(et); loss_values.append(deep_loss)

        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        result = {"q": q, "et": et, "loss": loss, "final_state": cell}
        if return_states:
            result["state"] = torch.stack(states, dim=1)
            result["storage"] = result["state"].sum(dim=-1)
            result["daily_residual"] = torch.stack(residuals, dim=1)
            result["cumulative_residual"] = (
                initial_total + forcing[..., 0].sum(dim=1)
                - q.sum(dim=1) - et.sum(dim=1) - loss.sum(dim=1) - cell.sum(dim=1)
            )
        return result


class NeuralFluxGate(nn.Module):
    """Flexible neural flux gates with exact, nonnegative mass accounting."""

    architecture = "fluxgate"
    conserves_mass = True
    state_size = 5

    def __init__(self, n_static: int, static_hidden: int = 64, gate_hidden: int = 64, embedding: int = 24):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, static_hidden), nn.SiLU(),
            nn.Linear(static_hidden, static_hidden), nn.SiLU(),
            nn.Linear(static_hidden, embedding + 3),
        )
        self.gates = nn.Sequential(
            nn.Linear(9 + embedding, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, gate_hidden), nn.SiLU(),
            nn.Linear(gate_hidden, 8),
        )
        with torch.no_grad():
            self.static_encoder[-1].bias[-3:].copy_(torch.tensor([-0.8, -0.5, -2.0]))
            self.gates[-1].bias.copy_(torch.tensor([-2.0, 0.0, -3.0, -5.0, -1.5, -2.0, -3.0, -1.0]))

    def static_terms(self, static: torch.Tensor):
        encoded = self.static_encoder(static)
        embedding, raw = encoded[:, :-3], encoded[:, -3:]
        return embedding, {
            "capacity": 50.0 + 1450.0 * torch.sigmoid(raw[:, 0]),
            "t0": -2.5 + 5.0 * torch.sigmoid(raw[:, 1]),
            "degree_day": 0.25 + 7.75 * torch.sigmoid(raw[:, 2]),
        }

    def initial_state(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        zeros = torch.zeros_like(params["capacity"])
        return torch.stack([zeros, 0.45 * params["capacity"], zeros, 0.04 * params["capacity"], zeros], dim=1)

    def forward(self, forcing: torch.Tensor, static: torch.Tensor, initial_state=None, return_states: bool = False):
        embedding, params = self.static_terms(static)
        state = self.initial_state(params) if initial_state is None else initial_state
        snow, soil, fast, slow, route = state.unbind(dim=1)
        initial_total = state.sum(dim=1)
        q_values, et_values, loss_values = [], [], []
        states, residuals = [], []
        for day in range(forcing.shape[1]):
            before = snow + soil + fast + slow + route
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            temperature = forcing[:, day, 2]
            snow_fraction = torch.sigmoid((params["t0"] - temperature) * 2.0)
            snowfall = p * snow_fraction
            rain = p - snowfall
            available_snow = snow + snowfall
            melt = torch.minimum(available_snow, F.relu(temperature - params["t0"]) * params["degree_day"])
            snow = available_snow - melt
            liquid = rain + melt

            gate_input = torch.cat([
                (torch.log1p(p) / 4.0)[:, None],
                (torch.log1p(pet) / 2.0)[:, None],
                (temperature / 20.0)[:, None],
                forcing[:, day, 3:5],
                torch.clamp(soil / params["capacity"], 0.0, 2.0)[:, None],
                (torch.log1p(fast) / 5.0)[:, None],
                (torch.log1p(slow) / 5.0)[:, None],
                (torch.log1p(route) / 5.0)[:, None],
                embedding,
            ], dim=1)
            gate = self.gates(gate_input)
            saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
            direct_fraction = torch.sigmoid(gate[:, 0]) * torch.sqrt(saturation + 1e-6)
            direct = liquid * direct_fraction
            soil = soil + liquid - direct
            evap_saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
            evap = torch.minimum(
                soil,
                pet * (0.25 + 1.35 * torch.sigmoid(gate[:, 1])) * torch.sqrt(evap_saturation + 1e-6),
            )
            soil = soil - evap
            recharge = soil * (0.00005 + 0.08 * torch.sigmoid(gate[:, 2]))
            soil = soil - recharge
            deep_loss = soil * (0.000001 + 0.002 * torch.sigmoid(gate[:, 3]))
            soil = soil - deep_loss
            overflow = F.relu(soil - params["capacity"])
            soil = torch.minimum(soil, params["capacity"])
            fast_share = torch.sigmoid(gate[:, 4])
            fast = fast + direct + overflow + recharge * fast_share
            slow = slow + recharge * (1.0 - fast_share)
            q_fast = fast * (0.005 + 0.895 * torch.sigmoid(gate[:, 5]))
            q_slow = slow * (0.0001 + 0.12 * torch.sigmoid(gate[:, 6]))
            fast = fast - q_fast
            slow = slow - q_slow
            route = route + q_fast + q_slow
            q = route * (0.01 + 0.89 * torch.sigmoid(gate[:, 7]))
            route = route - q
            after = snow + soil + fast + slow + route
            q_values.append(q); et_values.append(evap); loss_values.append(deep_loss)
            if return_states:
                states.append(torch.stack([snow, soil, fast, slow, route], dim=1))
                residuals.append(before + p - q - evap - deep_loss - after)

        final_state = torch.stack([snow, soil, fast, slow, route], dim=1)
        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        result = {"q": q, "et": et, "loss": loss, "final_state": final_state}
        if return_states:
            result["state"] = torch.stack(states, dim=1)
            result["storage"] = result["state"].sum(dim=-1)
            result["daily_residual"] = torch.stack(residuals, dim=1)
            result["cumulative_residual"] = (
                initial_total + forcing[..., 0].sum(dim=1)
                - q.sum(dim=1) - et.sum(dim=1) - loss.sum(dim=1) - final_state.sum(dim=1)
            )
        return result


def make_model(architecture: str, n_static: int) -> nn.Module:
    if architecture == "ealstm":
        return EntityAwareLSTM(n_static=n_static)
    if architecture == "dhbv":
        return DifferentiableHBV(n_static=n_static)
    if architecture == "mclstm":
        return TypedMassConservingLSTM(n_static=n_static)
    if architecture == "fluxgate":
        return NeuralFluxGate(n_static=n_static)
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
