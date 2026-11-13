"""Equation-preserving NeuralHydrology ports of the parent study models.

Only the interface is adapted from ``forward(forcing, static)`` to the
NeuralHydrology batch dictionary. Hydrologic equations, parameter bounds,
initial states, and network widths are unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


INPUTS = ("pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy")


def _forcing(data: dict) -> torch.Tensor:
    return torch.cat([data["x_d"][name] for name in INPUTS], dim=-1)


class PortedDifferentiableHBV(BaseModel):
    """Static-parameter regional HBV-like model with exact accounting."""

    module_parts = ["parameter_net"]
    conserves_mass = True
    state_size = 5

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n_static = len(cfg.static_attributes)
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

    def forward(self, data: dict) -> dict[str, torch.Tensor]:
        forcing = _forcing(data)
        static = data["x_s"]
        params = self.parameters_from_static(static)
        batch = forcing.shape[0]
        if data.get("initial_state") is None:
            snow = forcing.new_zeros(batch)
            soil = forcing.new_zeros(batch)
            fast = forcing.new_zeros(batch)
            slow = forcing.new_zeros(batch)
            route = forcing.new_zeros(batch)
        else:
            snow, soil, fast, slow, route = data["initial_state"].unbind(dim=1)
        initial_total = snow + soil + fast + slow + route
        q_values, et_values, loss_values, states, residuals = [], [], [], [], []
        for day in range(forcing.shape[1]):
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            temperature = forcing[:, day, 2]
            before = snow + soil + fast + slow + route
            snow_fraction = torch.sigmoid(-2.0 * (temperature - params["t0"]))
            snowfall = p * snow_fraction
            rainfall = p - snowfall
            melt = torch.minimum(
                snow + snowfall,
                params["degree_day"] * F.relu(temperature - params["t0"]),
            )
            snow = snow + snowfall - melt
            liquid = rainfall + melt
            saturation = torch.clamp(soil / params["capacity"], 0.0, 1.0)
            recharge = liquid * torch.pow(saturation + 1e-6, params["beta"])
            soil = soil + liquid - recharge
            overflow = F.relu(soil - params["capacity"])
            soil = torch.minimum(soil, params["capacity"])
            evap_fraction = torch.pow(
                torch.clamp(soil / params["capacity"], 0.0, 1.0) + 1e-6,
                params["evap_shape"],
            )
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
            state = torch.stack([snow, soil, fast, slow, route], dim=1)
            after = state.sum(dim=1)
            q_values.append(q)
            et_values.append(evap)
            loss_values.append(deep_loss)
            states.append(state)
            residuals.append(before + p - q - evap - deep_loss - after)
        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        state = torch.stack(states, dim=1)
        final_state = state[:, -1]
        return {
            "y_hat": q.unsqueeze(-1),
            "q": q,
            "et": et,
            "loss": loss,
            "state": state,
            "storage": state.sum(dim=-1),
            "final_state": final_state,
            "daily_residual": torch.stack(residuals, dim=1),
            "cumulative_residual": (
                initial_total + forcing[..., 0].sum(dim=1) - q.sum(dim=1)
                - et.sum(dim=1) - loss.sum(dim=1) - final_state.sum(dim=1)
            ),
        }


class PortedNeuralFluxGate(BaseModel):
    """Five-store neural flux-gate model with exact, nonnegative accounting."""

    module_parts = ["static_encoder", "gates"]
    conserves_mass = True
    state_size = 5

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n_static = len(cfg.static_attributes)
        self.static_encoder = nn.Sequential(
            nn.Linear(n_static, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 27),
        )
        self.gates = nn.Sequential(
            nn.Linear(33, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 8),
        )
        with torch.no_grad():
            self.static_encoder[-1].bias[-3:].copy_(torch.tensor([-0.8, -0.5, -2.0]))
            self.gates[-1].bias.copy_(
                torch.tensor([-2.0, 0.0, -3.0, -5.0, -1.5, -2.0, -3.0, -1.0])
            )

    def static_terms(self, static: torch.Tensor):
        encoded = self.static_encoder(static)
        embedding, raw = encoded[:, :-3], encoded[:, -3:]
        return embedding, {
            "capacity": 50.0 + 1450.0 * torch.sigmoid(raw[:, 0]),
            "t0": -2.5 + 5.0 * torch.sigmoid(raw[:, 1]),
            "degree_day": 0.25 + 7.75 * torch.sigmoid(raw[:, 2]),
        }

    def forward(self, data: dict) -> dict[str, torch.Tensor]:
        forcing = _forcing(data)
        embedding, params = self.static_terms(data["x_s"])
        zeros = torch.zeros_like(params["capacity"])
        state = data.get("initial_state")
        if state is None:
            state = torch.stack(
                [zeros, 0.45 * params["capacity"], zeros, 0.04 * params["capacity"], zeros], dim=1
            )
        snow, soil, fast, slow, route = state.unbind(dim=1)
        initial_total = state.sum(dim=1)
        q_values, et_values, loss_values, states, residuals = [], [], [], [], []
        for day in range(forcing.shape[1]):
            before = snow + soil + fast + slow + route
            p = torch.clamp(forcing[:, day, 0], min=0.0)
            pet = torch.clamp(forcing[:, day, 1], min=0.0)
            temperature = forcing[:, day, 2]
            snow_fraction = torch.sigmoid((params["t0"] - temperature) * 2.0)
            snowfall = p * snow_fraction
            rain = p - snowfall
            available_snow = snow + snowfall
            melt = torch.minimum(
                available_snow, F.relu(temperature - params["t0"]) * params["degree_day"]
            )
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
                pet * (0.25 + 1.35 * torch.sigmoid(gate[:, 1]))
                * torch.sqrt(evap_saturation + 1e-6),
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
            state = torch.stack([snow, soil, fast, slow, route], dim=1)
            after = state.sum(dim=1)
            q_values.append(q)
            et_values.append(evap)
            loss_values.append(deep_loss)
            states.append(state)
            residuals.append(before + p - q - evap - deep_loss - after)
        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        loss = torch.stack(loss_values, dim=1)
        state = torch.stack(states, dim=1)
        final_state = state[:, -1]
        return {
            "y_hat": q.unsqueeze(-1),
            "q": q,
            "et": et,
            "loss": loss,
            "state": state,
            "storage": state.sum(dim=-1),
            "final_state": final_state,
            "daily_residual": torch.stack(residuals, dim=1),
            "cumulative_residual": (
                initial_total + forcing[..., 0].sum(dim=1) - q.sum(dim=1)
                - et.sum(dim=1) - loss.sum(dim=1) - final_state.sum(dim=1)
            ),
        }
