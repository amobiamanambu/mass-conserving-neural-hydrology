"""Literature-source mass-conserving models exposed to NeuralHydrology.

The dHBV adapter calls the authors' HydroDL2 HBV implementation and the
authors' generic-deltaModel LSTM parameterizer directly.  The MCP adapter
vectorizes the released single-basin MA3 equations for a regional batch; a
separate parity test verifies its algebra against the authors' class.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.author_dmg_lstm import LstmModel
from neuralhydrology.utils.config import Config


def _cat(data: dict, names: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([data["x_d"][name] for name in names], dim=-1)


class AuthorDeltaHBV(BaseModel):
    """HydroDL2 HBV 1.0 driven by the authors' LSTM parameterizer.

    NeuralHydrology supplies normalized forcings and attributes to the LSTM.
    Unnormalized feature copies are passed to the unchanged HBV process core.
    """

    module_parts = ["parameterizer", "hbv"]
    conserves_mass = True
    state_size = 5

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Import HydroDL2 only when this architecture is instantiated.  Its
        # package-level noncommercial-license prompt must not block the wholly
        # independent MCP-MA3 adapter.
        from hydrodl2.models.hbv.hbv import Hbv

        hbv_config = {
            "warmup": 0,
            "warmup_states": False,
            "dy_drop": 0.0,
            "dynamic_params": {"Hbv": ["parBETA", "parBETAET"]},
            "variables": ["prcp", "tmean", "pet"],
            "routing": True,
            "comprout": False,
            # Retain HydroDL2's documented numerical floor. Setting this to
            # zero makes learned power-law gradients undefined when soil
            # storage is exhausted (0 * log(0)). The source default is
            # 1e-5 mm and is part of the stable author implementation.
            "nearzero": 1e-5,
            "nmul": 1,
            "cache_states": False,
        }
        self.hbv = Hbv(hbv_config, device=torch.device("cpu"))
        self.parameterizer = LstmModel(
            nx=3 + len(cfg.static_attributes),
            ny=self.hbv.learnable_param_count,
            hidden_size=64,
            dr=0.20,
            dpl=False,
            cache_states=False,
        )

    @staticmethod
    def _raw_forcing(data: dict) -> torch.Tensor:
        # HydroDL2 requires [time, basin, (precipitation, temperature, PET)].
        return _cat(
            data,
            ("pr_mm_copy1", "tmean_c_copy1", "pet_mm_copy1"),
        ).transpose(0, 1)

    @staticmethod
    def _parameter_inputs(data: dict) -> torch.Tensor:
        normalized = _cat(data, ("pr_mm", "tmean_c", "pet_mm"))
        static = data["x_s"][:, None, :].expand(-1, normalized.shape[1], -1)
        return torch.cat([normalized, static], dim=-1).transpose(0, 1)

    def forward(self, data: dict) -> dict[str, torch.Tensor]:
        raw = self._raw_forcing(data)
        parameters = self.parameterizer(self._parameter_inputs(data))
        # HydroDL2 stores its target device as an attribute in addition to the
        # normal torch module device.
        self.hbv.device = raw.device
        fluxes = self.hbv({"x_phy": raw}, parameters)
        final_states = self.hbv._state_cache
        if final_states is None:
            raise RuntimeError("HydroDL2 did not expose its final HBV states")

        q = fluxes["streamflow"].transpose(0, 1)
        q_unrouted = fluxes["streamflow_no_rout"].mean(dim=2).transpose(0, 1)
        et = fluxes["AET_hydro"].transpose(0, 1).squeeze(-1)
        final_state = torch.stack(
            [state.mean(dim=-1) for state in final_states], dim=-1
        )
        zero_loss = torch.zeros_like(et)
        return {
            "y_hat": q,
            "q": q.squeeze(-1),
            "q_unrouted": q_unrouted,
            "et": et,
            "loss": zero_loss,
            "final_state": final_state,
            "parameters": parameters,
        }


class AuthorMCPMA3Regional(BaseModel):
    """Regionalized form of the published two-store MCP-MA3 equations.

    The released MA3 model is single-basin.  A static-attribute network emits
    its 13 raw gate parameters and four storage-normalization constants for
    each catchment.  The two storage updates, PET constraint, gate functions,
    and order of operations are unchanged from the author source.
    """

    module_parts = ["parameter_net"]
    conserves_mass = True
    state_size = 2
    parameter_count_ma3 = 13

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n_static = len(cfg.static_attributes)
        self.parameter_net = nn.Sequential(
            nn.Linear(n_static, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.parameter_count_ma3 + 4),
        )
        with torch.no_grad():
            # Raw gate values near the center of the authors' initialization;
            # storage scales start in hydrologically plausible ranges.
            self.parameter_net[-1].bias[: self.parameter_count_ma3].fill_(0.5)
            self.parameter_net[-1].bias[self.parameter_count_ma3 :].copy_(
                torch.tensor([0.0, -0.4, -1.2, -1.0])
            )

    def parameters_from_static(
        self, static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.parameter_net(static)
        gates = raw[:, : self.parameter_count_ma3]
        scale_raw = raw[:, self.parameter_count_ma3 :]
        c_mean = 1000.0 * torch.sigmoid(scale_raw[:, 0])
        c_std = 1.0 + 499.0 * torch.sigmoid(scale_raw[:, 1])
        route_mean = 200.0 * torch.sigmoid(scale_raw[:, 2])
        route_std = 0.25 + 99.75 * torch.sigmoid(scale_raw[:, 3])
        return gates, c_mean, c_std, route_mean, route_std

    @staticmethod
    def _core(
        precipitation: torch.Tensor,
        pet: torch.Tensor,
        gates: torch.Tensor,
        c_mean: torch.Tensor,
        c_std: torch.Tensor,
        route_mean: torch.Tensor,
        route_std: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, days = precipitation.shape
        if initial_state is None:
            storage = precipitation.new_zeros(batch)
            route = precipitation.new_zeros(batch)
        else:
            storage, route = initial_state.unbind(dim=1)

        (
            weight_r_yom,
            weight_r_ylm,
            weight_r_yfm,
            bias_b0_yom,
            weight_b1_yom,
            weight_b2_yom,
            bias_b0_ylm,
            weight_b2_ylm,
            weight_r_yom_sw,
            weight_r_yfm_sw,
            bias_b0_yom_sw,
            weight_b1_yom_sw,
            weight_b2_yom_sw,
        ) = gates.unbind(dim=1)

        # This is algebraically identical to the authors' exp/sum ratios but
        # subtracts the row maximum internally, preventing overflow when a
        # regional training seed explores large raw gate weights.
        main_partition = torch.softmax(
            torch.stack([weight_r_yom, weight_r_ylm, weight_r_yfm], dim=1), dim=1
        )
        output_scale = main_partition[:, 0]
        loss_scale = main_partition[:, 1]
        route_partition = torch.softmax(
            torch.stack([weight_r_yom_sw, weight_r_yfm_sw], dim=1), dim=1
        )
        output_scale_route = route_partition[:, 0]

        initial_total = storage + route
        q_values: list[torch.Tensor] = []
        et_values: list[torch.Tensor] = []
        state_values: list[torch.Tensor] = []
        pre_state_values: list[torch.Tensor] = []
        residual_values: list[torch.Tensor] = []
        main_q_values: list[torch.Tensor] = []

        for day in range(days):
            p = torch.clamp(precipitation[:, day], min=0.0)
            day_pet = torch.clamp(pet[:, day], min=0.0)
            before = storage + route
            pre_state_values.append(torch.stack([storage, route], dim=1))

            main_state = (storage - c_mean) / c_std
            route_state = (route - route_mean) / route_std
            output_gate = output_scale * torch.sigmoid(
                bias_b0_yom
                + main_state * weight_b1_yom
                + route_state * weight_b2_yom
            )
            loss_gate = loss_scale * torch.sigmoid(
                bias_b0_ylm
                + ((day_pet - 2.9086) / 1.8980) * weight_b2_ylm
            )
            pet_limited = torch.where(
                storage > 0.0,
                torch.minimum(loss_gate, day_pet / torch.clamp(storage, min=1e-12)),
                loss_gate,
            )
            remember_gate = 1.0 - output_gate - pet_limited
            main_q = output_gate * storage
            actual_et = pet_limited * storage
            storage_next = remember_gate * storage + p

            route_output_gate = output_scale_route * torch.sigmoid(
                bias_b0_yom_sw
                + route_state * weight_b2_yom_sw
                + main_state * weight_b1_yom_sw
            )
            route_q = route_output_gate * route
            route = (1.0 - route_output_gate) * route + main_q
            storage = storage_next

            state = torch.stack([storage, route], dim=1)
            after = state.sum(dim=1)
            q_values.append(route_q)
            main_q_values.append(main_q)
            et_values.append(actual_et)
            state_values.append(state)
            residual_values.append(before + p - route_q - actual_et - after)

        q = torch.stack(q_values, dim=1)
        et = torch.stack(et_values, dim=1)
        state = torch.stack(state_values, dim=1)
        final_state = state[:, -1]
        residual = torch.stack(residual_values, dim=1)
        return {
            "q": q,
            "main_q": torch.stack(main_q_values, dim=1),
            "et": et,
            "loss": torch.zeros_like(et),
            "state": state,
            "pre_state": torch.stack(pre_state_values, dim=1),
            "storage": state.sum(dim=-1),
            "final_state": final_state,
            "daily_residual": residual,
            "cumulative_residual": (
                initial_total
                + precipitation.sum(dim=1)
                - q.sum(dim=1)
                - et.sum(dim=1)
                - final_state.sum(dim=1)
            ),
        }

    def forward(self, data: dict) -> dict[str, torch.Tensor]:
        precipitation = data["x_d"]["pr_mm_copy1"].squeeze(-1)
        pet = data["x_d"]["pet_mm_copy1"].squeeze(-1)
        parameters = self.parameters_from_static(data["x_s"])
        result = self._core(
            precipitation,
            pet,
            *parameters,
            initial_state=data.get("initial_state"),
        )
        result["y_hat"] = result["q"].unsqueeze(-1)
        result["regional_parameters"] = parameters[0]
        return result
