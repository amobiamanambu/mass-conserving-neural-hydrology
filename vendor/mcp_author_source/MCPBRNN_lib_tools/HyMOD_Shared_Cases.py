import torch
import torch.nn as nn


__all__ = [
    "MA3_PETconstraint_CellShared",
    "MA4_PETconstraint_CellShared",
    "MA5_PETconstraint_CellShared",
    "MA6_PETconstraint_CellShared",
]


_PET_MEAN = 2.9086
_PET_STD = 1.8980


def _prepare_input(x, batch_first):
    """Return input as [seq, time, features] for the legacy single-node layout."""
    if x.ndim != 3:
        raise ValueError(
            "Expected x with shape [time, seq, features] or [seq, time, features]."
        )

    if batch_first:
        x = x.transpose(0, 1)

    seq_len, n_steps, n_features = x.shape

    if seq_len != 1:
        raise ValueError("SALO cases require seq_length == 1.")

    if n_features < 2:
        raise ValueError("SALO cases require precipitation and PET inputs.")

    return x, n_steps


def _check_scale(name, value):
    if value == 0:
        raise ValueError(f"{name} must be non-zero.")


def _pet_constraint(loss_gate, pet, storage, relu):
    """Preserve the original PET constraint for the single-node cases."""
    if storage.item() > 0:
        return loss_gate - relu(loss_gate - pet / storage)
    return loss_gate


def _cat(series):
    return torch.cat(series, dim=0)


class MA3_PETconstraint_CellShared(nn.Module):
    """MA3-SALO: main storage coupled to one surface-routing storage."""

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA3-SALO is a single-node model and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Main storage.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Surface-routing storage.
        self.weight_r_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom_sw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize every trainable parameter; the legacy source missed six."""
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.uniform_(0.0, 1.0)

    def forward(self, x, cmean, cstd, cmean_sw, cstd_sw):
        _check_scale("cstd", cstd)
        _check_scale("cstd_sw", cstd_sw)

        x, n_steps = _prepare_input(x, self.batch_first)

        storage = x.new_zeros(1, 1)
        storage_sw = x.new_zeros(1, 1)

        discharge = []
        storage_series = []
        loss = []
        loss_constrained = []
        bypass = []

        gate_input = []
        gate_output = []
        gate_loss = []
        gate_loss_constrained = []
        gate_remember = []

        discharge_sw = []
        storage_sw_series = []
        gate_output_sw = []
        gate_remember_sw = []

        main_denominator = (
            torch.exp(self.weight_r_yom)
            + torch.exp(self.weight_r_ylm)
            + torch.exp(self.weight_r_yfm)
        )
        output_scale = torch.exp(self.weight_r_yom) / main_denominator
        loss_scale = torch.exp(self.weight_r_ylm) / main_denominator

        sw_denominator = (
            torch.exp(self.weight_r_yom_sw)
            + torch.exp(self.weight_r_yfm_sw)
        )
        output_scale_sw = torch.exp(self.weight_r_yom_sw) / sw_denominator

        for b in range(n_steps):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            main_state = (storage - cmean) / cstd
            sw_state = (storage_sw - cmean_sw) / cstd_sw

            output_gate = output_scale * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + torch.mm(main_state, self.weight_b1_yom)
                + torch.mm(sw_state, self.weight_b2_yom)
            )

            loss_gate = loss_scale * torch.sigmoid(
                self.bias_b0_ylm.unsqueeze(0)
                + torch.mm(
                    (pet - _PET_MEAN) / _PET_STD,
                    self.weight_b2_ylm,
                )
            )

            constrained_loss_gate = _pet_constraint(
                loss_gate,
                pet,
                storage,
                self.relu_l,
            )

            remember_gate = 1.0 - output_gate - constrained_loss_gate

            zero = torch.zeros_like(storage)

            # PRE-update diagnostics.
            main_discharge = output_gate * storage

            discharge.append(main_discharge)
            storage_series.append(storage)
            loss.append(loss_gate * storage)
            loss_constrained.append(constrained_loss_gate * storage)
            bypass.append(zero)

            gate_input.append(zero)
            gate_output.append(output_gate)
            gate_loss.append(loss_gate)
            gate_loss_constrained.append(constrained_loss_gate)
            gate_remember.append(remember_gate)

            # Main-storage update.
            storage_next = remember_gate * storage + precipitation

            # Surface routing uses PRE-update main discharge.
            output_gate_sw = output_scale_sw * torch.sigmoid(
                self.bias_b0_yom_sw.unsqueeze(0)
                + torch.mm(
                    (storage_sw - cmean_sw) / cstd_sw,
                    self.weight_b2_yom_sw,
                )
                + torch.mm(
                    (storage - cmean) / cstd,
                    self.weight_b1_yom_sw,
                )
            )
            remember_gate_sw = 1.0 - output_gate_sw

            discharge_sw.append(output_gate_sw * storage_sw)
            storage_sw_series.append(storage_sw)
            gate_output_sw.append(output_gate_sw)
            gate_remember_sw.append(remember_gate_sw)

            storage_sw = remember_gate_sw * storage_sw + main_discharge
            storage = storage_next

        return (
            _cat(discharge),
            _cat(storage_series),
            _cat(loss),
            _cat(loss_constrained),
            _cat(bypass),
            _cat(gate_input),
            _cat(gate_output),
            _cat(gate_loss),
            _cat(gate_loss_constrained),
            _cat(gate_remember),
            _cat(discharge_sw),
            _cat(storage_sw_series),
            _cat(gate_output_sw),
            _cat(gate_remember_sw),
        )


class MA4_PETconstraint_CellShared(nn.Module):
    """MA4-SALO: main storage coupled to one groundwater-routing storage."""

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA4-SALO is a single-node model and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Main storage: direct output, recharge, loss, remember.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yogw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Groundwater-routing storage.
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # Preserve the legacy reset order for reproducible fresh initialization.
            for parameter in (
                self.weight_r_yom,
                self.weight_r_yogw,
                self.weight_r_ylm,
                self.weight_r_yfm,
                self.bias_b0_yom,
                self.weight_b1_yom,
                self.weight_b2_yom,
                self.bias_b0_ylm,
                self.weight_b2_ylm,
                self.bias_b0_yogw,
                self.weight_b1_yogw,
                self.weight_b2_yogw,
                self.weight_r_yom_gw,
                self.weight_r_yfm_gw,
                self.bias_b0_yom_gw,
                self.weight_b1_yom_gw,
                self.weight_b2_yom_gw,
            ):
                parameter.uniform_(0.0, 1.0)

    def forward(self, x, cmean, cstd, cmean_gw, cstd_gw, Ini_C):
        _check_scale("cstd", cstd)
        _check_scale("cstd_gw", cstd_gw)

        x, n_steps = _prepare_input(x, self.batch_first)

        storage = x.new_zeros(1, 1)
        storage_gw = x.new_full((1, 1), float(Ini_C))

        discharge = []
        recharge = []
        storage_series = []
        loss = []
        loss_constrained = []
        bypass = []

        gate_input = []
        gate_output = []
        gate_recharge = []
        gate_loss = []
        gate_loss_constrained = []
        gate_remember = []

        discharge_gw = []
        storage_gw_series = []
        gate_output_gw = []
        gate_remember_gw = []

        main_denominator = (
            torch.exp(self.weight_r_yom)
            + torch.exp(self.weight_r_ylm)
            + torch.exp(self.weight_r_yfm)
            + torch.exp(self.weight_r_yogw)
        )
        output_scale = torch.exp(self.weight_r_yom) / main_denominator
        loss_scale = torch.exp(self.weight_r_ylm) / main_denominator
        recharge_scale = torch.exp(self.weight_r_yogw) / main_denominator

        gw_denominator = (
            torch.exp(self.weight_r_yom_gw)
            + torch.exp(self.weight_r_yfm_gw)
        )
        output_scale_gw = torch.exp(self.weight_r_yom_gw) / gw_denominator

        for b in range(n_steps):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            main_state = (storage - cmean) / cstd
            gw_state = (storage_gw - cmean_gw) / cstd_gw

            output_gate = output_scale * torch.sigmoid(
                self.bias_b0_yom.unsqueeze(0)
                + torch.mm(main_state, self.weight_b1_yom)
                + torch.mm(gw_state, self.weight_b2_yom)
            )

            loss_gate = loss_scale * torch.sigmoid(
                self.bias_b0_ylm.unsqueeze(0)
                + torch.mm(
                    (pet - _PET_MEAN) / _PET_STD,
                    self.weight_b2_ylm,
                )
            )

            recharge_gate = recharge_scale * torch.sigmoid(
                self.bias_b0_yogw.unsqueeze(0)
                + torch.mm(main_state, self.weight_b1_yogw)
                + torch.mm(gw_state, self.weight_b2_yogw)
            )

            constrained_loss_gate = _pet_constraint(
                loss_gate,
                pet,
                storage,
                self.relu_l,
            )

            remember_gate = (
                1.0
                - output_gate
                - recharge_gate
                - constrained_loss_gate
            )

            zero = torch.zeros_like(storage)

            main_discharge = output_gate * storage
            recharge_flux = recharge_gate * storage

            # PRE-update diagnostics.
            discharge.append(main_discharge)
            recharge.append(recharge_flux)
            storage_series.append(storage)
            loss.append(loss_gate * storage)
            loss_constrained.append(constrained_loss_gate * storage)
            bypass.append(zero)

            gate_input.append(zero)
            gate_output.append(output_gate)
            gate_recharge.append(recharge_gate)
            gate_loss.append(loss_gate)
            gate_loss_constrained.append(constrained_loss_gate)
            gate_remember.append(remember_gate)

            storage_next = remember_gate * storage + precipitation

            # Groundwater routing uses PRE-update recharge.
            output_gate_gw = output_scale_gw * torch.sigmoid(
                self.bias_b0_yom_gw.unsqueeze(0)
                + torch.mm(
                    (storage_gw - cmean_gw) / cstd_gw,
                    self.weight_b2_yom_gw,
                )
                + torch.mm(
                    (storage - cmean) / cstd,
                    self.weight_b1_yom_gw,
                )
            )
            remember_gate_gw = 1.0 - output_gate_gw

            discharge_gw.append(output_gate_gw * storage_gw)
            storage_gw_series.append(storage_gw)
            gate_output_gw.append(output_gate_gw)
            gate_remember_gw.append(remember_gate_gw)

            storage_gw = remember_gate_gw * storage_gw + recharge_flux
            storage = storage_next

        return (
            _cat(discharge),
            _cat(recharge),
            _cat(storage_series),
            _cat(loss),
            _cat(loss_constrained),
            _cat(bypass),
            _cat(gate_input),
            _cat(gate_output),
            _cat(gate_recharge),
            _cat(gate_loss),
            _cat(gate_loss_constrained),
            _cat(gate_remember),
            _cat(discharge_gw),
            _cat(storage_gw_series),
            _cat(gate_output_gw),
            _cat(gate_remember_gw),
        )


class MA5_PETconstraint_CellShared(nn.Module):
    """MA5-SALO: main storage coupled to surface and groundwater routing."""

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA5-SALO is a single-node model and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Main storage.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yogw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Surface-routing storage.
        self.weight_r_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom_sw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Groundwater-routing storage.
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for parameter in (
                self.weight_r_yom,
                self.weight_r_yogw,
                self.weight_r_ylm,
                self.weight_r_yfm,
                self.bias_b0_yom,
                self.weight_b1_yom,
                self.weight_b2_yom,
                self.weight_b3_yom,
                self.bias_b0_ylm,
                self.weight_b2_ylm,
                self.bias_b0_yogw,
                self.weight_b1_yogw,
                self.weight_b2_yogw,
                self.weight_b3_yogw,
                self.weight_r_yom_sw,
                self.weight_r_yfm_sw,
                self.bias_b0_yom_sw,
                self.weight_b1_yom_sw,
                self.weight_b2_yom_sw,
                self.weight_b3_yom_sw,
                self.weight_r_yom_gw,
                self.weight_r_yfm_gw,
                self.bias_b0_yom_gw,
                self.weight_b1_yom_gw,
                self.weight_b2_yom_gw,
                self.weight_b3_yom_gw,
            ):
                parameter.uniform_(0.0, 1.0)

    def forward(
        self,
        x,
        cmean,
        cstd,
        cmean_sw,
        cstd_sw,
        cmean_gw,
        cstd_gw,
        Ini_C,
    ):
        _check_scale("cstd", cstd)
        _check_scale("cstd_sw", cstd_sw)
        _check_scale("cstd_gw", cstd_gw)

        x, n_steps = _prepare_input(x, self.batch_first)

        storage = x.new_zeros(1, 1)
        storage_sw = x.new_zeros(1, 1)
        storage_gw = x.new_full((1, 1), float(Ini_C))

        discharge = []
        recharge = []
        storage_series = []
        loss = []
        loss_constrained = []
        bypass = []

        gate_input = []
        gate_output = []
        gate_recharge = []
        gate_loss = []
        gate_loss_constrained = []
        gate_remember = []

        discharge_sw = []
        storage_sw_series = []
        gate_output_sw = []
        gate_remember_sw = []

        discharge_gw = []
        storage_gw_series = []
        gate_output_gw = []
        gate_remember_gw = []

        main_denominator = (
            torch.exp(self.weight_r_yom)
            + torch.exp(self.weight_r_ylm)
            + torch.exp(self.weight_r_yfm)
            + torch.exp(self.weight_r_yogw)
        )
        output_scale = torch.exp(self.weight_r_yom) / main_denominator
        loss_scale = torch.exp(self.weight_r_ylm) / main_denominator
        recharge_scale = torch.exp(self.weight_r_yogw) / main_denominator

        sw_denominator = (
            torch.exp(self.weight_r_yom_sw)
            + torch.exp(self.weight_r_yfm_sw)
        )
        output_scale_sw = torch.exp(self.weight_r_yom_sw) / sw_denominator

        gw_denominator = (
            torch.exp(self.weight_r_yom_gw)
            + torch.exp(self.weight_r_yfm_gw)
        )
        output_scale_gw = torch.exp(self.weight_r_yom_gw) / gw_denominator

        for b in range(n_steps):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            main_state = (storage - cmean) / cstd
            sw_state = (storage_sw - cmean_sw) / cstd_sw
            gw_state = (storage_gw - cmean_gw) / cstd_gw

            output_state_1 = torch.addmm(
                self.bias_b0_yom.unsqueeze(0),
                main_state,
                self.weight_b1_yom,
            )
            output_state_2 = torch.mm(sw_state, self.weight_b2_yom)
            output_state_3 = torch.mm(gw_state, self.weight_b3_yom)
            output_gate = output_scale * torch.sigmoid(
                output_state_1 + output_state_2 + output_state_3
            )

            loss_gate = loss_scale * torch.sigmoid(
                self.bias_b0_ylm.unsqueeze(0)
                + torch.mm(
                    (pet - _PET_MEAN) / _PET_STD,
                    self.weight_b2_ylm,
                )
            )

            recharge_state_1 = torch.addmm(
                self.bias_b0_yogw.unsqueeze(0),
                main_state,
                self.weight_b1_yogw,
            )
            recharge_state_2 = torch.mm(sw_state, self.weight_b2_yogw)
            recharge_state_3 = torch.mm(gw_state, self.weight_b3_yogw)
            recharge_gate = recharge_scale * torch.sigmoid(
                recharge_state_1 + recharge_state_2 + recharge_state_3
            )

            constrained_loss_gate = _pet_constraint(
                loss_gate,
                pet,
                storage,
                self.relu_l,
            )

            remember_gate = (
                1.0
                - output_gate
                - recharge_gate
                - constrained_loss_gate
            )

            zero = torch.zeros_like(storage)

            main_discharge = output_gate * storage
            recharge_flux = recharge_gate * storage

            # PRE-update diagnostics.
            discharge.append(main_discharge)
            recharge.append(recharge_flux)
            storage_series.append(storage)
            loss.append(loss_gate * storage)
            loss_constrained.append(constrained_loss_gate * storage)
            bypass.append(zero)

            gate_input.append(zero)
            gate_output.append(output_gate)
            gate_recharge.append(recharge_gate)
            gate_loss.append(loss_gate)
            gate_loss_constrained.append(constrained_loss_gate)
            gate_remember.append(remember_gate)

            storage_next = remember_gate * storage + precipitation

            # Surface routing.
            sw_output_state_1 = torch.mm(main_state, self.weight_b1_yom_sw)
            sw_output_state_2 = torch.mm(sw_state, self.weight_b2_yom_sw)
            sw_output_state_3 = torch.addmm(
                self.bias_b0_yom_sw.unsqueeze(0),
                gw_state,
                self.weight_b3_yom_sw,
            )
            output_gate_sw = output_scale_sw * torch.sigmoid(
                sw_output_state_1 + sw_output_state_2 + sw_output_state_3
            )
            remember_gate_sw = 1.0 - output_gate_sw

            discharge_sw.append(output_gate_sw * storage_sw)
            storage_sw_series.append(storage_sw)
            gate_output_sw.append(output_gate_sw)
            gate_remember_sw.append(remember_gate_sw)

            # Groundwater routing.
            gw_output_state_1 = torch.mm(main_state, self.weight_b1_yom_gw)
            gw_output_state_2 = torch.mm(sw_state, self.weight_b2_yom_gw)
            gw_output_state_3 = torch.addmm(
                self.bias_b0_yom_gw.unsqueeze(0),
                gw_state,
                self.weight_b3_yom_gw,
            )
            output_gate_gw = output_scale_gw * torch.sigmoid(
                gw_output_state_1 + gw_output_state_2 + gw_output_state_3
            )
            remember_gate_gw = 1.0 - output_gate_gw

            discharge_gw.append(output_gate_gw * storage_gw)
            storage_gw_series.append(storage_gw)
            gate_output_gw.append(output_gate_gw)
            gate_remember_gw.append(remember_gate_gw)

            storage_sw = remember_gate_sw * storage_sw + main_discharge
            storage_gw = remember_gate_gw * storage_gw + recharge_flux
            storage = storage_next

        return (
            _cat(discharge),
            _cat(recharge),
            _cat(storage_series),
            _cat(loss),
            _cat(loss_constrained),
            _cat(bypass),
            _cat(gate_input),
            _cat(gate_output),
            _cat(gate_recharge),
            _cat(gate_loss),
            _cat(gate_loss_constrained),
            _cat(gate_remember),
            _cat(discharge_sw),
            _cat(storage_sw_series),
            _cat(gate_output_sw),
            _cat(gate_remember_sw),
            _cat(discharge_gw),
            _cat(storage_gw_series),
            _cat(gate_output_gw),
            _cat(gate_remember_gw),
        )


class MA6_PETconstraint_CellShared(nn.Module):
    """MA6-SALO: MA5-SALO plus a direct fast-path output from the main storage."""

    _LEGACY_UNUSED_KEY = "weight_r_yom_fp"

    def __init__(
        self,
        input_size: int,
        batch_first: bool = True,
        hidden_size: int = 1,
        initial_forget_bias: int = 0,
    ):
        super().__init__()

        del input_size, initial_forget_bias

        if hidden_size != 1:
            raise ValueError("MA6-SALO is a single-node model and requires hidden_size == 1.")

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Main storage.
        self.weight_r_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yofp = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yom = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yogw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yogw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_yofp = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yofp = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yofp = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yofp = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.bias_b0_ylm = nn.Parameter(torch.empty(hidden_size))
        self.weight_b2_ylm = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Surface-routing storage.
        self.weight_r_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom_sw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom_sw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Groundwater-routing storage.
        self.weight_r_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_r_yfm_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.empty(hidden_size))
        self.weight_b1_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b2_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_b3_yom_gw = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.relu_l = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for parameter in (
                self.weight_r_yom,
                self.weight_r_yogw,
                self.weight_r_yofp,
                self.weight_r_ylm,
                self.weight_r_yfm,
                self.bias_b0_yom,
                self.weight_b1_yom,
                self.weight_b2_yom,
                self.weight_b3_yom,
                self.bias_b0_ylm,
                self.weight_b2_ylm,
                self.bias_b0_yogw,
                self.weight_b1_yogw,
                self.weight_b2_yogw,
                self.weight_b3_yogw,
                self.bias_b0_yofp,
                self.weight_b1_yofp,
                self.weight_b2_yofp,
                self.weight_b3_yofp,
                self.weight_r_yom_sw,
                self.weight_r_yfm_sw,
                self.bias_b0_yom_sw,
                self.weight_b1_yom_sw,
                self.weight_b2_yom_sw,
                self.weight_b3_yom_sw,
                self.weight_r_yom_gw,
                self.weight_r_yfm_gw,
                self.bias_b0_yom_gw,
                self.weight_b1_yom_gw,
                self.weight_b2_yom_gw,
                self.weight_b3_yom_gw,
            ):
                parameter.uniform_(0.0, 1.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Old MA6 checkpoints contain one typo-created parameter that never
        # participates in forward(): weight_r_yom_fp. Drop only that dead key,
        # then retain strict checking for every scientifically active parameter.
        legacy_key = prefix + self._LEGACY_UNUSED_KEY
        state_dict.pop(legacy_key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x,
        cmean,
        cstd,
        cmean_sw,
        cstd_sw,
        cmean_gw,
        cstd_gw,
        Ini_C,
    ):
        _check_scale("cstd", cstd)
        _check_scale("cstd_sw", cstd_sw)
        _check_scale("cstd_gw", cstd_gw)

        x, n_steps = _prepare_input(x, self.batch_first)

        storage = x.new_zeros(1, 1)
        storage_sw = x.new_zeros(1, 1)
        storage_gw = x.new_full((1, 1), float(Ini_C))

        discharge = []
        recharge = []
        fast_path = []
        storage_series = []
        loss = []
        loss_constrained = []
        bypass = []

        gate_input = []
        gate_output = []
        gate_recharge = []
        gate_fast_path = []
        gate_loss = []
        gate_loss_constrained = []
        gate_remember = []

        discharge_sw = []
        storage_sw_series = []
        gate_output_sw = []
        gate_remember_sw = []

        discharge_gw = []
        storage_gw_series = []
        gate_output_gw = []
        gate_remember_gw = []

        main_denominator = (
            torch.exp(self.weight_r_yom)
            + torch.exp(self.weight_r_ylm)
            + torch.exp(self.weight_r_yfm)
            + torch.exp(self.weight_r_yogw)
            + torch.exp(self.weight_r_yofp)
        )
        output_scale = torch.exp(self.weight_r_yom) / main_denominator
        loss_scale = torch.exp(self.weight_r_ylm) / main_denominator
        recharge_scale = torch.exp(self.weight_r_yogw) / main_denominator
        fast_path_scale = torch.exp(self.weight_r_yofp) / main_denominator

        sw_denominator = (
            torch.exp(self.weight_r_yom_sw)
            + torch.exp(self.weight_r_yfm_sw)
        )
        output_scale_sw = torch.exp(self.weight_r_yom_sw) / sw_denominator

        gw_denominator = (
            torch.exp(self.weight_r_yom_gw)
            + torch.exp(self.weight_r_yfm_gw)
        )
        output_scale_gw = torch.exp(self.weight_r_yom_gw) / gw_denominator

        for b in range(n_steps):
            precipitation = x[0, b, 0].reshape(1, 1)
            pet = x[0, b, 1].reshape(1, 1)

            main_state = (storage - cmean) / cstd
            sw_state = (storage_sw - cmean_sw) / cstd_sw
            gw_state = (storage_gw - cmean_gw) / cstd_gw

            output_state_1 = torch.addmm(
                self.bias_b0_yom.unsqueeze(0),
                main_state,
                self.weight_b1_yom,
            )
            output_state_2 = torch.mm(sw_state, self.weight_b2_yom)
            output_state_3 = torch.mm(gw_state, self.weight_b3_yom)
            output_gate = output_scale * torch.sigmoid(
                output_state_1 + output_state_2 + output_state_3
            )

            loss_gate = loss_scale * torch.sigmoid(
                self.bias_b0_ylm.unsqueeze(0)
                + torch.mm(
                    (pet - _PET_MEAN) / _PET_STD,
                    self.weight_b2_ylm,
                )
            )

            recharge_state_1 = torch.addmm(
                self.bias_b0_yogw.unsqueeze(0),
                main_state,
                self.weight_b1_yogw,
            )
            recharge_state_2 = torch.mm(sw_state, self.weight_b2_yogw)
            recharge_state_3 = torch.mm(gw_state, self.weight_b3_yogw)
            recharge_gate = recharge_scale * torch.sigmoid(
                recharge_state_1 + recharge_state_2 + recharge_state_3
            )

            fast_path_state_1 = torch.addmm(
                self.bias_b0_yofp.unsqueeze(0),
                main_state,
                self.weight_b1_yofp,
            )
            fast_path_state_2 = torch.mm(sw_state, self.weight_b2_yofp)
            fast_path_state_3 = torch.mm(gw_state, self.weight_b3_yofp)
            fast_path_gate = fast_path_scale * torch.sigmoid(
                fast_path_state_1 + fast_path_state_2 + fast_path_state_3
            )

            constrained_loss_gate = _pet_constraint(
                loss_gate,
                pet,
                storage,
                self.relu_l,
            )

            remember_gate = (
                1.0
                - output_gate
                - recharge_gate
                - fast_path_gate
                - constrained_loss_gate
            )

            zero = torch.zeros_like(storage)

            main_discharge = output_gate * storage
            recharge_flux = recharge_gate * storage
            fast_path_flux = fast_path_gate * storage

            # PRE-update diagnostics.
            discharge.append(main_discharge)
            recharge.append(recharge_flux)
            fast_path.append(fast_path_flux)
            storage_series.append(storage)
            loss.append(loss_gate * storage)
            loss_constrained.append(constrained_loss_gate * storage)
            bypass.append(zero)

            gate_input.append(zero)
            gate_output.append(output_gate)
            gate_recharge.append(recharge_gate)
            gate_fast_path.append(fast_path_gate)
            gate_loss.append(loss_gate)
            gate_loss_constrained.append(constrained_loss_gate)
            gate_remember.append(remember_gate)

            storage_next = remember_gate * storage + precipitation

            # Surface routing.
            sw_output_state_1 = torch.mm(main_state, self.weight_b1_yom_sw)
            sw_output_state_2 = torch.mm(sw_state, self.weight_b2_yom_sw)
            sw_output_state_3 = torch.addmm(
                self.bias_b0_yom_sw.unsqueeze(0),
                gw_state,
                self.weight_b3_yom_sw,
            )
            output_gate_sw = output_scale_sw * torch.sigmoid(
                sw_output_state_1 + sw_output_state_2 + sw_output_state_3
            )
            remember_gate_sw = 1.0 - output_gate_sw

            discharge_sw.append(output_gate_sw * storage_sw)
            storage_sw_series.append(storage_sw)
            gate_output_sw.append(output_gate_sw)
            gate_remember_sw.append(remember_gate_sw)

            # Groundwater routing.
            gw_output_state_1 = torch.mm(main_state, self.weight_b1_yom_gw)
            gw_output_state_2 = torch.mm(sw_state, self.weight_b2_yom_gw)
            gw_output_state_3 = torch.addmm(
                self.bias_b0_yom_gw.unsqueeze(0),
                gw_state,
                self.weight_b3_yom_gw,
            )
            output_gate_gw = output_scale_gw * torch.sigmoid(
                gw_output_state_1 + gw_output_state_2 + gw_output_state_3
            )
            remember_gate_gw = 1.0 - output_gate_gw

            discharge_gw.append(output_gate_gw * storage_gw)
            storage_gw_series.append(storage_gw)
            gate_output_gw.append(output_gate_gw)
            gate_remember_gw.append(remember_gate_gw)

            storage_sw = remember_gate_sw * storage_sw + main_discharge
            storage_gw = remember_gate_gw * storage_gw + recharge_flux
            storage = storage_next

        return (
            _cat(discharge),
            _cat(recharge),
            _cat(fast_path),
            _cat(storage_series),
            _cat(loss),
            _cat(loss_constrained),
            _cat(bypass),
            _cat(gate_input),
            _cat(gate_output),
            _cat(gate_recharge),
            _cat(gate_fast_path),
            _cat(gate_loss),
            _cat(gate_loss_constrained),
            _cat(gate_remember),
            _cat(discharge_sw),
            _cat(storage_sw_series),
            _cat(gate_output_sw),
            _cat(gate_remember_sw),
            _cat(discharge_gw),
            _cat(storage_gw_series),
            _cat(gate_output_gw),
            _cat(gate_remember_gw),
        )
