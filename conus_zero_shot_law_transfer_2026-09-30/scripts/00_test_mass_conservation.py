#!/usr/bin/env python3
"""Small deterministic numerical-closure test for both constrained models."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydro_models import make_model  # noqa: E402


def main() -> None:
    torch.manual_seed(20260930)
    forcing = torch.rand(5, 120, 5, dtype=torch.float64)
    forcing[..., 2] = forcing[..., 2] * 30.0 - 10.0
    forcing[..., 3:] = forcing[..., 3:] * 2.0 - 1.0
    static = torch.randn(5, 99, dtype=torch.float64)
    for architecture in ["dhbv", "mclstm", "fluxgate"]:
        model = make_model(architecture, 99).double()
        result = model(forcing, static, return_states=True)
        daily = float(result["daily_residual"].detach().abs().max())
        cumulative = float(result["cumulative_residual"].detach().abs().max())
        if daily > 1e-10 or cumulative > 1e-9:
            raise RuntimeError(f"{architecture} failed closure: daily={daily}, cumulative={cumulative}")
        print(f"{architecture}: daily={daily:.3e}, cumulative={cumulative:.3e}")


if __name__ == "__main__":
    main()
