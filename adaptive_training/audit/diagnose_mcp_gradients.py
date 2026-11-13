#!/usr/bin/env python3
"""Find where MCP-MA3 training produces non-finite gradients (no weights are changed).

For each checkpoint it runs real training batches (same data, loss, batch
size and clipping as training), computes gradients without an optimizer step,
and records how many batches give a non-finite gradient norm and which
parameters carry it. For the first such batch it reruns the backward pass in
PyTorch anomaly mode, which reports the forward operation (file and line) that
created the bad gradient, and it records extreme values of the model's
internal quantities on that batch.

Record only in this package. The script was run as
`adaptive_training/05_diagnose_mcp_gradients.py` on the failed MCP-MA3 replay
checkpoints, which are not shipped, so it cannot be rerun from here. Its output
is `adaptive_training/provenance/mcp_gradient_diagnostic.json`.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "vendor/neuralhydrology_literature_adapter"))
os.environ.setdefault("NGEN_SILENT", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from neuralhydrology.datasetzoo import get_dataset  # noqa: E402
from neuralhydrology.datautils.utils import load_scaler  # noqa: E402
from neuralhydrology.modelzoo import get_model  # noqa: E402
from neuralhydrology.training import get_loss_obj, get_regularization_obj  # noqa: E402
from neuralhydrology.utils.config import Config  # noqa: E402

MODEL = "author_mcp_ma3"
BATCHES = 64  # one pseudo-epoch
CASES = [  # (label, seed, checkpoint epoch); seed 11 is the clean control
    ("seed11_selected", 11, None),
    # Training logs: seed 29 skipped nearly every update from epoch 108, seed 47 from epoch 80.
    ("seed29_last_good", 29, 100),
    ("seed29_stuck", 29, 110),
    ("seed47_last_good", 47, 70),
    ("seed47_stuck", 47, 90),
]


def checkpoint_path(seed: int, epoch: int | None) -> tuple[Path, Path, int]:
    base = HERE / "runs_adaptive" / MODEL / f"seed{seed}" / "base_run"
    if epoch is None:
        epoch = int(json.loads((HERE / f"adaptive_receipt_{MODEL}.json").read_text())["selected_epoch"])
    found = list(base.glob(f"**/model_epoch{epoch:03d}.pt"))
    if not found:
        raise FileNotFoundError(f"No epoch-{epoch} checkpoint for seed {seed} under {base}")
    return base, found[0], epoch


def batch_to(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[key] = {k: v for k, v in value.items()}
        else:
            out[key] = value
    return out


def internal_extremes(model, data) -> dict:
    """Extreme values of the regional MCP-MA3 parameters and states on one batch."""
    with torch.no_grad():
        gates, c_mean, c_std, route_mean, route_std = model.parameters_from_static(data["x_s"])
        result = model(data)
        storage = result["state"][..., 0]
        route = result["state"][..., 1]
        return {
            "raw_gate_abs_max": float(gates.abs().max()),
            "c_mean_min": float(c_mean.min()), "c_std_min": float(c_std.min()),
            "route_mean_min": float(route_mean.min()), "route_std_min": float(route_std.min()),
            "main_storage_min_mm": float(storage.min()), "main_storage_max_mm": float(storage.max()),
            "fraction_main_storage_below_1e-6_mm": float((storage < 1e-6).float().mean()),
            "routing_storage_min_mm": float(route.min()),
            "q_max_mm_day": float(result["q"].max()), "q_min_mm_day": float(result["q"].min()),
        }


def diagnose(label: str, seed: int, epoch: int | None) -> dict:
    base, checkpoint, epoch = checkpoint_path(seed, epoch)
    cfg = Config(base / "config.yml")
    cfg.update_config({"device": "cpu"})
    torch.manual_seed(1234)
    model = get_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.train()
    dataset = get_dataset(cfg=cfg, period="train", is_train=True, scaler=load_scaler(base))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    loss_obj = get_loss_obj(cfg)
    loss_obj.set_regularization_terms(get_regularization_obj(cfg))
    bad, nan_loss, norms, first_bad, bad_parameters = 0, 0, [], None, {}
    for index, data in enumerate(loader):
        if index >= BATCHES:
            break
        data = model.pre_model_hook(batch_to(data), is_train=True)  # same pre-processing as the trainer
        model.zero_grad()
        prediction = model(data)
        loss, _ = loss_obj(prediction, data)
        if not torch.isfinite(loss):
            nan_loss += 1
            continue
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        # float32 norm, exactly what clip_grad_norm_ computes during training
        total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        total64 = torch.sqrt(sum((g.double() ** 2).sum() for g in grads))
        largest = max(float(g.abs().max()) for g in grads)
        norms.append(float(total))
        if first_bad is None and not torch.isfinite(total):
            overflow_only = bool(torch.isfinite(total64)) and all(torch.isfinite(g).all() for g in grads)
        if not torch.isfinite(total):
            bad += 1
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    bad_parameters[name] = bad_parameters.get(name, 0) + 1
            if first_bad is None:
                first_bad = {"batch_index": index, "loss": float(loss), "internal": internal_extremes(model, data),
                             "float64_gradient_norm": float(total64), "largest_abs_gradient": largest,
                             "gradients_finite_but_float32_norm_overflows": overflow_only}
                model.zero_grad()
                with warnings.catch_warnings(record=True) as caught, torch.autograd.detect_anomaly(check_nan=True):
                    warnings.simplefilter("always")
                    try:
                        loss2, _ = loss_obj(model(data), data)
                        loss2.backward()
                        first_bad["anomaly_error"] = None
                    except RuntimeError as error:
                        first_bad["anomaly_error"] = str(error).splitlines()[0]
                    first_bad["anomaly_forward_trace"] = [str(w.message)[-3000:] for w in caught][-2:]
    finite = [n for n in norms if np.isfinite(n)]
    return {
        "label": label, "seed": seed, "checkpoint_epoch": epoch, "batches_checked": min(BATCHES, len(loader)),
        "nan_loss_batches": nan_loss, "nonfinite_gradient_batches": bad,
        "median_finite_gradient_norm": float(np.median(finite)) if finite else None,
        "parameters_with_nonfinite_gradients": bad_parameters,
        "first_nonfinite_batch": first_bad,
    }


def main() -> None:
    report = {"model": MODEL, "note": "No weights were updated; gradients only.", "cases": []}
    for label, seed, epoch in CASES:
        try:
            result = diagnose(label, seed, epoch)
        except Exception as error:  # keep going so one missing checkpoint does not hide the others
            result = {"label": label, "error": repr(error), "traceback": traceback.format_exc()[-2000:]}
        report["cases"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "first_nonfinite_batch"}, indent=2), flush=True)
    (HERE / "mcp_gradient_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote mcp_gradient_diagnostic.json")


if __name__ == "__main__":
    main()
