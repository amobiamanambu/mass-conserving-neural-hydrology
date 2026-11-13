#!/usr/bin/env python3
"""Seeds 29 and 47: replay the seed-11 learning-rate schedule to the selected epoch.

No development evaluation and no second selection (PROTOCOL.md, step 7).
Usage: python 02_replay_other_seeds.py --model hybrid_shm --seed 29
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ADAPTER = PROJECT / "vendor" / "neuralhydrology_literature_adapter"
MODELS = ("hybrid_shm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")
FLOAT64_CLIPPING_MODELS = {"author_mcp_ma3"}
# A replay is valid only if at most 1 % of its optimizer updates were skipped.
MAX_SKIPPED_FRACTION = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=(29, 47), required=True)
    args = parser.parse_args()
    receipt = json.loads((HERE / f"adaptive_receipt_{args.model}.json").read_text())
    if receipt["status"] == "running":
        raise RuntimeError(f"Seed-11 training for {args.model} has not finished")
    selected = int(receipt["selected_epoch"])
    base = HERE / "runs_adaptive" / args.model / f"seed{args.seed}" / "base_run"
    destination = base / "continue_training_from_epoch030"
    checkpoint = destination / f"model_epoch{selected:03d}.pt"
    if selected == 30 or checkpoint.exists():
        print(f"SKIP {args.model} seed {args.seed}: epoch {selected} already available")
        return
    if destination.exists():
        raise RuntimeError(f"Incomplete earlier replay found; delete it and rerun: {destination}")
    schedule = {int(k): float(v) for k, v in receipt["learning_rate_schedule"].items() if int(k) <= selected}
    override = HERE / "overrides" / f"{args.model}_seed{args.seed}_replay_to_epoch{selected:03d}.yml"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(yaml.safe_dump({
        "epochs": selected - 30,
        "learning_rate": {0: schedule[31], **schedule},
        "validate_every": None, "save_weights_every": 10, "save_validation_results": False, "verbose": 0,
        # NeuralHydrology's own safeguard: a batch whose loss is NaN is skipped
        # (no weight update) instead of ending the run; five NaN batches in a
        # row still stop training. No effect when no NaN occurs. Added after
        # MCP-MA3 seeds 29 and 47 hit a single NaN batch at the peak rate.
        "allow_subsequent_nan_losses": 5,
        # Project safeguard (vendored trainer, default off): skip an update whose
        # gradient norm is not finite instead of letting clipping write NaN weights.
        "skip_nonfinite_gradient_steps": True,
        # Amendment 2026-11-11 (MCP-MA3 only): the float32 gradient norm overflowed
        # for very large but finite gradients (05_diagnose_mcp_gradients.py), so the
        # clipping norm is computed in float64. Identical to standard clipping
        # whenever the float32 norm is finite.
        **({"float64_gradient_clipping": True} if args.model in FLOAT64_CLIPPING_MODELS else {}),
    }, sort_keys=False))
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ADAPTER), "MPLCONFIGDIR": "/private/tmp/nh_mpl_cache", "NGEN_SILENT": "1",
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"), "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "2")})
    subprocess.run([sys.executable, "-m", "neuralhydrology.nh_run", "continue_training", "--run-dir", str(base),
                    "--config-file", str(override)], cwd=PROJECT, env=env, check=True, stdin=subprocess.DEVNULL)
    if not checkpoint.exists():
        raise RuntimeError(f"Replay ended without the selected checkpoint: {checkpoint}")
    log = destination / "output.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    skipped = text.count("Loss is Nan; ignoring step")
    skipped_gradients = text.count("Non-finite gradient norm; skipping update")
    (HERE / f"replay_receipt_{args.model}_seed{args.seed}.json").write_text(json.dumps({
        "model": args.model, "seed": args.seed, "selected_epoch": selected,
        "learning_rate_schedule": {str(k): v for k, v in schedule.items()},
        "skipped_nan_loss_batches": skipped, "skipped_nonfinite_gradient_updates": skipped_gradients,
        "updates": (selected - 30) * 64,
        "float64_gradient_clipping": args.model in FLOAT64_CLIPPING_MODELS,
        "skipped_fraction": (skipped + skipped_gradients) / ((selected - 30) * 64),
        "valid": (skipped + skipped_gradients) <= MAX_SKIPPED_FRACTION * (selected - 30) * 64,
    }, indent=2) + "\n")
    print(checkpoint, f"skipped NaN-loss batches: {skipped}; skipped non-finite-gradient updates: {skipped_gradients}")


if __name__ == "__main__":
    main()
