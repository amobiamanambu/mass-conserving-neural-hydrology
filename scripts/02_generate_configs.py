#!/usr/bin/env python3
"""Generate locked national NeuralHydrology configs for all five lineages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generic_blind"
LISTS = ROOT / "data" / "basin_lists"
DEST = ROOT / "configs"
SEEDS = (11, 29, 47)
MODELS = ("hybrid_shm", "mclstm", "author_dhbv", "author_mcp_ma3", "ported_fluxgate")


def base(model: str, seed: int, features: list[str]) -> dict:
    cfg = {
        "experiment_name": f"national_literature_{model}_seed{seed}",
        "run_dir": str((ROOT / "runs").resolve()),
        "train_basin_file": str((LISTS / "train_500.txt").resolve()),
        "validation_basin_file": str((LISTS / "validation_200.txt").resolve()),
        "test_basin_file": str((LISTS / "test_3982.txt").resolve()),
        "train_start_date": "05/10/2003",
        "train_end_date": "31/12/2014",
        "validation_start_date": "01/01/2015",
        "validation_end_date": "31/12/2018",
        "test_start_date": "01/01/2019",
        "test_end_date": "31/12/2024",
        "seed": seed,
        "device": "cpu",
        "validate_every": 10 if seed == 11 else None,
        "validate_n_random_basins": 200,
        "cache_validation_data": True,
        "metrics": ["NSE", "KGE", "Alpha-NSE", "Beta-NSE"],
        "head": "regression",
        "output_activation": "linear",
        "optimizer": "Adam",
        "loss": "NSE",
        "batch_size": 256,
        "epochs": 30,
        "max_updates_per_epoch": 64,
        "clip_gradient_norm": 5,
        "predict_last_n": 1,
        "seq_length": 365,
        "num_workers": 0,
        "log_interval": 20,
        "log_tensorboard": False,
        "save_weights_every": 10,
        "save_validation_results": True,
        "save_all_output": False,
        "verbose": 1,
        "dataset": "generic",
        "data_dir": str(DATA.resolve()),
        "target_variables": ["q_mm_day"],
        "static_attributes": features,
        "use_basin_id_encoding": False,
    }
    if model == "hybrid_shm":
        cfg.update({
            "model": "hybrid_model",
            "conceptual_model": "SHM",
            "hidden_size": 64,
            "initial_forget_bias": 3,
            "output_dropout": 0.0,
            "warmup_period": 0,
            "dynamic_inputs": ["pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy"],
            "duplicate_features": {"pr_mm": 1, "pet_mm": 1, "tmean_c": 2},
            "dynamic_conceptual_inputs": [
                "pr_mm_copy1", "pet_mm_copy1", "tmean_c_copy1", "tmean_c_copy2"
            ],
            "custom_normalization": {
                "pr_mm_copy1": {"centering": None, "scaling": None},
                "pet_mm_copy1": {"centering": None, "scaling": None},
                "tmean_c_copy1": {"centering": None, "scaling": None},
                "tmean_c_copy2": {"centering": None, "scaling": None},
                "q_mm_day": {"centering": None, "scaling": None},
            },
            "learning_rate": {0: 0.001, 20: 0.0005, 25: 0.0001},
            "clip_gradient_norm": 1,
        })
    elif model == "mclstm":
        cfg.update({
            "model": "mclstm",
            "hidden_size": 64,
            "initial_forget_bias": -3,
            "output_dropout": 0.0,
            "mass_inputs": ["pr_mm"],
            "dynamic_inputs": ["pet_mm", "tmean_c", "sin_doy", "cos_doy"],
            "custom_normalization": {
                "pr_mm": {"centering": None, "scaling": None},
                "q_mm_day": {"centering": None, "scaling": None},
            },
            "learning_rate": {0: 0.01, 20: 0.005, 25: 0.001},
            "clip_gradient_norm": 1,
        })
    elif model in {"author_dhbv", "author_mcp_ma3"}:
        dynamic = ["pr_mm", "pet_mm", "tmean_c"] if model == "author_dhbv" else ["pr_mm", "pet_mm"]
        raw = [f"{name}_copy1" for name in dynamic]
        rate = 0.001 if model == "author_dhbv" else 0.002
        cfg.update({
            "model": model,
            "dynamic_inputs": dynamic,
            "duplicate_features": {name: 1 for name in dynamic},
            "dynamic_conceptual_inputs": raw,
            "custom_normalization": {
                **{name: {"centering": None, "scaling": None} for name in raw},
                "q_mm_day": {"centering": None, "scaling": None},
            },
            "learning_rate": {0: rate, 20: rate / 2.0, 25: rate / 10.0},
        })
    elif model == "ported_fluxgate":
        dynamic = ["pr_mm", "pet_mm", "tmean_c", "sin_doy", "cos_doy"]
        cfg.update({
            "model": "ported_fluxgate",
            "dynamic_inputs": dynamic,
            "custom_normalization": {
                **{name: {"centering": None, "scaling": None} for name in dynamic},
                "q_mm_day": {"centering": None, "scaling": None},
            },
            "learning_rate": {0: 0.0015, 20: 0.00075, 25: 0.00015},
        })
    else:
        raise ValueError(model)
    return cfg


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    attributes = pd.read_csv(
        DATA / "attributes" / "attributes.csv", dtype={"basin": str}
    ).set_index("basin")
    donors = (LISTS / "train_500.txt").read_text().split()
    standard_deviation = attributes.loc[donors].std()
    excluded = standard_deviation[
        standard_deviation.eq(0) | standard_deviation.isna()
    ].index.tolist()
    features = [name for name in attributes if name not in excluded]
    files = []
    for model in MODELS:
        for seed in SEEDS:
            config = base(model, seed, features)
            path = DEST / f"full_{model}_seed{seed}.yml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            files.append(str(path.relative_to(ROOT)))
    manifest = {
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "split": {"donor": 500, "development": 200, "target": 3982},
        "static_feature_count": len(features),
        "excluded_zero_variance_features": excluded,
        "candidate_epochs": [10, 20, 30],
        "epochs": 30,
        "updates_per_epoch": 64,
        "target_information_used": False,
        "files": files,
    }
    (ROOT / "provenance" / "config_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
