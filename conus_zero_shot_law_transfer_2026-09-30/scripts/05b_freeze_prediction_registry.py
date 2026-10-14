#!/usr/bin/env python3
"""Verify and freeze all outcome-blind national predictions and fingerprints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (PROJECT / "data/evaluation/q_target_2019_2024.npy").exists():
        raise RuntimeError("Target discharge already exists; cannot create an outcome-blind registry")
    expected = [f"ealstm_q_seed{s}" for s in (11, 29, 47)]
    expected += [
        f"{architecture}_{rung}_seed{seed}"
        for architecture in ("dhbv", "fluxgate")
        for rung in ("q", "qet")
        for seed in (11, 29, 47)
    ]
    records = []
    for run in expected:
        receipt_path = PROJECT / "runs/frozen_refit" / run / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        prediction_path = PROJECT / receipt["prediction"]
        prediction = np.load(prediction_path, mmap_mode="r")
        if prediction.shape != (3982, 2192) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid prediction array for {run}: {prediction.shape}")
        if sha256(prediction_path) != receipt["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {run}")
        record = {
            "run": run,
            "architecture": receipt["architecture"],
            "rung": receipt["rung"],
            "seed": receipt["seed"],
            "prediction": receipt["prediction"],
            "prediction_sha256": receipt["prediction_sha256"],
            "target_discharge_accessed": receipt["target_discharge_accessed"],
        }
        if receipt["architecture"] != "ealstm":
            law_receipt_path = PROJECT / "outputs/law_interventions" / f"{run}_receipt.json"
            law = json.loads(law_receipt_path.read_text())
            for field, checksum_field in (("curves", "curves_sha256"), ("fingerprints", "fingerprints_sha256")):
                path = PROJECT / law[field]
                if sha256(path) != law[checksum_field]:
                    raise RuntimeError(f"Law checksum mismatch for {run}/{field}")
            fingerprint = pd.read_parquet(PROJECT / law["fingerprints"])
            if len(fingerprint) != 3982:
                raise RuntimeError(f"Fingerprint count mismatch for {run}")
            record.update({
                "antecedent_states": receipt["antecedent_states"],
                "antecedent_states_sha256": receipt["antecedent_states_sha256"],
                "law_curves": law["curves"],
                "law_curves_sha256": law["curves_sha256"],
                "law_fingerprints": law["fingerprints"],
                "law_fingerprints_sha256": law["fingerprints_sha256"],
            })
        records.append(record)
    registry = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_discharge_accessed": False,
        "expected_runs": len(expected),
        "target_gauges": 3982,
        "target_days": 2192,
        "runs": records,
    }
    path = PROJECT / "provenance/frozen_prediction_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n")
    checksum_path = PROJECT / "provenance/frozen_prediction_registry.sha256"
    checksum_path.write_text(f"{sha256(path)}  {path.name}\n")
    print(json.dumps({"registry": str(path), "sha256": sha256(path), "runs": len(records)}, indent=2))


if __name__ == "__main__":
    main()
