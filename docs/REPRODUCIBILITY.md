# Reproducibility guide

## What can be checked immediately

The repository includes the frozen split, model definitions, selected epoch
counts, fitted checkpoints, predictive metrics and fingerprints, statistical
summaries, independent audit outputs, and SHA-256 receipts. These support code
inspection and result reconciliation without downloading all raw inputs.

Run:

```bash
python workflow/validate_release.py
pytest -q
python conus_zero_shot_law_transfer_2026-09-30/scripts/00_test_mass_conservation.py
```

## Outcome-blind execution boundary

The primary workflow separates model development from target evaluation:

1. build the eligible population and freeze gauge roles;
2. train on donor gauges and select stopping epochs on development gauges;
3. refit frozen configurations and archive all target predictions and model
   states without opening target discharge;
4. freeze a prediction registry and its SHA-256 digest;
5. run zero-input interventions and archive drainage fingerprints; and
6. open target discharge for KGE, recession, and identification analyses.

Receipts record whether a stage accessed target discharge. The temporal
cross-fit audit further separates the target-discharge periods used for skill
qualification and observed recession benchmarking.

## Full reconstruction

End-to-end reconstruction requires the external sources listed in
`docs/DATA_SOURCES.md`, substantial disk space, and a machine suitable for
national neural hydrology training. The canonical stage order is stored in
`workflow/stages.json`.

```bash
python workflow/run_stage.py --list
python workflow/run_stage.py --group core --execute
python workflow/run_stage.py --group original --execute
python workflow/run_stage.py --group mcp12 --execute
python workflow/run_stage.py --group stress --execute
python workflow/run_stage.py --group reference --execute
python workflow/run_stage.py --group synthesis --execute
python workflow/run_stage.py --group trajectory --execute
python workflow/run_stage.py --group audit --execute
```

The process-observation ladder has its own frozen launcher:

```bash
python conus_daily_process_observability_2026-10-12/scripts/run_experiment.py --stage all
```

## Files intentionally not stored in GitHub

- raw provider downloads;
- the 4,682 × 7,759 forcing tensor and companion model-ready arrays;
- full target hydrograph arrays;
- complete 90-day intervention trajectories for every run;
- rendered article graphics and article source files.

These omissions do not change the analysis code or frozen results. The large
scientific arrays will be supplied in the versioned Zenodo archive.

## Numerical precision

The original intervention archive was produced in float32. Script
`scripts/11_verify_float64_interventions.py` replayed all 12 original
interventions in float64. The float64 receipts and compact fingerprints are in
`outputs/law_interventions_float64_audit/`. This replay reduced the maximum
closure residual to numerical roundoff while leaving the reported response
metrics unchanged at their published precision.
