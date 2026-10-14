# Mass-Conserving Neural Hydrology: Drainage-Response Identification

Reproducible CONUS experiments supporting the study **Can Mass-Conserving
Neural Hydrologic Models Identify Transferable Catchment Drainage Responses?**

This private pre-submission repository contains the authoritative scientific
code, frozen experiment definitions, trained checkpoints, compact derived
data, analysis products, and audit receipts. It does not contain article text,
document-building code, rendered publication graphics, writing instructions,
or superseded exploratory experiments.

## Scientific question

The experiments test whether useful zero-shot streamflow prediction and exact
water accounting are sufficient for distinct mass-conserving model
architectures to recover the same drainage response at unseen catchments.
Five primary conserving architectures are evaluated:

- differentiable HBV (`dhbv`);
- a neural flux-gate model (`fluxgate`);
- a three-partition mass-conserving perceptron (`mcp_ma3`);
- a conservative neural ordinary differential equation (`mc_node`); and
- a mass-conserving LSTM with typed outputs (`mclstm_dr`).

The frozen national design uses 500 donor gauges, 200 development gauges, and
3,982 unseen target gauges. Model responses are tested with common zero-input
interventions, full 30-day drainage trajectories, rain-screened observed
recessions, connected-gauge contrasts, reference-only retraining, numerical
precision checks, temporal cross-fitting, and an added process-observation
experiment using soil moisture and evapotranspiration.

## Repository map

```text
.
├── conus_zero_shot_law_transfer_2026-09-30/
│   ├── scripts/                       # primary models and analysis
│   ├── config/                        # frozen design and model selection
│   ├── data/frozen/                   # immutable split and feature catalog
│   ├── data/processed/                # compact station metadata
│   ├── runs/                          # frozen checkpoints and receipts
│   ├── outputs/                       # evaluation and identification products
│   ├── multimodel_neural_conservation_stress_test_2026-10-03/
│   ├── reference_only_conus_experiment_2026-10-03/
│   ├── independent_mcp_stress_test_2026-10-03/
│   ├── r7_curve_robustness_2026-10-09/
│   ├── temporal_crossfit_circularity_audit_2026-10-11/
│   └── audit/                         # independent inferential reanalysis
├── conus_daily_process_observability_2026-10-12/
│   ├── scripts/                       # soil-moisture and ET observation ladder
│   ├── protocols/                     # frozen scientific protocol
│   ├── runs/ and outputs/             # checkpoints and response products
│   └── results/ and provenance/       # summaries and source receipts
├── workflow/                          # run index and release validation
├── docs/                              # data, methods, and reproduction notes
└── tests/                             # fast structural checks
```

## Quick verification

Create the archived environment and run the lightweight checks:

```bash
conda env create -f environment.yml
conda activate mcnh-identifiability
python workflow/validate_release.py
pytest -q
python conus_zero_shot_law_transfer_2026-09-30/scripts/00_test_mass_conservation.py
```

`workflow/validate_release.py` checks the immutable split, the five-model
coverage, the process-observation result matrix, file sizes, local-path
leakage, release contents, and SHA-256 manifest. It does not retrain models.

To inspect the audited execution order without running expensive stages:

```bash
python workflow/run_stage.py --list
python workflow/run_stage.py --group original
```

Add `--execute` only after reconstructing the external inputs described in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Full refitting and national
prediction are computationally expensive.

## Reproduction levels

1. **Artifact verification:** use the included checkpoints, compact outputs,
   receipts, and checksums to inspect the reported analysis without obtaining
   raw provider data.
2. **Analysis replay:** use the archived model products and run downstream
   identification, sensitivity, and audit scripts.
3. **End-to-end reconstruction:** obtain the provider-controlled inputs, build
   the model-ready arrays, select epochs on development gauges, refit on donor
   gauges, freeze target predictions, and then open target discharge for
   evaluation. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Large model-ready arrays and complete intervention trajectories are omitted
from GitHub. They will be deposited in the companion Zenodo archive and are
reconstructable from the frozen code and source manifests.

## Data and code licenses

Source code is released under the [MIT License](LICENSE). Repository-authored
derived tables and receipts are released under
[CC BY 4.0](LICENSE-DATA.md). External datasets retain the licenses and terms
of their providers and are not redistributed here.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). A versioned DOI
will be added after the linked Zenodo record is reserved.
