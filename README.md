# Mass-Conserving Neural Hydrology: Drainage-Response Identification

This repository contains the reproducible analysis for the study **Can
Mass-Conserving Neural Hydrologic Models Identify Transferable Catchment
Drainage Responses?** It tests whether mass-conserving models that predict
streamflow adequately at unseen catchments recover the same drainage response.

The final experiment uses the NeuralHydrology framework, a fixed national split
of 500 training gauges, 200 development gauges, and 3,982 unseen target gauges,
and three independently trained seeds. Four architectures passed the numerical
closure audit and entered the response comparison:

- Hybrid-SHM;
- a differentiable HBV adapter (`author_dhbv`);
- MCP-MA3; and
- a NeuralHydrology-compatible neural flux-gate model.

MC-LSTM is retained as a predictive benchmark but excluded from response
inference because its tested implementation did not pass the structural closure
audit.

## Main result

At gauges where both models achieved KGE >= 0.60, exact conservation and useful
streamflow transfer did not guarantee the same drainage response. Under the
prespecified zero-PET intervention, Hybrid-SHM differed from dHBV and the flux
gate by more than their within-architecture seed variation. The same directional
separation remained when PET was held at its seasonal climatology. MCP-MA3's
zero-PET response included rising flow at many gauges, so the climatological-PET
test is reported as an explicit sensitivity rather than a replacement for the
prespecified experiment. Complete statistics are in
`adaptive_training/outputs/evaluation_amended/` and
`adaptive_training/outputs/evaluation_pet_climatology/`.

## Repository contents

```text
configs/                 relocatable NeuralHydrology configurations
scripts/                 initial training, evaluation, and audit code
adaptive_training/       plateau training, interventions, and final analyses
data/basin_lists/        frozen donor, development, and target identifiers
vendor/                  exact model-framework sources and third-party licenses
workflow/                archive manifests and release validation
tests/                   lightweight reproducibility tests
docs/                    data, run-order, provenance, and archive documentation
```

The repository intentionally excludes manuscript text, document-building code,
publication figures, writing instructions, superseded exploratory experiments,
and failed MCP runs. Large binary products are stored in the companion private
Zenodo review archive (record 23358088): model-ready data, selected checkpoints,
frozen predictions, intervention trajectories, and the full gauge-level process
tables.

## Quick verification

```bash
conda env create -f environment.yml
conda activate mcnh-identification
python workflow/validate_release.py
pytest -q
```

These checks validate file structure, hashes, model coverage, fixed population
counts, result-table consistency, and the absence of local paths or publication
artifacts. They do not retrain the models.

## Reproduction levels

1. **Inspect results.** Use the compact tables and receipts included here.
2. **Replay the analysis.** Add the Zenodo `data/`, `runs/`, and large
   `adaptive_training/outputs/` products, then run the analysis stages in
   `docs/RUN_ORDER.md`.
3. **Retrain end to end.** Build the model-ready data, run the epoch-30 training,
   and continue with the outcome-blind adaptive protocol in
   `adaptive_training/PROTOCOL.md`.

## Licenses and citation

Repository-authored code is licensed under MIT. Repository-authored derived
tables and receipts are licensed under CC BY 4.0. Vendored sources retain their
original licenses; see `THIRD_PARTY_NOTICES.md`. External datasets remain subject
to provider terms. Citation metadata are in `CITATION.cff`.
