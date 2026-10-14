# Methods and run order

## Primary national experiment

The primary experiment uses one immutable spatial split and three random seeds
(11, 29, and 47). Daily inputs span 5 October 2003 through 31 December 2024.
Donor refits end in 2018; target evaluation covers 2019–2024.

The canonical source order is:

1. `00_build_attributes_and_split.py` — build attributes and freeze roles.
2. `01_build_forcing_and_training_targets.py` — assemble GridMET forcing,
   static matrices, and donor/development discharge.
3. `02_extract_donor_ssebop.py` — extract donor ET summaries.
4. `03_select_on_development.py` — select stopping epochs without target flow.
5. `04_refit_and_predict_targets.py` — refit and save outcome-blind predictions.
6. `05_run_zero_input_interventions.py` — compute model drainage fingerprints.
7. `05b_freeze_prediction_registry.py` — seal the prediction registry.
8. `06_open_target_discharge_and_evaluate.py` — evaluate target hydrographs.
9. `07_extract_observed_recessions.py` — build the rain-screened benchmark.
10. `08*` and `09*` — identification, sensitivity, reference, and mechanism tests.
11. `11*` through `13*` — precision, topology, and practical-response audits.

Exact commands and expected outputs are in `workflow/stages.json`. The runner
prints commands by default and requires `--execute` to change scientific
outputs.

## Additional architecture tests

The multi-model directory contains the three primary additional conserving
architectures and two structural variants. The independent MCP directory
contains the earlier 12-cell diagnostic. Each uses the same national split,
forcing, static attributes, seeds, target period, and intervention definition.

## Reference-only experiment

Reference gauges are re-split into 400 donor, 100 development, and 413 target
gauges. Models are retrained within that population; this is not merely a
filter applied to the national predictions.

## Drainage-response test

Frozen models are initialized from 12 antecedent states and run for 90 days
with precipitation and PET set to zero and temperature fixed at 10 °C. The
normalized response is `R(t) = Q(t) / Q(0)`. Seven-day retention (`R7`) is an
interpretable summary, while the full-curve analysis compares days 1–30.

## Process-observation ladder

The process experiment compares Q-only training with soil-moisture, ET, and
combined auxiliary supervision. It preserves the split, seeds, architecture
definitions, and target evaluation boundary of the source experiment. Its
frozen scientific design is in
`conus_daily_process_observability_2026-10-12/protocols/frozen_protocol.md`.
