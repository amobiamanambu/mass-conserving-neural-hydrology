# Data dictionary

## Core populations

- `data/basin_lists/train_500.txt`: donor gauges used for model fitting.
- `data/basin_lists/validation_200.txt`: development gauges used for stopping and
  checkpoint selection.
- `data/basin_lists/test_3982.txt`: unseen target gauges used only after model
  selection was frozen.
- `data/forcing_only_anchors.npz`: twelve forcing-derived antecedent anchors per
  eligible target gauge.

## Compact final products

- `adaptive_training/outputs/evaluation_amended/amended_architecture_summary.csv`:
  model skill, drainage-response summaries, seed reproducibility, and observed
  recession comparisons.
- `adaptive_training/outputs/evaluation_amended/amended_pairwise_summary.csv`:
  pairwise response comparisons across KGE thresholds.
- `adaptive_training/outputs/evaluation_pet_climatology/`:
  the same response comparison with seasonal PET retained.
- `adaptive_training/outputs/frozen/*_fingerprints.csv`:
  seven-day fingerprints for individual model seeds.
- `adaptive_training/provenance/`:
  protocol amendments, blind-target checks, selection receipts, source parity,
  and closure evidence.

The large NPZ archives use the same model and seed labels. Their hashes are
recorded in the adjacent JSON receipts and the Zenodo manifest.
