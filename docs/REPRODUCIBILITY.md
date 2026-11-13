# Reproducibility design

The archive separates checkpoint selection from target evaluation.

- Training uses 500 donor gauges from 5 October 2003 through 31 December 2014.
- Development selection uses 200 gauges from 2015 through 2018.
- Evaluation uses 3,982 unseen gauges from 2019 through 2024.
- Three seeds (11, 29, and 47) are retained for every architecture.
- Seed 11 determines the adaptive learning-rate schedule and selected epoch from
  development KGE only; seeds 29 and 47 replay that schedule without a second
  selection.
- The forcing-only antecedent anchors are outcome blind.
- Streamflow skill comparisons are conditioned on gauges at which both models
  meet the stated KGE threshold.
- The zero-PET intervention was prespecified. PET-at-climatology was added as a
  sensitivity test after the zero-PET behavior had been inspected.
- dHBV enters the response comparison through a float64 replay because its
  float32 residuals were precision noise rather than structural mass loss.
- MCP-MA3 enters after a stable softmax partition and float64 gradient-norm
  calculation were verified against the original equations and accepted by the
  prespecified skipped-update rule.
- MC-LSTM remains descriptive because the tested implementation did not satisfy
  the structural closure audit.

The companion Zenodo archive contains the large inputs and products omitted
from GitHub. `ZENODO_ARCHIVE.md` documents the file groups and checksums.
