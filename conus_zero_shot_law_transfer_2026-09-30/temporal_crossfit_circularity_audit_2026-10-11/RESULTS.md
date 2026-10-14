# Temporal cross-fit circularity audit

The audit used disjoint target-discharge periods for predictive qualification and the observed-recession benchmark. Archived model fingerprints were not recomputed or tuned.

## Results

| Design | Gauges | Q+ET rho [95% CI] | Median absolute R7 gap [95% CI] | dHBV-observed rho [95% CI] | Flux-gate-observed rho [95% CI] |
|---|---:|---:|---:|---:|---:|
| All 3,982 targets / no target Q | 3,982 | 0.844 [0.836, 0.853] | 0.112 [0.109, 0.115] | — | — |
| Full 2019–24 / same record | 2,317 | 0.750 [0.732, 0.767] | 0.103 [0.099, 0.106] | 0.312 [0.277, 0.345] | 0.364 [0.331, 0.397] |
| Skill 2019–21 / R7 2022–24 | 2,028 | 0.784 [0.765, 0.800] | 0.100 [0.095, 0.103] | 0.312 [0.275, 0.350] | 0.347 [0.310, 0.385] |
| Skill 2022–24 / R7 2019–21 | 1,604 | 0.803 [0.786, 0.820] | 0.106 [0.102, 0.111] | 0.365 [0.327, 0.402] | 0.424 [0.383, 0.460] |

## Cross-fit sample construction

| Fold | Mutual KGE-qualified | Usable benchmark R7 | Final gauges | Median events |
|---|---:|---:|---:|---:|
| skill_2019_2021__benchmark_2022_2024 | 2,371 | 3,373 | 2,028 | 7.0 |
| skill_2022_2024__benchmark_2019_2021 | 2,196 | 3,094 | 1,604 | 5.0 |

## Interpretation

Both temporal folds retained material response disagreement. The same-record design is therefore not required for the central dHBV–flux-gate magnitude result.

The cross-architecture rank correlations rose above the prespecified 0.75 criterion in both temporal folds. The cross-fit therefore supports persistent magnitude disagreement, but not a claim that response ranks must also disagree after temporal separation.

Both architectures remained below the observed-benchmark rank criterion and above its magnitude-error criterion in both folds. The all-target analysis also retained a median gap above 0.10 without using target discharge for either selection or benchmarking.

The audit addresses reuse of target discharge for KGE qualification and observed-recession extraction. It does not convert the discharge-derived recession proxy into a direct storage observation.
