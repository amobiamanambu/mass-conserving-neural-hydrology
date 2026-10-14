# Temporal cross-fit circularity audit

This isolated audit tests whether reusing 2019–2024 target discharge for both
KGE qualification and the observed-recession proxy affected the original
dHBV–flux-gate conclusion. It does not modify the archived primary analysis, the archived
model outputs, or the main analysis pipeline.

The two cross-fit folds are:

1. KGE from 2019–2021 and observed recessions from 2022–2024.
2. KGE from 2022–2024 and observed recessions from 2019–2021.

Within a fold, no target-discharge day contributes to both model qualification
and the observational benchmark. Model fingerprints remain the archived,
outcome-blind intervention products used in the primary analysis.

Run from the project root with the bundled Python runtime:

```bash
python temporal_crossfit_circularity_audit_2026-10-11/run_temporal_crossfit.py
```

The script writes gauge-level data, bootstrap draws, summary tables, a receipt,
a short results report, and a CHE-style diagnostic figure inside this folder.
