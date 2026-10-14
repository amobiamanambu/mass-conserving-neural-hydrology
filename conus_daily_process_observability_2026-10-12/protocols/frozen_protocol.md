# Frozen protocol: daily process observations and drainage-response identification

Frozen on 2026-10-12 before any new model was trained or any new intervention
result was examined.

The same-day pre-result source audit replaced the proposed daily SSEBop
extraction with the official monthly series for 2015-2018 after confirming
that summer daily files approach 15 MB each. This amendment was made before
process tensors, augmented checkpoints, or new evaluation results existed;
the retired daily source probes are archived under
`provenance/retired_daily_probe/`.

## Question

Do dynamic soil-moisture and evapotranspiration observations make the drainage
response of exactly mass-conserving hydrologic models more reproducible across
random seeds and architectures at unseen catchments?

The experiment tests a remedy for the non-identification diagnosed in the
source experiment. It does not retest whether streamflow-only training is
sufficient; that result remains frozen in the original project.

## Population and temporal split

The experiment reuses the immutable national split from the source experiment:

- 500 donor gauges for optimization;
- 200 development gauges for epoch selection and loss diagnostics;
- 3,982 target gauges for zero-shot evaluation.

No target discharge, target process observation, observed recession, or
intervention result may be used to set loss weights, epochs, architecture, or
observation operators.

The common daily record is 2003-10-05 through 2024-12-31. The Q-only
checkpoints and their epoch counts were frozen in the source experiment before
this study began. Process-supervised counterparts inherit those epoch counts
and are fitted to donor data through 2018. Development gauges are used only to
test whether the intended process metric improves. The frozen target
evaluation period is 2019-2024.

## Dynamic observations

1. NLDAS-2 Noah L4 hourly 0.125-degree output, version 2.0
   (`NLDAS_NOAH0125_H`, DOI `10.5067/T4OW83T8EXDO`). One 12:00 UTC snapshot per
   day is used. This is a daily dynamic snapshot, not a 24-hour average. The
   archived variables are 0-10 cm and 0-100 cm soil-water storage in kg m-2,
   numerically equivalent to millimeters.
2. USGS SSEBop monthly actual evapotranspiration, version 2, DOI
   `10.5066/P9L2YMV`, for 2015-2018. Monthly totals enter the loss directly.
   The 48-month record provides a dynamic ET constraint at spatially held-out
   development gauges while keeping the streamed, checksummed archive
   tractable. It does not constrain daily ET timing.
3. ESA CCI Combined surface soil moisture v09.2 is retained as an independent
   satellite validation target on the dates and gauges already archived in
   the preceding weekly experiment. It is never an input or a training target
   in the confirmatory daily ladder.

NLDAS is model-generated land-surface output and SSEBop uses remote sensing
plus meteorological forcing. They are process constraints, not direct truth.
Their value is judged by whether they improve independent target-gauge tests.

## Models and observation ladder

Two exactly mass-conserving families are copied from the frozen source code:
differentiable HBV and the neural flux-gate model. Three seeds (11, 29,
47) are used for every rung:

1. `q`: streamflow only;
2. `q_sm`: streamflow plus daily NLDAS root-zone anomaly;
3. `q_et`: streamflow plus monthly SSEBop ET;
4. `q_sm_et`: streamflow plus both process observations.

Process observations are auxiliary targets during donor training, never input
predictors at transfer time. The total auxiliary weight is fixed at 0.20. ET
loss is active only for training windows overlapping 2015-2018; discharge and
soil-moisture losses retain the full donor period. When both process targets
are available, their standardized losses are averaged so that the total
process weight does not double.

The soil observation operator converts NLDAS 0-100 cm storage to a robust
within-basin relative-saturation proxy using its 5th and 95th percentiles over
2005-2014, then compares that quantity with model soil storage divided by the
model's learned soil capacity. The ET operator compares log-transformed
calendar-month totals on dates with adequate source coverage. For donor
optimization, every normalization statistic is estimated from donor training
dates only; target-basin percentiles are used only for held-out evaluation.

## Competence and learning checks

For each architecture and seed, an augmented rung is usable only if:

1. target-period streamflow KGE decreases by no more than 0.05 relative to the
   matched Q-only run;
2. median target KGE is at least 0.30 for the architecture pair being compared;
3. the targeted process metric improves on development gauges relative to the
   matched Q-only run;
4. daily and cumulative mass-balance residuals remain within float64 numerical
   tolerance.

Failed competence makes the corresponding identification comparison
inconclusive rather than negative.

## Primary response test

The primary endpoint is the entire normalized zero-input drainage curve, not
R7 alone. Frozen models are initialized from the same 12 forcing-defined
antecedent dates at every target gauge, then run for 90 days with precipitation
and PET set to zero and temperature fixed at 10 degrees C. For each mutually
skill-qualified gauge, seed-median architecture disagreement is measured as
the mean absolute difference between normalized Q(t)/Q(0) curves over days
1-30. Days 7, 14, and 30 are prespecified summaries.

The observation ladder supports improved identification only if, relative to
Q-only:

1. median 30-day curve disagreement falls by at least 30%;
2. the 95% ecoregion-stratified bootstrap interval for the reduction excludes
   zero;
3. cross-seed curve disagreement falls by at least 30% in both architectures;
4. cross-architecture rank agreement for R7 is at least 0.75;
5. agreement with rain-screened USGS recession curves and nested-gauge
   contrasts improves rather than merely forcing the two models toward the
   same wrong response;
6. the direction holds at mutual KGE thresholds 0.30, 0.40, 0.50, and 0.60 and
   in reference gauges.

R7 remains a secondary, interpretable point on the curve. A result based only
on R7 cannot satisfy the protocol.

## Independent process checks

NLDAS-supervised models are evaluated against held-out NLDAS dates and against
ESA CCI surface-soil-moisture anomalies on overlapping target gauges. SSEBop-
supervised models are evaluated on held-out monthly ET totals at development
gauges. Improvement against the training product without improvement against
the independent satellite product is reported as product imitation, not
physical identification.

## Decision language

- **Evidence for a remedy:** `q_sm`, `q_et`, or `q_sm_et` passes every
  competence and response criterion.
- **Partial constraint:** process prediction improves and response disagreement
  shrinks, but one or more identification criteria fail.
- **No remedy in this design:** competence passes but response disagreement
  does not materially shrink.
- **Inconclusive:** coverage, predictive competence, or numerical closure fails.

Interpretation was deferred until the complete frozen result matrix had been
generated and independently checked.
