# Compact data dictionary

## Frozen design

`data/frozen/station_split.csv`

- `GAGE_ID`: eight-character USGS station identifier.
- `role`: `donor`, `development`, or `target`.
- `ecoregion`: hydrologic ecoregion used for stratification.
- `quality_tier`: source-data completeness tier.
- `reference_status`: GAGES-II reference classification.
- remaining columns: location, drainage area, and the 99 model attributes.

## Predictive evaluation

`outputs/evaluation/target_model_metrics.parquet` contains gauge-level KGE,
NSE, log1p-NSE, bias, architecture, rung, and seed. The companion
`target_skill_summary.csv` contains ensemble and seed summaries.

## Intervention fingerprints

Each `*_fingerprints.parquet` or `*_fingerprints.csv` stores gauge- and
anchor-level summaries from a frozen zero-input intervention, including R7,
R30, R90, half-life, and storage–discharge diagnostics when defined. Receipt
files record checkpoints, precision, closure residuals, and input hashes.

## Identification products

- `gauge_level_identification.parquet`: original-pair, gauge-level comparison.
- `primary_identification_population.parquet`: mutually skill-qualified set.
- `sensitivity_metrics.csv`: skill and recession-rule sensitivity grid.
- `subgroup_metrics.csv`: ecoregion and data-quality summaries.
- `nested_pair_metrics.csv`: connected upstream–downstream contrasts.
- `strengthened_test/`: common-date anchors, flux traces, proxy reliability,
  reference controls, and cross-architecture decomposition.

## Process-observation products

`conus_daily_process_observability_2026-10-12/results/` contains predictive
competence, process learning, paired architecture-gap changes, seed stability,
connected-gauge contrasts, and the frozen decision record.
