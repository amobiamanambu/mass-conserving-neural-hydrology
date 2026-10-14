# Data sources

## Frozen source-experiment inputs

Daily GridMET forcing, USGS discharge, 99 static attributes, and the immutable
500/200/3,982 spatial split are read from
`../conus_zero_shot_law_transfer_2026-09-30/data/`. Their hashes are recorded
by `scripts/00_freeze_inputs.py`; the files are not copied or modified.

## NLDAS-2 Noah

- Product: NLDAS Noah Land Surface Model L4 Hourly 0.125 x 0.125 degree V2.0
- Short name: `NLDAS_NOAH0125_H`
- DOI: `10.5067/T4OW83T8EXDO`
- Source: NASA GES DISC / Earthdata Cloud
- Temporal sample used: one 12:00 UTC snapshot per day
- Variables: `SoilM_0_10cm`, `SoilM_0_100cm`

## SSEBop

- Product: USGS monthly SSEBop actual evapotranspiration, version 2
- DOI: `10.5066/P9L2YMV`
- Source: USGS Early Warning and Environmental Monitoring Program
- Native file pattern: `mYYYYMM.zip`
- Native units: millimeters per month
- Dynamic archive period: 2015-2018
- Training target: calendar-month total

## Independent satellite check

ESA CCI Combined surface soil moisture v09.2 basin-week values and source
manifests are reused read-only from
`../conus_hydrologic_law_identifiability_2026-09-30/`.
