# External data sources

Raw provider data are not redistributed. The archived scripts expect the
following sources; provenance files record products, dates, variables, and
checksums used in the completed experiments.

| Source | Use | Archived reference |
|---|---|---|
| USGS GAGES-II | gauge population, reference status, and catchment attributes | Falcone (2011) report, https://doi.org/10.3133/70046617; data release, https://doi.org/10.5066/P96CPHOT |
| USGS National Water Information System | daily discharge and observed recession extraction | https://waterdata.usgs.gov/nwis |
| GridMET | daily precipitation, reference evapotranspiration, and air temperature | https://www.climatologylab.org/gridmet.html |
| USGS SSEBop v2 | monthly actual evapotranspiration | https://doi.org/10.5066/P9L2YMV |
| NASA NLDAS-2 Noah L4 v2.0 | daily 12:00 UTC soil-water snapshots | https://doi.org/10.5067/T4OW83T8EXDO |
| ESA CCI Soil Moisture | independent satellite soil-moisture check | https://climate.esa.int/en/projects/soil-moisture/ |
| NHDPlusV2.1 | connected upstream–downstream gauge topology | https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus |

## Frozen population

The GAGES-II continental population was filtered by discharge and forcing
coverage before spatially stratified assignment:

- 500 donor gauges for model optimization;
- 200 development gauges for stopping-epoch selection and diagnostics; and
- 3,982 target gauges held out from model fitting.

The immutable assignments and 99 static attributes are in
`conus_zero_shot_law_transfer_2026-09-30/data/frozen/station_split.csv`.

## Expected local data roots

The original scripts resolve project-relative model-ready files. Provider
downloads should be stored outside version control. Before full reconstruction,
inspect the constants near the top of scripts 00–02 and set the local source
locations to your data root. Never place credentials in source code.

Key provenance records include:

- `provenance/population_and_split_summary.json`;
- `provenance/donor_ssebop_manifest.csv`;
- `provenance/model_data_summary.json`;
- `conus_daily_process_observability_2026-10-12/provenance/data_sources.md`;
- `conus_daily_process_observability_2026-10-12/provenance/frozen_input_checksums.csv`.
