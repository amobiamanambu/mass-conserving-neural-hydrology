# Data sources

The model-ready archive combines public hydrologic and meteorological products.
Provider terms remain authoritative.

- **USGS streamflow:** daily discharge used for donor/development training and
  target evaluation.
- **GridMET:** daily precipitation, reference evapotranspiration, and temperature
  forcings.
- **GAGES-II:** basin attributes, reference status, and gauge metadata.
- **NHDPlus/NLDI:** connected-gauge relationships used for the nested-pair
  diagnostic.
- **Rain-free observed recessions:** derived from USGS hydrographs with the
  archived extraction rules.

The GitHub repository includes basin identifiers and compact derived results.
The private Zenodo review archive contains the model-ready arrays and supporting
source tables needed for replay. No account credentials or provider tokens are
archived.
