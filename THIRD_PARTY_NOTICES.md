# Third-party software notices

The repository contains source snapshots needed to reproduce the analysis.

- `vendor/neuralhydrology_literature_adapter/` is based on NeuralHydrology
  1.13.0, upstream commit `9528d1d13eb82c9f038ee489d43784bc7aeb08cc`,
  and retains its BSD 3-Clause license. The snapshot includes the model adapters
  and opt-in numerical safeguards used in the archived runs.
- `vendor/mcp_author_source/` retains the MIT license and copyright notice of
  Yuan-Heng Wang.
- `vendor/reference_helpers/` contains repository-specific replay helpers used
  to audit numerical closure and does not replace the original model sources.

NeuralHydrology should be cited through the citation metadata included in its
vendored directory (Kratzert et al., 2022, DOI 10.21105/joss.04050).
