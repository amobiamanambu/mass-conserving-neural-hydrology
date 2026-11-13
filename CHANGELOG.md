# Changelog

## 1.0.0 - 2026-11-12

- Replaced the superseded independent-model formulation with the final
  NeuralHydrology-based national experiment.
- Added outcome-blind adaptive training to development plateau for four
  closure-admitted architectures and three seeds.
- Added the float64 dHBV closure replay and the stable MCP-MA3 gradient-clipping
  implementation.
- Added both the prespecified zero-PET intervention and the climatological-PET
  sensitivity analysis.
- Added compact final results, audit receipts, relative-path configurations,
  third-party notices, and automated release validation.
- Excluded failed MCP runs, manuscript files, rendered publication figures, and
  superseded exploratory products.

## 1.0.1 - 2026-11-13

- Replaced executable workstation paths in receipts with paths relative to
  `adaptive_training/` and aligned the replay scripts with those paths.
- Replaced the macOS-specific cache path with the system temporary directory.
- Added the stable MCP-MA3 rerun wrapper and corrected the corresponding
  protocol and provenance records.
- Updated the post-Zenodo self-containment check to follow the released archive
  layout and verify the selected checkpoints and receipts.
