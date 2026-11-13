# Dependency relocation note

On 2026-11-08, exact copies of the frozen arrays, observational benchmarks,
reference-only summaries, parity receipts, author-equation source, and
intervention helpers required by this experiment were placed under
`data/source_inputs/`, `adaptive_training/provenance/source_parity/`, and
`vendor/`. Their hashes are checked by
`scripts/00b_verify_self_contained.py` at the package root after the Zenodo
archives have been extracted.

For the collaborator package (12 November 2026), absolute workstation paths
inside 18 JSON receipts were rewritten as paths relative to
`adaptive_training/` (for example `runs_adaptive/...`, or `../runs/...` for the
MC-LSTM runs at the package root). The scripts join these paths to the
`adaptive_training/` folder, so they resolve on any machine. Only path text
changed: no number, array, checkpoint or recorded data hash changed.

Because the text changed, the SHA-256 of `preopening_freeze_registry.json`
changed too. The digest of the registry as frozen before target opening is
kept in `preopening_freeze_registry.original.sha256` (7178552b...), which is
also the value recorded in `outputs/evaluation/national_scientific_audit.json`.
`preopening_freeze_registry.sha256` holds the digest of the released file.
If the analysis chain is rerun, scripts that hash the registry record the
released digest.
