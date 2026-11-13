# Provenance notes

The scientific products were generated under the dated project directory
`neuralhydrology_literature_five_model_national_2026-11-01` and finalized in the
adaptive-training run completed on 11 November 2026.

For release, absolute workstation paths in copied text receipts were replaced by
`${PROJECT_ROOT}`, `${LEGACY_SOURCE}`, `${PORT_SOURCE}`, `${CONDA_ENV}`, and
`${PYTORCH_BUILD_ROOT}`. This path normalization changed no numerical field,
array, checkpoint, or scientific hash. The original and normalized SHA-256
digests of the pre-opening registry are retained side by side in
`adaptive_training/provenance/`.

The release form of `adaptive_training/00_prepare.py` discovers the archived
epoch-30 runs under `runs/` instead of referring to the original workstation's
dated convergence directory. The copied states and their hashes remain the
same. Runtime wrappers use the active Python interpreter rather than a local
absolute interpreter path.
