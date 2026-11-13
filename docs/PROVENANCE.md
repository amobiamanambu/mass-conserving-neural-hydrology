# Provenance notes

The scientific products were generated under the dated project directory
`neuralhydrology_literature_five_model_national_2026-11-01` and finalized in the
adaptive-training run completed on 11 November 2026.

For release, absolute workstation paths in copied text receipts were removed.
Paths that scripts read (checkpoints, run folders, receipts) are now relative to
`adaptive_training/`, so they resolve on any machine; a few descriptive fields
still use placeholders such as `${LEGACY_SOURCE}` or `${CONDA_ENV}`. This
changed no numerical field, array, checkpoint, or recorded data hash. The
original and released SHA-256 digests of the pre-opening registry are kept side
by side in `adaptive_training/provenance/` (see `RELOCATION_NOTE.md` there).

The release form of `adaptive_training/00_prepare.py` discovers the archived
epoch-30 runs under `runs/` instead of referring to the original workstation's
dated convergence directory. The copied states and their hashes remain the
same. Runtime wrappers use the active Python interpreter rather than a local
absolute interpreter path.
