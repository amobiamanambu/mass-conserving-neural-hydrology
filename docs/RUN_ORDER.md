# Run order

## Fast verification

```bash
python workflow/validate_release.py
pytest -q
```

## Analysis replay from the Zenodo archive

Extract the Zenodo files into the repository root without changing their
relative paths. Verify the complete overlay before replaying the analysis:

```bash
python scripts/00b_verify_self_contained.py
cd adaptive_training
python 04_run_analysis.py
```

`04_run_analysis.py` verifies existing receipts and regenerates missing frozen
predictions, zero-PET interventions, the float64 dHBV replay,
climatological-PET interventions, pairwise comparisons, internal-process
diagnostics, and independent audits. The script uses the active Python
interpreter and refuses to start an expensive stage when less than 8 GiB is
free.

## End-to-end retraining

1. Materialize the provider inputs described in `DATA_SOURCES.md`.
2. Build the blinded NeuralHydrology dataset:

   ```bash
   python scripts/01_prepare_generic_dataset.py
   python scripts/01b_prepare_forcing_only_anchors.py
   python scripts/02_generate_configs.py
   ```

3. Train the fixed epoch-30 starting runs:

   ```bash
   python scripts/03_train_models.py --model all --seed all
   ```

4. Continue the closure-admitted models with the frozen adaptive rule:

   ```bash
   cd adaptive_training
   bash run_all.sh
   ```

The seed-11 stopping and checkpoint selection use only 200 development gauges.
Seeds 29 and 47 replay the seed-11 learning-rate schedule to the selected epoch.
Target discharge is opened only after checkpoint selection is frozen. See
`adaptive_training/PROTOCOL.md` for the complete rule and amendments.

Model training and national evaluation are computationally expensive. Use the
Zenodo checkpoints for audit and downstream replay unless full retraining is
scientifically necessary.
