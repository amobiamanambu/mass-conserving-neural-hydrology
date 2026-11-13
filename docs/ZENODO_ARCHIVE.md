# Zenodo archive layout

The companion Zenodo review record is split into independently downloadable
archives:

1. `mcnh_code_snapshot.zip`: the GitHub release at the cited commit.
2. `mcnh_model_ready_data.zip`: blinded NeuralHydrology inputs, source tables,
   basin lists, and forcing-only anchors.
3. `mcnh_selected_checkpoints.zip`: epoch-30 starting states, selected adaptive
   checkpoints, scalers, configurations, and validation histories.
4. `mcnh_frozen_predictions.zip`: target predictions and zero-PET intervention
   trajectories.
5. `mcnh_pet_sensitivity_and_audits.zip`: climatological-PET trajectories,
   float64 dHBV replays, final evaluations, internal-process tables, and audit
   receipts.

`ZENODO_MANIFEST.csv` records each archive's byte size and SHA-256 digest.
`ZENODO_CONTENTS.csv` records every archived file and its digest. Failed MCP
runs, manuscript text, document-building code, and rendered publication figures
are excluded.
