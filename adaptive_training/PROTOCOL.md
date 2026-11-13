# Adaptive learning-rate training protocol (fixed 2026-11-09, before any run)

Why: the earlier development-plateau runs (folder `convergence_sensitivity_2026-11-08/`,
not included in this package) continued from epoch 30 at the last, decayed learning rate of the original
schedule. A learning-rate restart check (script 21 there) raised seed-11
median development KGE by 0.023 (Hybrid-SHM), 0.044 (dHBV) and 0.045 (flux
gate), so those plateaus were step-size limits, not convergence. This folder
retrains the four closure-admitted lineages with one adaptive rule.

Outcome blindness: stopping and checkpoint selection use only the 200
development gauges (2015-2018). Target discharge, target KGE, R7 and all
intervention responses are not read until selection is frozen.

## Rule (seed 11 of each admitted lineage)

1. Start from the original epoch-30 weights and Adam moments. Set the learning
   rate to the lineage's original peak rate (Hybrid-SHM and dHBV 1e-3,
   MCP-MA3 2e-3, flux gate 1.5e-3). Gradient clipping, loss, batch size and
   data are unchanged.
2. Train in blocks of 10 pseudo-epochs (640 updates). After each block,
   compute median KGE at the 200 development gauges.
3. A block is a material improvement when its median development KGE is at
   least 0.005 above the running best (the running best starts at the
   epoch-30 value).
4. After 3 consecutive blocks without material improvement, halve the
   learning rate and reset the counter. At most 3 halvings (final rate =
   peak / 8).
5. After the third halving, stop after 5 consecutive blocks without material
   improvement. Stop in any case at epoch 400.
6. Select the checkpoint with the highest median development KGE among
   epochs 30 to the stopping epoch; a later epoch breaks an exact tie.
7. Seeds 29 and 47 start from their own epoch-30 weights and replay the
   exact seed-11 learning-rate schedule (the epochs at which the rate
   changed) up to the seed-11-selected epoch, with no development
   evaluation and no second selection.

MC-LSTM failed the closure audit, stays excluded from response inference and
is not retrained; its epoch-30 outputs are linked unchanged.

## After selection

The unchanged, audited analysis chain is rerun in this folder: batched blind
predictions (05b) with stock-evaluator parity, zero-PET interventions (06,
07, 08), the dHBV float64 replay (08b), the prespecified analysis (09, 11,
12, 13), the amended four-lineage analysis (16), the PET-at-climatology
interventions and comparison (00c, 17), the independent amended audit (18),
and the dHBV floor check (20).

## Amendment 2026-11-10 (after seed-11 selection, before any target use)

MCP-MA3 seeds 29 and 47 each stopped during the replay with a single NaN
training loss at the peak learning rate (epochs about 51 and 67). The replay
now sets NeuralHydrology's `allow_subsequent_nan_losses: 5`, which skips a
batch with a NaN loss without updating the weights and still stops training
after five NaN batches in a row. The option has no effect on runs without NaN
losses, so the six completed replays and all seed-11 runs are unaffected. The
number of skipped batches is written to `replay_receipt_<model>_seed<seed>.json`.
Target discharge had not been read when this change was made.

### Amendment 2026-11-10, second part

Skipping NaN losses was not enough: in both MCP-MA3 replays one update had a
non-finite gradient norm, gradient clipping then wrote NaN weights, and every
later batch was NaN. The vendored NeuralHydrology trainer now has an opt-in
config flag, `skip_nonfinite_gradient_steps` (default False), that skips an
update whose gradient norm is not finite and logs it. Only the seed 29/47
replays set it. All other runs, including every completed one, are unaffected
because the flag is off by default. Skipped updates are counted in
`replay_receipt_<model>_seed<seed>.json`. The unpatched trainer and config
files were kept in the original working folder and are not part of this package;
the change is the `skip_nonfinite_gradient_steps` block in the vendored
`training/basetrainer.py` and `utils/config.py`.

## Amendment 3 (2026-11-11): float64 gradient-clipping norm for MCP-MA3

Reason. The MCP-MA3 seed-29 and seed-47 replays skipped 9,132 and 10,833 of
14,080 optimizer updates, so those runs were not valid. The diagnostic
(`audit/diagnose_mcp_gradients.py`, result in
`provenance/mcp_gradient_diagnostic.json`) showed that the
gradients were very large but mostly finite (largest element 3.8e31, float64
norm 1.9e32). `clip_grad_norm_` computes the norm in float32, where it
overflows to inf, and the clipped gradient then becomes NaN. The seed-11 run
never overflowed and is unchanged.

Change. A new trainer option `float64_gradient_clipping` (vendored
basetrainer.py and config.py, default off) computes the clipping norm in
float64 and rescales the gradient exactly as `clip_grad_norm_` does. When the
float32 norm is finite the update is the same as standard clipping up to
rounding. An update is skipped only if the gradient itself holds inf or NaN.
The option is set for the MCP-MA3 seed-29 and seed-47 replays only. The
seed-11 schedule and selected epoch (250) are not changed.

Validity rule, fixed before the rerun. A replay is valid if at most 1 % of
its optimizer updates are skipped. If either seed fails, MCP-MA3 is reported
as numerically unstable at the peak learning rate and kept as descriptive
only; it does not enter the pairwise claims.

Result. Seeds 29 and 47 skipped 0 of 14,080 updates each
(`replay_receipt_author_mcp_ma3_seed29.json`, `..._seed47.json`), so MCP-MA3
meets the rule and enters the comparison.

The failed MCP-MA3 replays are not part of this collaborator package. The
diagnostic reads those failed checkpoints, so it cannot be rerun from this
package; its output is kept as the record. The corrected replay is launched
with `run_mcp_fix.sh`.

## Release notes on the seed replays

- The seed-29 and seed-47 replays of Hybrid-SHM, dHBV and the flux gate ran
  before replay receipts were added, so they have no
  `replay_receipt_*.json`. They ran with no safeguard switched on, finished
  every epoch, and therefore skipped no update.
- In the released `02_replay_other_seeds.py` the two NaN safeguards
  (`allow_subsequent_nan_losses`, `skip_nonfinite_gradient_steps`) are set for
  every model. They change nothing unless a NaN or a non-finite gradient
  occurs, which did not happen for those six replays. A fresh rerun writes a
  receipt for every replay, and its skipped-update counts show whether a
  safeguard was ever used.
