#!/bin/bash
# MCP-MA3 seed-29/47 replay with the float64 clipping norm (PROTOCOL.md,
# Amendment 3), followed by selection, the analysis chain and the package check.
# In the original study this rerun replaced two failed replays; those failed
# runs are not part of this package. Safe to restart: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nh_mpl_cache" NGEN_SILENT=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
mkdir -p logs
echo "START MCP-MA3 replay $(date)"

for s in 29 47; do
  if [ -e runs_adaptive/author_mcp_ma3/seed$s/base_run/continue_training_from_epoch030 ] && \
     [ ! -e runs_adaptive/author_mcp_ma3/seed$s/base_run/continue_training_from_epoch030/model_epoch250.pt ]; then
    echo "An unfinished seed-$s replay exists. Move runs_adaptive/author_mcp_ma3/seed$s/base_run/continue_training_from_epoch030 aside and rerun."
    exit 2
  fi
done

$PY 02_replay_other_seeds.py --model author_mcp_ma3 --seed 29 < /dev/null > logs/02_replay_author_mcp_ma3_seed29.log 2>&1 &
p1=$!
$PY 02_replay_other_seeds.py --model author_mcp_ma3 --seed 47 < /dev/null > logs/02_replay_author_mcp_ma3_seed47.log 2>&1 &
p2=$!
wait $p1; wait $p2
$PY - <<'PYEOF'
import json, sys
bad = []
for s in (29, 47):
    r = json.load(open(f"replay_receipt_author_mcp_ma3_seed{s}.json"))
    skipped = r["skipped_nan_loss_batches"] + r["skipped_nonfinite_gradient_updates"]
    print(f"  MCP-MA3 seed {s}: skipped {skipped} of {r['updates']} updates, valid={r['valid']}")
    if not r["valid"]:
        bad.append(s)
if bad:
    sys.exit(f"STOP: replay not valid for seeds {bad} (more than 1 % of updates skipped)")
PYEOF

echo "Freeze selection $(date)"
$PY 03_build_selection_receipt.py > logs/03_selection.log 2>&1
echo "Analysis chain $(date)"
$PY 04_run_analysis.py > logs/04_analysis.log 2>&1
echo "Package check $(date)"
$PY ../workflow_steps/validate_package.py > logs/05_validate_package.log 2>&1
echo "ALL DONE $(date)"
