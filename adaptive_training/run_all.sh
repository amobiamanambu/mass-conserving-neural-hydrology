#!/bin/bash
# Adaptive-learning-rate retraining and full re-analysis (see PROTOCOL.md).
# Safe to restart: finished steps are skipped. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/nh_mpl_cache" NGEN_SILENT=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p logs
MODELS="hybrid_shm author_dhbv author_mcp_ma3 ported_fluxgate"
echo "START $(date)"

echo "STEP 0 prepare $(date)"
$PY 00_prepare.py > logs/00_prepare.log 2>&1

echo "STEP 1 seed-11 adaptive training, 4 models in parallel $(date)"
pids=""
for m in $MODELS; do
  $PY 01_adaptive_train_seed11.py --model $m < /dev/null > logs/01_seed11_$m.log 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait $p; done
for m in $MODELS; do
  $PY -c "import json;r=json.load(open('adaptive_receipt_$m.json'));print('  $m', r['status'], 'selected epoch', r['selected_epoch'], 'dev KGE %.3f' % r['selected_median_development_kge'])"
done

echo "STEP 2 seeds 29 and 47 replay the seed-11 schedule, 4 at a time $(date)"
for m in $MODELS; do for s in 29 47; do echo "$m $s"; done; done | \
  xargs -P 4 -L 1 bash -c '"$0" 02_replay_other_seeds.py --model $1 --seed $2 < /dev/null > logs/02_replay_$1_seed$2.log 2>&1' $PY

echo "STEP 3 freeze selection $(date)"
$PY 03_build_selection_receipt.py > logs/03_selection.log 2>&1
cat logs/03_selection.log

echo "STEP 4 predictions, interventions, PET probe, analyses, audits $(date)"
$PY 04_run_analysis.py > logs/04_analysis.log 2>&1

echo "STEP 5 check the package (counts, hashes, figures, paths) $(date)"
$PY ../workflow_steps/validate_package.py > logs/05_validate_package.log 2>&1
touch ALL_DONE
echo "ALL DONE $(date)"
