#!/usr/bin/env bash
# Full Level-0 pipeline, run sequentially on the compute box.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_level0.sh \
#       > logs/level0.log 2>&1 < /dev/null &'
#
# Resumable at two levels: each stage has a sentinel file, and the reporting stages
# pass --skip-existing so individual (baseline, k-shot, seed) cells are not recomputed.
# Safe to re-run after an interruption.
#
# IMPORTANT: stages that depend on tuning abort if tuning did not complete. The first
# version of this script used `set -uo pipefail` without `-e`, so when the 20-shot
# sweep died of an out-of-memory error partway through, the merge and reporting stages
# ran anyway -- and three baselines were silently reported at an untuned default
# learning rate. The run looked successful. Hence `fail()` below, and hence the
# explicit coverage check after the sweep rather than trusting an exit code alone.

set -uo pipefail

PY=.venv/bin/python
EPISODES=${EPISODES:-600}
SEEDS=${SEEDS:-"0 1 2"}
TUNED=results/best_lr_eurosat.json

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# -- 1. report 1- and 5-shot (tuned by the earlier sweeps) -------------------
if [ ! -f results/level0/.done_1_5 ]; then
    stage "reporting 1- and 5-shot, ${EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_baselines.py \
        --k-shot 1 5 --episodes "$EPISODES" --seeds $SEEDS \
        --tuned "$TUNED" --skip-existing --out results/level0 \
        || fail "1/5-shot reporting failed"
    touch results/level0/.done_1_5
else
    echo "[skip] 1/5-shot reporting already done"
fi

# -- 2. tune 10- and 20-shot -------------------------------------------------
# The first attempt (preserved as results/sweep_hi_v1) swept 1e-5..1e-3 and had two
# problems: it OOMed at 20-shot after two baselines, and three surviving cells peaked
# at the top edge of that grid, making their "optimum" only a lower bound. Those 35
# measurements are still valid, so rather than redo them this sweeps only the missing
# region and merges everything.
#
#   10-shot: v1 covered 1e-5..1e-3 for all five baselines, so only extend upward.
#   20-shot: v1 covered only last_layer_ft and stage4_ft, so sweep the full grid.
#
# --grad-checkpoint is mandatory at 20-shot for baselines whose trainable parameters
# span every unit (BatchNorm affine, progressive unfreezing, full fine-tuning).
if [ ! -f results/sweep_hi/.done ]; then
    stage "tuning 10-shot (upper extension)"
    $PY experiments/sweep_lr.py \
        --k-shot 10 --episodes 75 --lrs 3e-3 1e-2 --inner-steps 20 \
        --grad-checkpoint --out results/sweep_hi10 \
        || fail "10-shot extension sweep failed"

    stage "tuning 20-shot (full grid)"
    $PY experiments/sweep_lr.py \
        --k-shot 20 --episodes 75 \
        --lrs 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 --inner-steps 20 \
        --grad-checkpoint --out results/sweep_hi20 \
        || fail "20-shot sweep failed (check for OOM)"

    mkdir -p results/sweep_hi && touch results/sweep_hi/.done
else
    echo "[skip] 10/20-shot sweeps already done"
fi

# -- 3. merge every sweep ----------------------------------------------------
stage "merging sweeps"
$PY experiments/merge_sweeps.py \
    results/sweep results/sweep_ext results/sweep_hi_v1 \n    results/sweep_hi10 results/sweep_hi20 \
    --out "$TUNED" || fail "merge failed"


# Verify the merged grid actually covers every cell, and that nothing is still pinned
# to a grid edge. An exit code of 0 from the sweeps is not sufficient evidence: a
# partially-populated tuning file is precisely what caused three baselines to be
# reported at an untuned default last time.
$PY - <<'EOF' || fail "tuning coverage incomplete -- see above"
import json, sys
best = json.load(open("results/best_lr_eurosat.json"))
want = {"last_layer_ft", "stage4_ft", "bn_only_ft", "progressive_unfreeze", "full_ft"}
missing = [(b, k) for b in sorted(want) for k in ("1", "5", "10", "20")
           if k not in best.get(b, {})]
if missing:
    print("missing tuned settings:", missing); sys.exit(1)
print("tuning coverage complete for all baselines at 1/5/10/20-shot")
EOF

# -- 4. report 10- and 20-shot ----------------------------------------------
if [ ! -f results/level0/.done_10_20 ]; then
    stage "reporting 10- and 20-shot, ${EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_baselines.py \
        --k-shot 10 20 --episodes "$EPISODES" --seeds $SEEDS \
        --tuned "$TUNED" --grad-checkpoint --skip-existing --out results/level0 \
        || fail "10/20-shot reporting failed"
    touch results/level0/.done_10_20
else
    echo "[skip] 10/20-shot reporting already done"
fi

stage "complete"
