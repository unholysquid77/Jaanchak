#!/usr/bin/env bash
# Full Level-0 pipeline, run sequentially on the compute box.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_level0.sh \
#       > logs/level0.log 2>&1 < /dev/null &'
#
# Ordered so the headline result lands first: 1- and 5-shot are already tuned, so
# their reporting run goes before the 10/20-shot tuning sweep. If the machine is
# needed for something else partway through, stages 1 and 2 alone are a complete
# table for the two shot counts that matter most.
#
# Every stage is skipped if its output already exists, so the script can be re-run
# after an interruption without repeating finished work.

set -uo pipefail

PY=.venv/bin/python
EPISODES=${EPISODES:-600}
SEEDS=${SEEDS:-"0 1 2"}
TUNED=results/best_lr_eurosat.json

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }

# -- 1. report 1- and 5-shot (already tuned) ---------------------------------
if [ ! -f results/level0/.done_1_5 ]; then
    stage "reporting 1- and 5-shot, ${EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_baselines.py \
        --k-shot 1 5 --episodes "$EPISODES" --seeds $SEEDS \
        --tuned "$TUNED" --out results/level0 \
    && mkdir -p results/level0 && touch results/level0/.done_1_5
else
    echo "[skip] 1/5-shot reporting already done"
fi

# -- 2. tune 10- and 20-shot -------------------------------------------------
# Narrower grid than the 1/5-shot sweep: the optima there landed between 1e-5 and
# 3e-3, and 20 inner steps won almost everywhere, so sweeping 5 steps again would
# mostly buy confirmation. 20-shot needs checkpointing to fit in 10 GiB.
if [ ! -f results/sweep_hi/.done ]; then
    stage "tuning 10- and 20-shot"
    $PY experiments/sweep_lr.py \
        --k-shot 10 20 --episodes 75 \
        --lrs 1e-5 3e-5 1e-4 3e-4 1e-3 --inner-steps 20 \
        --out results/sweep_hi \
    && touch results/sweep_hi/.done
else
    echo "[skip] 10/20-shot sweep already done"
fi

# -- 3. merge every sweep ----------------------------------------------------
stage "merging sweeps"
$PY experiments/merge_sweeps.py \
    results/sweep results/sweep_ext results/sweep_hi \
    --out "$TUNED"

# -- 4. report 10- and 20-shot ----------------------------------------------
if [ ! -f results/level0/.done_10_20 ]; then
    stage "reporting 10- and 20-shot, ${EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_baselines.py \
        --k-shot 10 20 --episodes "$EPISODES" --seeds $SEEDS \
        --tuned "$TUNED" --grad-checkpoint --out results/level0 \
    && touch results/level0/.done_10_20
else
    echo "[skip] 10/20-shot reporting already done"
fi

stage "complete"
