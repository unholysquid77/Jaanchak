#!/usr/bin/env bash
# The EuroSAT arm: does the structural claim hold on a second, much easier domain?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_eurosat_probe.sh \
#       > logs/eurosat_probe.log 2>&1 < /dev/null &'
#
# On ISIC the findings were: displacement ~90% per-channel affine, near rank-1 at the
# top, units redundant, every parameter group plateauing at ~49.5, and a diagonal basis
# (ssf, 48.61) beating an energy-ordered non-diagonal one at any rank (rank4 45.13,
# rank16 44.79). A structural claim resting on one dataset is a single-dataset claim,
# and EuroSAT is the useful contrast: a frozen backbone already reaches 84.6% there at
# 5-shot, so the domain shift is mild and the headroom is 3.1 points against ISIC's
# 11.5. If the same structure appears under a much weaker shift, it is a property of
# few-shot adaptation rather than of medical imaging.
#
# Fractions are 0.2/0.1/0.7, matching the existing Level-0 EuroSAT table rather than
# ISIC's 0.1/0.4/0.5 -- EuroSAT is ten balanced classes of 2,700 images, so it has none
# of the thin-class problem that forced ISIC's split, and 15 queries are sampleable on
# the train split too. Deliberately NOT reusing ISIC's numbers as a default anywhere.
#
# ~5 GPU-hours. Resumable by sentinel. Kill by PID.

set -uo pipefail

PY=.venv/bin/python
DATA="--dataset eurosat --fractions 0.2 0.1 0.7"
SHOT=${SHOT:-5}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# -- 1. displacement structure ----------------------------------------------
if [ ! -f results/subspace_eurosat/.done ]; then
    stage "displacement spectrum of full_ft on EuroSAT (200 episodes)"
    $PY experiments/diagnose_subspace.py $DATA \
        --k-shot "$SHOT" --episodes 200 --baseline full_ft \
        --inner-lr 1e-5 --inner-steps 20 --out results/subspace_eurosat \
        || fail "EuroSAT displacement probe failed"
    mkdir -p results/subspace_eurosat && touch results/subspace_eurosat/.done
else
    echo "[skip] displacement probe already done"
fi

# -- 2. the diagonal vs non-diagonal frontier -------------------------------
# inner-lr 1e-5 is full_ft's tuned EuroSAT 5-shot setting from
# docs/EXPERIMENT_LOG.md section 2, not ISIC's 1e-4. Extracting a displacement with
# the wrong learning rate would measure a different adaptation than the one the
# Level-0 table reports.
if [ ! -f results/frontier_eurosat/.done ]; then
    stage "diagonal vs non-diagonal frontier on EuroSAT"
    $PY experiments/subspace_frontier.py $DATA \
        --k-shot "$SHOT" --n-query 15 --eval-n-query 15 \
        --extract-episodes 100 --eval-episodes 200 \
        --inner-lr 1e-5 --inner-steps 20 \
        --ranks 1 2 4 8 16 --fit-lrs 0.01 0.05 0.2 \
        --out results/frontier_eurosat \
        || fail "EuroSAT frontier failed"
    mkdir -p results/frontier_eurosat && touch results/frontier_eurosat/.done
else
    echo "[skip] frontier already done"
fi

# -- 3. the parameter-type plateau ------------------------------------------
# Does every group plateau on EuroSAT too? The existing Level-0 table already has
# frozen_proto, last_layer_ft, bn_only_ft, stage4_ft, progressive_unfreeze and full_ft
# at 5-shot; this adds the four parameter-type groups on the same split.
if [ ! -f results/eurosat_groups/.done ]; then
    stage "parameter-type groups on EuroSAT val"
    $PY experiments/sweep_lr.py $DATA \
        --k-shot "$SHOT" --episodes 100 \
        --baselines bn_scale_ft bn_shift_ft conv_only_ft downsample_ft \
        --lrs 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
        --inner-steps 20 --out results/eurosat_groups \
        || fail "EuroSAT group sweep failed"
    touch results/eurosat_groups/.done
else
    echo "[skip] group sweep already done"
fi

stage "EuroSAT arm complete"
