#!/usr/bin/env bash
# Four probes that decide whether the project continues as ARC or pivots.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_probe.sh \
#       > logs/probe.log 2>&1 < /dev/null &'
#
# Context. ARC's Level-1 gate is null on two datasets, and on ISIC bn_only_ft adapts
# 0.22% of ResNet-50 and ties full fine-tuning (49.44 vs 49.55). That suggests the
# binding constraint in few-shot adaptation is the *dimensionality* of the update
# rather than its location -- which the gate does not address, because it modulates
# the magnitude of a 23M-parameter update estimated from 25 images.
#
# BatchNorm affine is a per-channel scale and shift: a diagonal affine map on each
# unit's activations. So bn_only_ft is already an activation-subspace method. These
# probes test that reading and bound what a better basis could buy.
#
# Resumable via per-stage sentinels. ~2.5-3 GPU-hours total. Kill by PID.

set -uo pipefail

PY=.venv/bin/python
DATA="--dataset isic --fractions 0.1 0.4 0.5"
# 5-shot only, and 20 inner steps only. Both justified by the ISIC sweep rather than
# assumed: 1-shot spans 27.6-30.9 across every baseline, barely above the 20% chance
# floor, so it cannot separate methods; and 20 steps beat 5 in nearly every 5-shot
# cell. Spending the grid on configurations that cannot discriminate is the mistake
# the EuroSAT 10/20-shot sweep already paid for once.
SHOT=${SHOT:-5}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# -- P0a. Is the displacement of full fine-tuning diagonal and low-rank? -----
if [ ! -f results/subspace/.done_full ]; then
    stage "P0a displacement spectrum of full_ft (200 episodes)"
    $PY experiments/diagnose_subspace.py $DATA \
        --k-shot "$SHOT" --episodes 200 --baseline full_ft \
        --inner-lr 1e-4 --inner-steps 20 --out results/subspace \
        || fail "P0a failed"
    mkdir -p results/subspace && touch results/subspace/.done_full
else
    echo "[skip] P0a already done"
fi

# -- P0b. Does bn_only_ft move the representation the same way? --------------
# If the two displacements occupy the same subspace, bn_only_ft's success is because
# it spans the directions adaptation actually uses -- which both explains the ISIC
# tie and says a better basis has to beat the diagonal on its own terms. If they
# differ, BatchNorm is finding a *different* and cheaper route to the same accuracy,
# which is a more interesting result.
if [ ! -f results/subspace/.done_bn ]; then
    stage "P0b displacement spectrum of bn_only_ft (200 episodes)"
    $PY experiments/diagnose_subspace.py $DATA \
        --k-shot "$SHOT" --episodes 200 --baseline bn_only_ft \
        --inner-lr 3e-2 --inner-steps 20 --out results/subspace \
        || fail "P0b failed"
    touch results/subspace/.done_bn
else
    echo "[skip] P0b already done"
fi

# -- E2. Do the gate's signals predict realised per-unit benefit? ------------
# 2 + 2*17 = 36 adaptation runs per episode, so 30 episodes, not hundreds.
if [ ! -f results/unit_benefit/.done ]; then
    stage "E2 per-unit realised benefit vs signals (30 episodes, ~40 min)"
    $PY experiments/diagnose_unit_benefit.py $DATA \
        --k-shot "$SHOT" --episodes 30 --warmup 5 \
        --inner-lr 1e-4 --inner-steps 20 --out results/unit_benefit \
        || fail "E2 failed"
    mkdir -p results/unit_benefit && touch results/unit_benefit/.done
else
    echo "[skip] E2 already done"
fi

# -- E1. The parameter-type frontier ----------------------------------------
# Four new groups need their own learning rates; reporting them at a shared default
# would make them strawmen, which is the defect that made three EuroSAT baselines
# meaningless. Same 3e-6..3e-1 grid as the ISIC sweep, since bn_only_ft's optimum
# turned out to be at 3e-2 -- two decades above where the original grid stopped.
if [ ! -f results/isic_groups/.done ]; then
    stage "E1 tuning the parameter-type groups on val (32 configs, ~1.5 h)"
    $PY experiments/sweep_lr.py $DATA \
        --k-shot "$SHOT" --episodes 100 \
        --baselines bn_scale_ft bn_shift_ft conv_only_ft downsample_ft \
        --lrs 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
        --inner-steps 20 \
        --out results/isic_groups \
        || fail "E1 group sweep failed"
    touch results/isic_groups/.done
else
    echo "[skip] E1 already done"
fi

stage "probes complete"
cat <<'MSG'

Read in this order:
  results/subspace/subspace_isic_full_ft_5shot.json     diagonal fraction, effective rank
  results/subspace/subspace_isic_bn_only_ft_5shot.json  same, for the 0.22% baseline
  results/unit_benefit/unit_benefit_isic_5shot.json     signal-vs-benefit correlations
  results/isic_groups/best_lr_isic.json                 the parameter-type frontier

Decision points:
  high diagonal fraction + low effective rank -> a pre-extracted basis can carry
      adaptation, and BatchNorm-only is near the diagonal ceiling
  no signal correlates with realised benefit  -> the state vector is the problem,
      not the controller; ARC's Level 1 is a dead end rather than an untuned one
MSG
