#!/usr/bin/env bash
# Mechanism prediction 2: does regularising full_ft's parameters close the gap?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_wd.sh \
#       > logs/vit_wd.log 2>&1 < /dev/null &'
#
# THE PREDICTION, on record before the data. §3.5: full_ft loses on EuroSAT because
# 85.8M parameters taking 20 steps on 25 support images over-optimise. §3.6 confirmed
# the steps consequence (less optimisation helps full_ft, hurts attn). This tests the
# REGULARISATION consequence:
#
#   full_ft       should IMPROVE at some weight decay > 0
#   attn_only_ft  should NOT improve, or improve less -- it is 2.4M parameters and
#                 §3.6 showed it wants MORE optimisation, not less
#
# A gap that closes under regularisation is the mechanism's signature. If NEITHER
# baseline moves, the over-parameterisation story is in trouble and §4 records it; if
# BOTH improve equally, weight decay is just a better optimiser setting here and says
# nothing about capacity.
#
# WHY wd IS SWEPT AT THE TUNED lr RATHER THAN JOINTLY. lr and weight decay interact,
# so the clean design is a 2-D grid -- which is 6x4 cells per baseline and does not
# fit the budget. Sweeping wd at each baseline's own tuned lr answers the directional
# question (does ANY regularisation beat the unregularised best?) and that is what the
# prediction is about. The limitation is real and belongs in the write-up: a null here
# is weaker than a positive, because a null could mean "wd does not help" or "wd needs
# a different lr". A POSITIVE result does not have that escape hatch, which is the
# asymmetry §3.3.6's pre-registration note makes use of too.
#
# wd=0 is bit-identical to the existing runs (tests/test_weight_decay.py asserts the
# single-group collapse), so the 0 column is a free replication of §3.2's numbers and
# is included deliberately as a noise-floor check.
#
# ~2 GPU-hours.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

WDS=${WDS:-"0 1e-4 1e-3 1e-2 1e-1"}
TUNED="results/vit_best_lr_eurosat.json"
PAIR="full_ft attn_only_ft"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

[ -f "$TUNED" ] || fail "missing $TUNED -- the k=5 tuned lrs are the baseline this sweeps around"

stage "preflight: baselines resolve, and weight decay is actually plumbed"
for base in $PAIR; do
    $PY - "$base" <<'PYEOF' || fail "baseline not registered"
import sys
from arc.train.baselines import BASELINES
if sys.argv[1] not in BASELINES:
    print(f"!!! {sys.argv[1]!r} is not in BASELINES"); sys.exit(1)
print(f"[preflight] {sys.argv[1]} -> ok")
PYEOF
done
# Guard against running a whole sweep against a build where the flag is ignored.
$PY - <<'PYEOF' || fail "inner_weight_decay is not wired into EpisodeRunner"
import inspect
from arc.train.episodic import EpisodeRunner
sig = inspect.signature(EpisodeRunner.__init__)
assert "inner_weight_decay" in sig.parameters, "EpisodeRunner has no inner_weight_decay"
print("[preflight] EpisodeRunner.inner_weight_decay -> ok")
PYEOF

for wd in $WDS; do
    out="results/vit_wd_${wd}"
    if [ -f "$out/.done" ]; then
        echo "[skip] wd=${wd} already done"
        continue
    fi
    stage "weight decay = ${wd}, 100 episodes, tuned lrs, 20 steps"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines $PAIR --tuned "$TUNED" \
        --inner-steps 20 --inner-weight-decay "$wd" \
        $NOAMP --skip-existing --out "$out" \
        || fail "wd=${wd} failed"
    mkdir -p "$out" && touch "$out/.done"
done

stage "done"
echo "Rows live in results/vit_wd_{0,1e-4,1e-3,1e-2,1e-1}."
echo "wd=0 should reproduce results/vit_2b_eurosat to within the ~0.6 floor"
echo "(attn 85.96, full_ft 83.24 at seed 0) -- if it does not, STOP: the param-group"
echo "change altered the wd=0 path and every number in the document is suspect."
echo
echo "Read PAIRED per-episode within each wd, then compare the gap across wd."
echo "PREDICTED: full_ft improves at some wd>0, attn does not (or improves less),"
echo "so attn - full_ft SHRINKS as wd rises."
