#!/usr/bin/env bash
# Mechanism prediction 2, second attempt: regularise toward the PRETRAINED point.
#
#   ssh prashant 'cd ~/arc && setsid nohup bash scripts/run_vit_l2sp.sh \
#       > logs/vit_l2sp.log 2>&1 < /dev/null &'
#
# WHY THIS EXISTS. §3.9 ran the pre-registered regularisation prediction with ordinary
# weight decay and it FAILED -- full_ft got worse at every dose (-0.31 / -1.08 / -3.44
# / -5.12 across 1e-4 .. 1e-1), the opposite of predicted. Two readings survive that:
#
#   (a) the over-optimisation account is wrong, and full_ft loses for some other
#       reason entirely; or
#   (b) decay toward the ORIGIN is the wrong regulariser from a pretrained
#       initialisation. "Stay close to where you started" is the constraint the
#       account actually wants, and ||theta||^2 does not express it -- it charges
#       full price for the pretrained weights themselves, so the dose that would
#       constrain the adaptation also destroys the features being adapted.
#
# l2-SP separates them. It penalises ||theta - theta_0||^2, which is zero at the
# pretrained point, so it constrains the adaptation WITHOUT charging for pretraining.
#
# THE PREDICTION, on record before the data:
#
#   full_ft       should IMPROVE at some lambda > 0 -- this is reading (b)
#   attn_only_ft  should NOT improve, or improve less: 2.4M parameters, and §3.6
#                 showed it wants MORE optimisation, not less
#
# so attn - full_ft SHRINKS as lambda rises. That is the same signature §3.9 looked
# for and did not find; finding it here promotes §3.9 from "failed" to "the
# regulariser was wrong", and NOT finding it promotes §3.9 from "failed" to
# "decided against the account", which is the more interesting outcome and the reason
# this is worth two GPU-hours. EITHER RESULT IS REPORTABLE. There is no way for this
# run to be uninformative, which is not true of most of the queue.
#
# WHAT WOULD MAKE IT UNINFORMATIVE ANYWAY, and is guarded: a lambda grid that misses
# the transition. Adam normalises per parameter, so the penalty gradient
# lambda*(theta - theta_0) either sits far below the loss gradient and does nothing
# or far above it and freezes the model. The useful range is narrow and its location
# is not predictable from the parameter count -- the lr/size heuristic in §1 got the
# shallow end of the depth curve exactly backwards, and the lesson taken there was to
# stop guessing and sweep wide. So the grid spans FOUR ORDERS OF MAGNITUDE, and the
# log prints the drift ||theta - theta_0|| so a grid that missed can be seen to have
# missed rather than being read as a null.
#
# lambda=0 RUNS NO l2-SP CODE AT ALL (the penalty is behind `if self.inner_l2sp > 0`),
# so the lambda=0 column is a bit-exactness check on the episodic.py edit, not just a
# replication: it must land on results/vit_2b_eurosat within the ~0.6 floor
# (attn 85.96, full_ft 83.24 at seed 0). It is the first stage deliberately, so a
# broken edit costs one cell instead of six.
#
# Swept at each baseline's own tuned lr, not jointly -- same budget argument and same
# asymmetry as §3.9's driver: a null here is weaker than a positive, because a null
# could mean "l2-SP does not help" or "l2-SP needs a different lr".
#
# ~2.5 GPU-hours.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

# Wide on purpose. See the grid note above.
LAMS=${LAMS:-"0 1e-2 1e-1 1 10 100"}
TUNED="results/vit_best_lr_eurosat.json"
PAIR="full_ft attn_only_ft"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

[ -f "$TUNED" ] || fail "missing $TUNED -- the k=5 tuned lrs are what this sweeps around"

stage "preflight: baselines resolve, and l2-SP is actually plumbed"
for base in $PAIR; do
    $PY - "$base" <<'PYEOF' || fail "baseline not registered"
import sys
from arc.train.baselines import BASELINES
if sys.argv[1] not in BASELINES:
    print(f"!!! {sys.argv[1]!r} is not in BASELINES"); sys.exit(1)
print(f"[preflight] {sys.argv[1]} -> ok")
PYEOF
done
# A whole sweep against a build where the flag is silently ignored is the failure
# mode this guards. Checking the signature is not enough -- --inner-l2sp also has to
# reach the runner -- so the driver's argument parser is checked too.
$PY - <<'PYEOF' || fail "inner_l2sp is not wired through"
import inspect
from arc.train.episodic import EpisodeRunner
assert "inner_l2sp" in inspect.signature(EpisodeRunner.__init__).parameters, \
    "EpisodeRunner has no inner_l2sp"
import pathlib
src = pathlib.Path("experiments/run_baselines.py").read_text()
assert "--inner-l2sp" in src, "run_baselines.py has no --inner-l2sp flag"
assert "inner_l2sp=args.inner_l2sp" in src, \
    "the flag is parsed but never passed to EpisodeRunner"
assert '"inner_l2sp": runner.inner_l2sp' in src, \
    "the dose is not recorded in the result file, so the run cannot state its own dose"
print("[preflight] inner_l2sp: runner + flag + wiring + recording -> ok")
PYEOF
# And that it refuses to run both regularisers at once, since a sweep that quietly
# coupled l2-SP to leftover weight decay would measure neither (§3.9's confound).
$PY - <<'PYEOF' || fail "the mutual-exclusion guard is missing"
import inspect
from arc.train.episodic import EpisodeRunner
src = inspect.getsource(EpisodeRunner.__init__)
assert "pick one" in src, "inner_weight_decay and inner_l2sp can be set together"
print("[preflight] mutual exclusion -> ok")
PYEOF

for lam in $LAMS; do
    out="results/vit_l2sp_${lam}"
    if [ -f "$out/.done" ]; then
        echo "[skip] lambda=${lam} already done"
        continue
    fi
    stage "l2-SP lambda = ${lam}, 100 episodes, tuned lrs, 20 steps"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines $PAIR --tuned "$TUNED" \
        --inner-steps 20 --inner-l2sp "$lam" \
        $NOAMP --skip-existing --out "$out" \
        || fail "lambda=${lam} failed"
    mkdir -p "$out" && touch "$out/.done"

    # The lambda=0 column is the bit-exactness check on the episodic.py edit. Stop
    # here rather than burn five more cells on a build whose unregularised path moved.
    if [ "$lam" = "0" ]; then
        $PY - <<'PYEOF' || fail "lambda=0 did not reproduce the existing numbers"
import glob, json, statistics as st
ref = {"attn_only_ft": 85.96, "full_ft": 83.24}
for f in glob.glob("results/vit_l2sp_0/*.json"):
    j = json.load(open(f))
    got = st.mean([e["accuracy"] * 100 for e in j["episodes"]])
    want = ref.get(j["baseline"])
    if want is None:
        continue
    d = got - want
    print(f"[lambda=0] {j['baseline']:<14} {got:6.2f} vs {want:6.2f}  diff {d:+.2f}")
    assert abs(d) < 0.6, (
        f"{j['baseline']} moved {d:+.2f} from its committed value at lambda=0, where "
        f"NO l2-SP code runs. That means the episodic.py edit changed the "
        f"unregularised path and every number in the document is suspect. Do not "
        f"interpret the rest of this sweep."
    )
print("[lambda=0] within the ~0.6 floor -- the edit did not disturb the base path")
PYEOF
    fi
done

stage "done"
echo "Rows live in results/vit_l2sp_{0,1e-2,1e-1,1,10,100}."
echo
echo "Read PAIRED per-episode within each lambda, then the GAP across lambda."
echo "PREDICTED: full_ft improves at some lambda>0, attn does not (or less), so"
echo "attn - full_ft SHRINKS as lambda rises."
echo
echo "BEFORE reading a null: check that the grid caught the transition. If full_ft's"
echo "accuracy is flat at EVERY lambda including 100, the penalty never bit and the"
echo "grid is too low, not the account refuted. If accuracy collapses from 1e-2"
echo "onward, the grid is too high and the interesting range is below it. Either way"
echo "extend the grid before writing anything down -- §3.10's block-5 edge is the"
echo "standing reminder that a grid boundary flatters whatever you already believe."
