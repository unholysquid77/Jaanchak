#!/usr/bin/env bash
# The capacity sweep: attention in a growing suffix of the block stack.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_capacity.sh \
#       > logs/vit_capacity.log 2>&1 < /dev/null &'
#
# WHAT THIS IS FOR. RESULTS_VIT.md §3.3.1 orders four parameter TYPES by size and finds
# an inverted U on EuroSAT peaking near a third of the backbone:
#
#   frozen 79.22 (0%)  ln_only 84.18 (0.045%)  attn 85.56 (33%)  mlp 84.38 (66%)  full_ft 83.07 (100%)
#
# §4 names the hole: those four differ in more than size, so the ordering cannot
# separate "how many parameters" from "which parameters". Attention and MLP are not
# interchangeable units of capacity, and neither is LayerNorm affine.
#
# This varies capacity inside ONE type. attn_last{1,2,3,4,6,8}_ft adapt attention in the
# last n of 12 blocks -- the same kind of parameter, nested (each n a strict superset of
# the last), exactly linear in n (2.36M per block, asserted in tests). attn_only_ft is
# the n=12 endpoint under the name it already has, so the curve ends on a measurement
# that already exists rather than on a fresh one.
#
# WHAT EACH OUTCOME WOULD MEAN. If accuracy rises to a peak below n=12 and falls, the
# inverted U is about capacity and §3.3.1's reading holds. If it rises monotonically to
# n=12, then attention is simply the right thing to adapt and the U in §3.3.1 is about
# WHICH parameters -- mlp and full_ft losing for reasons other than their size. Either
# way the answer is worth more than the current four points, and the second outcome
# would require rewriting §3.3.1 rather than extending it.
#
# A suffix, not a prefix: depth is the one ordering a block stack has, and a prefix
# adapts layers whose output every later layer re-processes, confounding capacity with
# how far a change propagates.
#
# ~4.5 GPU-hours (6 learning-rate sweeps at 50 episodes, then 6 measurements at 100).

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

CAPACITY_NS=${CAPACITY_NS:-"1 2 3 4 6 8"}

# Per-n learning rates have to be swept, not assumed. The fraction-of-backbone rule
# that has now caught me out six times says a smaller group wants a larger step, and
# n=1 is 2.8% of the backbone against attn_only_ft's 33% -- more than a decade apart in
# fraction. attn_only_ft peaked at 1e-4 and the inner-step work found its optimum did
# NOT move upward when the regime changed, so this grid brackets 1e-4 on both sides
# rather than running off in one direction.
CAP_LRS="3e-5 1e-4 3e-4 1e-3"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

run_one() {
    local n="$1"
    local base="attn_last${n}_ft"
    local sweep="results/vit_cap${n}_lr"
    local out="results/vit_cap${n}"

    if [ -f "$out/.done" ]; then
        echo "[skip] n=${n} already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "n=${n}: tune ${base} over $CAP_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $CAP_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "n=${n} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "n=${n}: merge and check for a grid edge"
    $PY experiments/merge_sweeps.py "$sweep" --dataset eurosat \
        --out "results/vit_best_lr_cap${n}.json" \
        --edges-out "results/vit_edges_cap${n}.json" \
        || fail "n=${n} merge failed"

    # An edge here is worse than usual: the sweep's whole claim is that each point is
    # at ITS optimum, so one bounded point bends the curve and the bend is the result.
    $PY - "results/vit_edges_cap${n}.json" "$n" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at n={sys.argv[2]} -- this point is a BOUND, not an optimum:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! extend before reading the curve. One bounded point bends it, and the")
    print("!!! bend is exactly what this sweep is looking for.\n")
else:
    print(f"\nn={sys.argv[2]} has an interior optimum.\n")
PYEOF

    stage "n=${n}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_cap${n}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "n=${n} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# Ascending, so a kill leaves a prefix of the curve rather than a scatter, and so the
# cheapest points land first.
for n in $CAPACITY_NS; do
    run_one "$n"
done

stage "capacity sweep complete"
echo "Curve: results/vit_cap{1,2,3,4,6,8} plus attn_only_ft from results/vit_2b_eurosat"
echo "as the n=12 endpoint (85.96 at seed 0), and frozen_proto 79.27 as n=0."
echo "Check frac_mod on every row: it must be n * 2.75% of the backbone."
echo "Single draw each. Any peak under a point tall is inside the noise floor in section 4."
