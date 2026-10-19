#!/usr/bin/env bash
# Does the block have to be the LAST one?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position.sh \
#       > logs/vit_position.log 2>&1 < /dev/null &'
#
# §3.3.2 swept attention as a growing SUFFIX and found a flat line from one block
# upward: n=1 scores 86.51 against full_ft's 83.24, and n=2..6 add nothing. §3.3.3
# turns that into "attention is the active ingredient and one block of it is enough".
#
# But every point in that sweep contained block 11. What was actually established is
# that THE LAST block is enough, and §4 says so. This separates the two claims. Same
# parameter type, same 2.75% of the backbone, same count at every depth (asserted in
# tests), one block, five positions:
#
#   block 0, 2, 5, 8   -- this sweep
#   block 11           -- already measured, it IS attn_last1_ft at 86.51
#
# WHAT THE OUTCOMES MEAN. If block 0 does as well as block 11, position is irrelevant
# and §3.3.3's claim is about attention per se -- the stronger and more surprising
# result, since it would mean one 2.4M-parameter slice anywhere in the stack beats
# fine-tuning all 85.8M. If accuracy climbs with depth, the claim must be narrowed to
# the last block, and the shape of the climb is then itself the finding: it would say
# adaptation has to happen where the representation is already task-specific.
#
# Each position is tuned separately. The suffix sweep found 3e-4 for n<=4 and 1e-4 for
# n>=6, so one block wants 3e-4 near the top of the stack -- but there is no reason to
# assume an early block shares that, and assuming it is how five of my grids went
# wrong. Same 4-point grid as the capacity sweep so the two are comparable.
#
# ~3 GPU-hours.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

POSITIONS=${POSITIONS:-"0 2 5 8"}
POS_LRS="3e-5 1e-4 3e-4 1e-3"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

run_pos() {
    local i="$1"
    local base="attn_block${i}_ft"
    local sweep="results/vit_pos${i}_lr"
    local out="results/vit_pos${i}"

    if [ -f "$out/.done" ]; then
        echo "[skip] block ${i} already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "block ${i}: tune ${base} over $POS_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $POS_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "block ${i} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "block ${i}: merge and check for a grid edge"
    $PY experiments/merge_sweeps.py "$sweep" --dataset eurosat \
        --out "results/vit_best_lr_pos${i}.json" \
        --edges-out "results/vit_edges_pos${i}.json" \
        || fail "block ${i} merge failed"

    $PY - "results/vit_edges_pos${i}.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at block {sys.argv[2]} -- this point is a BOUND:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! An undertuned early block would look like a depth effect, which is")
    print("!!! exactly the conclusion this sweep is testing. Extend before reading.\n")
else:
    print(f"\nblock {sys.argv[2]} has an interior optimum.\n")
PYEOF

    stage "block ${i}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_pos${i}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "block ${i} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# Deepest first. If only part of this finishes, block 8 next to block 11's 86.51 is
# the most informative single comparison, and block 0 is the most likely to differ,
# so running 8 then 5 then 2 then 0 walks from "probably the same" toward "probably
# not" rather than leaving a hole in the middle.
for i in 8 5 2 0; do
    run_pos "$i"
done

stage "positional sweep complete"
echo "Compare results/vit_pos{0,2,5,8} against attn_last1_ft (block 11) at 86.51,"
echo "and against full_ft 83.24 -- both seed 0, results/vit_cap1 and vit_2b_eurosat."
echo "Single draw each: a spread under ~0.6 is one flat line, not a depth trend."
