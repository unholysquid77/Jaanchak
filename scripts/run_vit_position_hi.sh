#!/usr/bin/env bash
# Extend the positional sweep's grid upward where it topped out.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_hi.sh \
#       > logs/vit_position_hi.log 2>&1 < /dev/null &'
#
# Run AFTER scripts/run_vit_position.sh finishes. It merges with that script's
# sweeps, so the low end is reused rather than recomputed.
#
# WHY. Block 8 peaked at 1e-3, the top of the grid, rising the whole way:
#   3e-05:80.29  1e-04:83.09  3e-04:86.72  1e-03:87.49
# and the 100-episode measurement at 1e-3 came out at 88.24, the highest number
# anywhere in this project. That is a BOUND, not an optimum.
#
# Block 11 is bracketed on the same grid -- it peaks interior at 3e-4 (86.56, with
# 1e-3 at 86.32) -- so the two are not directly comparable yet: one is a measurement
# and the other a lower bound. The comparison that matters (does a middle block beat
# the last one) therefore needs this extension before it can be quoted with a
# magnitude, even though its SIGN is already safe: block 8 is at least 88.24 against
# block 11's 86.51, which clears the ~0.6 noise floor whichever way the extension
# goes, because extending upward can only raise block 8.
#
# Seventh mispitched grid top, and the pattern is now unmistakable: every single one
# has been too low. The rule I have been using -- smaller fraction of backbone wants
# a larger step -- is right, and I keep under-applying it. A single mid-stack block
# at 2.75% wanted 1e-3 where the whole attention stack at 33% wanted 1e-4, a factor
# of 10 across a factor of 12 in parameters. That is a much steeper dependence than
# the capacity sweep showed within a suffix, where 3e-4 held from n=1 to n=4.
#
# Only blocks the first pass flagged get extended, read from its own edge reports --
# not a hardcoded list, so this cannot silently skip one or redo a bracketed point.
#
# ~1.5 GPU-hours if every block needs it.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

POS_LRS_HI="3e-3 1e-2"
POSITIONS=${POSITIONS:-"0 2 5 8"}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

needs_extension() {
    local i="$1"
    local edges="results/vit_edges_pos${i}.json"
    [ -f "$edges" ] || return 1
    $PY - "$edges" <<'PYEOF'
import json, sys
edges = json.loads(open(sys.argv[1]).read())
sys.exit(0 if any(e["side"] == "top" for e in edges) else 1)
PYEOF
}

run_hi() {
    local i="$1"
    local base="attn_block${i}_ft"
    local hi="results/vit_pos${i}_lr_hi"
    local out="results/vit_pos${i}_v2"

    if [ -f "$out/.done" ]; then
        echo "[skip] block ${i} extension already done"
        return
    fi
    if ! needs_extension "$i"; then
        echo "[skip] block ${i} peaked interior on the first grid -- nothing to extend"
        return
    fi

    if [ ! -f "$hi/.done" ]; then
        stage "block ${i}: extend upward over $POS_LRS_HI"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $POS_LRS_HI \
            --inner-steps 20 $NOAMP --out "$hi" \
            || fail "block ${i} high-lr sweep failed"
        mkdir -p "$hi" && touch "$hi/.done"
    fi

    stage "block ${i}: re-merge both grids"
    $PY experiments/merge_sweeps.py "results/vit_pos${i}_lr" "$hi" --dataset eurosat \
        --out "results/vit_best_lr_pos${i}.json" \
        --edges-out "results/vit_edges_pos${i}.json" \
        || fail "block ${i} re-merge failed"

    $PY - "results/vit_edges_pos${i}.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
top = [e for e in edges if e["side"] == "top"]
if top:
    print(f"\n!!! STILL AT A TOP EDGE at block {sys.argv[2]} after extending to 1e-2:")
    for e in top:
        print(f"    {e['baseline']} lr={e['lr']:.0e}")
    print("!!! Two extensions in one direction means the regime is not where I think")
    print("!!! it is. Do not quote a magnitude for this block; extend again.\n")
else:
    print(f"\nblock {sys.argv[2]} now has an interior optimum.\n")
PYEOF

    # Fresh output directory: --skip-existing would keep the row measured at the old
    # bound, which is the number this whole script exists to replace.
    stage "block ${i}: re-measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_pos${i}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "block ${i} re-measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

for i in $POSITIONS; do
    run_hi "$i"
done

stage "positional extension complete"
echo "Read results/vit_pos{i}_v2 where it exists, else results/vit_pos{i}."
echo "Block 11 is attn_last1_ft at 86.51 (results/vit_cap1), interior at 3e-4."
echo "full_ft is 83.24 (results/vit_2b_eurosat). All seed 0, single draw."
