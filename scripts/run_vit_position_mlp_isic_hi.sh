#!/usr/bin/env bash
# Close top grid edges in the MLP positional sweep.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_mlp_hi.sh \
#       > logs/vit_position_mlp_hi.log 2>&1 < /dev/null &'
#
# MLP block 5 tuned to 3e-3, the top of its grid, so 85.91 is a LOWER BOUND. Its
# tuning curve is 0.8760 at 1e-3 and 0.8765 at 3e-3 -- essentially tied, so the
# optimum is probably at or just past 3e-3 rather than far beyond it. That is a
# reason to extend by two points and stop, not a reason to skip: "probably close"
# is what rule 4 in RESULTS_VIT.md exists to refuse.
#
# WHY IT MATTERS HERE SPECIFICALLY. The MLP sweep's whole job is the type x depth
# comparison, and block 5 is one of the depths where MLP appears to beat attention
# (>=85.91 against 82.04). An understated MLP row shrinks that gap, so this edge
# biases AGAINST the interaction I am describing -- the opposite direction from
# block 8's edge on the attention curve, which inflated its headline. Both get
# closed on the same rule regardless of which way they lean, because deciding which
# edges to chase by how they lean is how a result stops being a measurement.
#
# Reads the edge report rather than a hardcoded depth list, so it closes whatever
# the sweep actually flagged and refuses to re-tune a settled point.
#
# ~1 GPU-hour per depth.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

DEPTHS=${DEPTHS:-"0 2 5 8 11"}
HI_LRS=${HI_LRS:-"1e-2 3e-2"}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

extend_one() {
    local i="$1"
    local base="mlp_block${i}_ft"
    local sweep="results/vit_isicmlppos${i}_lr_hi"
    local out="results/vit_isicmlppos${i}_v2"
    local edges="results/vit_edges_isicmlppos${i}.json"

    if [ ! -f "$edges" ]; then
        echo "[skip] block ${i}: no edge report yet (depth not measured?)"
        return
    fi
    # Only a TOP edge is extendable upward. A bottom edge needs a different grid and
    # is reported rather than guessed at.
    if ! $PY - "$edges" <<'PYEOF' > /dev/null 2>&1
import json, sys
edges = json.loads(open(sys.argv[1]).read())
sys.exit(0 if any(e["side"] == "top" for e in edges) else 1)
PYEOF
    then
        $PY - "$edges" "$i" <<'PYEOF'
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    for e in edges:
        print(f"[note] block {sys.argv[2]}: {e['side']} edge at lr={e['lr']:.0e} -- "
              f"not extendable upward, needs its own grid")
else:
    print(f"[skip] block {sys.argv[2]}: interior optimum, nothing to extend")
PYEOF
        return
    fi

    if [ -f "$out/.done" ]; then
        echo "[skip] block ${i} already extended and re-measured"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "MLP block ${i}: extend upward over $HI_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $ISIC_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $HI_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "block ${i} extension sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "MLP block ${i}: merge original grid with extension"
    $PY experiments/merge_sweeps.py "results/vit_isicmlppos${i}_lr" "$sweep" --dataset isic \
        --out "results/vit_best_lr_isicmlppos${i}_v2.json" \
        --edges-out "results/vit_edges_isicmlppos${i}_v2.json" \
        || fail "block ${i} merge failed"

    $PY - "results/vit_edges_isicmlppos${i}_v2.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    for e in edges:
        print(f"\n!!! block {sys.argv[2]} STILL AT A {e['side'].upper()} EDGE: lr={e['lr']:.0e}")
    print("!!! Two extensions in one direction means the regime is not where I think")
    print("!!! it is. Do NOT extend a third time silently -- record that this depth's")
    print("!!! optimum lies outside any learning rate I would call reasonable, and say")
    print("!!! so in the write-up as a property of the result.\n")
else:
    print(f"\nblock {sys.argv[2]} now has an interior optimum.\n")
PYEOF

    stage "MLP block ${i}: re-measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $ISIC_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_isicmlppos${i}_v2.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "block ${i} re-measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

for i in $DEPTHS; do
    extend_one "$i"
done

stage "done"
echo "Where results/vit_isicmlppos{i}_v2 exists, read it INSTEAD of results/vit_isicmlppos{i}."
echo "Attention comparators, EuroSAT seed 0: block0 78.80, block2 78.87,"
echo "block5 82.04, block8 87.69, block11 86.51; full_ft 83.24, frozen 79.27."
