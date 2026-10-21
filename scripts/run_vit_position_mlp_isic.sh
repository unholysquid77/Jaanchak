#!/usr/bin/env bash
# Type x depth on ISIC: does the MIRROR hold in the MLP as well as in attention?
#
# What is established (docs/RESULTS_VIT.md):
#   * On EuroSAT, depth dominates parameter type, and the MLP curve reproduces the
#     attention curve's shape with an interior peak at block 8. So on EuroSAT the
#     depth effect is not about attention.
#   * On ISIC the ATTENTION curve is mirrored -- shallow wins, deep loses, four
#     shallow-vs-deep comparisons of opposite sign to EuroSAT's, three draws a side.
#   * The type x depth interaction has only ever been measured on EuroSAT. §4 says so.
#
# This closes that gap. Two outcomes, both worth having:
#   * MLP mirrors too  -> the mirror is a property of the DOMAIN SHIFT, not of a
#     parameter type, and the paper's interpretation (the curve locates where the
#     pretrained representation stops transferring) gains its second type.
#   * MLP does NOT mirror -> the interaction is real and dataset-specific, the
#     EuroSAT "it is depth, not attention" claim does not transfer to strong shift,
#     and §3.3 needs splitting by dataset.
#
# ISIC fractions are 0.1/0.4/0.5, NOT EuroSAT's 0.2/0.1/0.7 -- ISIC's smallest classes
# cannot spare a large training share and still leave 5-way episodes in the remainder.
# tests/test_results_provenance.py pins both pairs.
#
# Derived from run_vit_position_mlp.sh by substitution; the only intended differences
# are the dataset, the fractions, the output directory names, and the sweep ORDER
# (see POSITIONS).
#
# ~5 GPU-hours. Resumable: every cell has its own .done sentinel.
set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

# Order matters and is NOT the EuroSAT order this driver was derived from. On ISIC the
# attention curve is MIRRORED: the shallow half wins and the deep end loses. So the two
# depths that decide whether the mirror holds in MLP are 2 and 5 (ISIC attention's
# plateau) against 11 (its worst). Those three come first; 8 and 0 fill in the shape.
POSITIONS=${POSITIONS:-"5 11 2 8 0"}
POS_LRS=${POS_LRS:-"1e-5 3e-5 1e-4 3e-4 1e-3 3e-3"}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

for i in $POSITIONS; do
    case "$i" in
        0|2|5|8|11) ;;
        *) fail "position '$i' is not one of the five depths the attention sweep used (0 2 5 8 11)" ;;
    esac
done

run_pos() {
    local i="$1"
    local base="mlp_block${i}_ft"
    local sweep="results/vit_isicmlppos${i}_lr"
    local out="results/vit_isicmlppos${i}"

    if [ -f "$out/.done" ]; then
        echo "[skip] MLP block ${i} already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "MLP block ${i}: tune ${base} over $POS_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $ISIC_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $POS_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "MLP block ${i} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "MLP block ${i}: merge and check for a grid edge"
    $PY experiments/merge_sweeps.py "$sweep" --dataset isic \
        --out "results/vit_best_lr_isicmlppos${i}.json" \
        --edges-out "results/vit_edges_isicmlppos${i}.json" \
        || fail "MLP block ${i} merge failed"

    $PY - "results/vit_edges_isicmlppos${i}.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at MLP block {sys.argv[2]} -- this point is a BOUND:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! An undertuned MLP block at the SHALLOW end reads as 'the mirror does")
    print("!!! not hold in MLP', which is the hypothesis this sweep exists to test.")
    print("!!! Extend THAT depth before reading the curve.\n")
else:
    print(f"\nMLP block {sys.argv[2]} has an interior optimum.\n")
PYEOF

    stage "MLP block ${i}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $ISIC_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_isicmlppos${i}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "MLP block ${i} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# See the POSITIONS comment above for why this order differs from the EuroSAT driver's.
for i in $POSITIONS; do
    run_pos "$i"
done

stage "MLP positional sweep complete"
echo "Compare results/vit_isicmlppos{0,2,5,8,11} against, all EuroSAT seed 0:"
echo "  mlp_only_ft   84.76  (results/vit_2b_eurosat) -- the whole-type value"
echo "  full_ft       83.24  (results/vit_2b_eurosat)"
echo "  frozen_proto  79.27  (results/vit_2b_eurosat)"
echo "  attn_block{0,2,5,8} 78.80 / 78.87 / 82.04 / >=88.24, attn block 11 86.51"
echo
echo "Read it PAIRED and per-episode against the attention sweep at the SAME depth --"
echo "same seed means the same 100 episodes, so block-for-block the pairing is exact."
echo "Spread across depth is the quantity to compare: attention's was >=9.4 points."
