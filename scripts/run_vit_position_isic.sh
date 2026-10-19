#!/usr/bin/env bash
# Is the depth effect a property of the backbone, or of mild shift?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_isic.sh \
#       > logs/vit_position_isic.log 2>&1 < /dev/null &'
#
# RESULTS_VIT.md §3.3.2 is the largest effect measured on this arm: five interventions
# identical in size and parameter type (attention in one block, 2.75% of the backbone)
# span >=9.4 points on EuroSAT, against 6.5 for the whole parameter-TYPE axis and 1.18
# for attn-vs-mlp inside it. Block 8 alone beats full fine-tuning by 5.00 +/- 1.03.
#
# All of it is EuroSAT, where full_ft LOSES and headroom is 6.34. ISIC is the other
# regime: full_ft wins there by 2.20 +/- 0.30 and headroom is 13.24. §4 calls this the
# single most informative unrun experiment on the arm, and this is it.
#
# WHAT THE OUTCOMES MEAN, and both are publishable:
#
#   Same shape on ISIC  -> depth is a property of the backbone and the task's useful
#                          adaptation simply lives at a particular height. The claim
#                          generalises across shift magnitude, and §3.5's headroom
#                          story has to be re-derived with a positional partition on
#                          both sides rather than a type partition.
#   Flat on ISIC        -> the depth effect is a MILD-SHIFT phenomenon. Under strong
#                          shift every depth is useful, which is the same shape as
#                          §3.5's existing claim about joint type movement and would
#                          make the two readings one mechanism rather than two.
#   Different peak      -> the peak's position tracks shift magnitude, which is a
#                          sharper statement than either of the above and makes the
#                          depth curve a measurement OF the shift.
#
# Note the asymmetry with the EuroSAT sweep: block 11 was free there because it is
# attn_last1_ft from the capacity sweep. On ISIC nothing positional has been measured,
# so all five depths are run here.
#
# GRID. The EuroSAT sweep taught one thing that overrides the fraction-of-backbone
# heuristic: a single EARLY block wants a LOW learning rate (block 0 tuned to 3e-5 and
# still sat at its bottom edge) while a single LATE block wants a HIGH one (block 8
# tuned to 1e-3 and sat at its TOP edge). Size predicts the late end and gets the
# early end backwards. So every depth gets the same wide 6-point grid rather than a
# 4-point window centred on a guess -- that is how five of my grids went wrong, and
# here a mispitched early block would look exactly like the depth effect being tested.
# ISIC's attn_only_ft tuned to 3e-4 against EuroSAT's 1e-4, so the window is shifted
# one point up relative to the EuroSAT sweep's.
#
# ~5 GPU-hours.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

POSITIONS=${POSITIONS:-"8 5 11 2 0"}
POS_LRS=${POS_LRS:-"1e-5 3e-5 1e-4 3e-4 1e-3 3e-3"}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# Guard the baseline list the same way run_vit_2b.sh does after the GROUPS incident:
# the failure mode that cost four hours was a plausible WRONG list, not an empty one.
for i in $POSITIONS; do
    case "$i" in
        0|2|5|8|11) ;;
        *) fail "position '$i' is not one of the five swept depths (0 2 5 8 11)" ;;
    esac
done

run_pos() {
    local i="$1"
    local base="attn_block${i}_ft"
    local sweep="results/vit_isicpos${i}_lr"
    local out="results/vit_isicpos${i}"

    if [ -f "$out/.done" ]; then
        echo "[skip] ISIC block ${i} already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "ISIC block ${i}: tune ${base} over $POS_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $ISIC_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $POS_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "ISIC block ${i} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "ISIC block ${i}: merge and check for a grid edge"
    $PY experiments/merge_sweeps.py "$sweep" --dataset isic \
        --out "results/vit_best_lr_isicpos${i}.json" \
        --edges-out "results/vit_edges_isicpos${i}.json" \
        || fail "ISIC block ${i} merge failed"

    $PY - "results/vit_edges_isicpos${i}.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at ISIC block {sys.argv[2]} -- this point is a BOUND:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! Six points wide and still at an edge means the regime is not where")
    print("!!! this grid assumes. Extend THAT depth before reading the curve: an")
    print("!!! undertuned block is indistinguishable from a real depth effect, and")
    print("!!! the depth effect is the whole question.\n")
else:
    print(f"\nISIC block {sys.argv[2]} has an interior optimum.\n")
PYEOF

    stage "ISIC block ${i}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $ISIC_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_isicpos${i}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "ISIC block ${i} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# Order: 8 first because it is EuroSAT's peak and the single most informative point
# (does the peak transfer?); then 5 and 11 to bracket it; then 2 and 0, the end that
# was indistinguishable from frozen on EuroSAT. A partial run this way still answers
# the headline question.
for i in $POSITIONS; do
    run_pos "$i"
done

stage "ISIC positional sweep complete"
echo "Compare results/vit_isicpos{0,2,5,8,11} against ISIC seed 0 in"
echo "results/vit_2b_isic: frozen_proto 38.26-ish, full_ft the winner there."
echo
echo "Read it PAIRED and per-episode (same seed = same 100 episodes), and remember"
echo "the ~0.6 between-run floor: a single-draw gap under ~1 point is not a result."
echo "The EuroSAT spread to beat is >=9.4 points across the same five depths."
