#!/usr/bin/env bash
# Is the depth effect about ATTENTION, or just about DEPTH?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_mlp.sh \
#       > logs/vit_position_mlp.log 2>&1 < /dev/null &'
#
# RESULTS_VIT.md §3.3.2 holds attention's size and type fixed and varies depth: >=9.4
# points across five blocks, with block 8 alone beating full fine-tuning by 5.00. §4
# records that nothing sweeps depth within any OTHER parameter type, so two readings
# of that result both survive:
#
#   "about attention"  -- attention is the active ingredient and depth governs where
#                         it can act. Predicts MLP stays near its whole-type value
#                         (84.76 at seed 0) at every depth, with no comparable spread,
#                         and no single MLP block beating full_ft by anything like 5.
#   "about depth"      -- three-quarters height is where this task's useful adaptation
#                         lives, whatever type sits there. Predicts the same shape:
#                         block 0 near frozen, a peak around 8, and that peak beating
#                         full_ft.
#
# The predictions differ in sign and in size, so one sweep separates them. This is the
# experiment that decides which of the two §3.3 is allowed to claim.
#
# WHAT IS AND IS NOT HELD FIXED. Depth is exact and type varies, with size riding
# along: an MLP block is 5.50% of the backbone against attention's 2.75% (asserted in
# tests/test_selectors_vit.py). That is the OPPOSITE confound from the attention
# sweep's, and it is why the pair brackets the question rather than either doing it
# alone -- if MLP is flat across depth DESPITE being the larger intervention, size is
# not driving either curve. If MLP instead shows the same shape, the larger size makes
# that the conservative direction too. Neither outcome is confounded in the direction
# that would flatter the existing claim.
#
# GRID. Same 6-point grid at every depth, for the reason the attention sweep paid for:
# a single early block wanted a LOW lr (block 0 tuned to 3e-5, bottom edge) while a
# single late block wanted a HIGH one (block 8 tuned to 1e-3, TOP edge). Size predicts
# the late end and gets the early end backwards, so no window centred on a guess. An
# undertuned MLP block would read as "MLP is flat across depth", which is one of the
# two hypotheses being tested -- the single most dangerous artefact available here.
# mlp_only_ft (66% of the backbone) tuned to 1e-4 on EuroSAT; one block is 12x smaller.
#
# ~5 GPU-hours.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

POSITIONS=${POSITIONS:-"8 11 0 5 2"}
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
    local sweep="results/vit_mlppos${i}_lr"
    local out="results/vit_mlppos${i}"

    if [ -f "$out/.done" ]; then
        echo "[skip] MLP block ${i} already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "MLP block ${i}: tune ${base} over $POS_LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines "$base" --lrs $POS_LRS \
            --inner-steps 20 $NOAMP --out "$sweep" \
            || fail "MLP block ${i} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "MLP block ${i}: merge and check for a grid edge"
    $PY experiments/merge_sweeps.py "$sweep" --dataset eurosat \
        --out "results/vit_best_lr_mlppos${i}.json" \
        --edges-out "results/vit_edges_mlppos${i}.json" \
        || fail "MLP block ${i} merge failed"

    $PY - "results/vit_edges_mlppos${i}.json" "$i" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at MLP block {sys.argv[2]} -- this point is a BOUND:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! An undertuned MLP block reads as 'MLP is flat across depth', which is")
    print("!!! one of the two hypotheses this sweep exists to separate. Extend THAT")
    print("!!! depth before reading the curve.\n")
else:
    print(f"\nMLP block {sys.argv[2]} has an interior optimum.\n")
PYEOF

    stage "MLP block ${i}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines "$base" --tuned "results/vit_best_lr_mlppos${i}.json" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "MLP block ${i} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# Order: 8 and 11 first, the two depths where attention was strongest -- if MLP is
# going to show the effect at all it shows there, and those two points alone decide
# between the hypotheses. Then 0, the depth where attention did nothing, which is the
# other end of the discriminating comparison. 5 and 2 fill in the shape.
for i in $POSITIONS; do
    run_pos "$i"
done

stage "MLP positional sweep complete"
echo "Compare results/vit_mlppos{0,2,5,8,11} against, all EuroSAT seed 0:"
echo "  mlp_only_ft   84.76  (results/vit_2b_eurosat) -- the whole-type value"
echo "  full_ft       83.24  (results/vit_2b_eurosat)"
echo "  frozen_proto  79.27  (results/vit_2b_eurosat)"
echo "  attn_block{0,2,5,8} 78.80 / 78.87 / 82.04 / >=88.24, attn block 11 86.51"
echo
echo "Read it PAIRED and per-episode against the attention sweep at the SAME depth --"
echo "same seed means the same 100 episodes, so block-for-block the pairing is exact."
echo "Spread across depth is the quantity to compare: attention's was >=9.4 points."
