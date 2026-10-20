#!/usr/bin/env bash
# The reference row scripts/run_vit_shots.sh should have included.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_shots_ref.sh \
#       > logs/vit_shots_ref.log 2>&1 < /dev/null &'
#
# WHY THIS EXISTS, as a design miss worth recording. run_vit_shots.sh tests how the
# attn-vs-full_ft GAP moves with support-set size, and runs exactly those two
# baselines. At k=1 the gap came out +1.25 ± 1.18 against +2.72 at k=5 -- i.e. it
# SHRANK where the mechanism predicted it would grow. But with only those two rows
# that reading is not available, because two explanations are observationally
# identical:
#
#   (a) over-optimisation is not the story, and the gap genuinely shrinks with less
#       data -- a failed prediction, which is a real result
#   (b) at k=1 (five support images) NEITHER method adapts usefully, so there is no
#       gap to have, and the comparison is uninformative rather than negative
#
# frozen_proto distinguishes them: if attn and full_ft both sit near frozen at k=1,
# it is (b). A difference between two treatments cannot be interpreted without
# knowing whether either treatment did anything, and I built a driver that could not
# tell. The accuracies (62.15 and 60.89 at k=1 against 85.96 and 83.24 at k=5) make
# (b) plausible enough that it has to be ruled out.
#
# frozen_proto takes NO inner steps and NO tuning -- there is nothing to adapt -- so
# this is minutes, not hours, and k=5 is included as a free consistency check against
# the existing three-draw value of 79.22.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"
SHOTS=${SHOTS:-"1 5 20"}

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

for k in $SHOTS; do
    out="results/vit_shotsref${k}"
    if [ -f "$out/.done" ]; then
        echo "[skip] k=${k} reference already done"
        continue
    fi
    stage "frozen_proto reference, k=${k}, 100 episodes"
    # No --tuned: frozen_proto adapts nothing, so there is no learning rate to tune and
    # --inner-lr is inert for it. inner-steps is left at 20 to match the other rows'
    # invocation exactly, which for a zero-parameter selector changes nothing (and the
    # frac_mod column in the output proves it: it must read 0.000).
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$k" --episodes 100 \
        --baselines frozen_proto \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "k=${k} reference failed"
    mkdir -p "$out" && touch "$out/.done"
done

stage "done"
echo "CHECK FIRST: frac_mod must be 0.000 on every row. If it is not, frozen_proto is"
echo "adapting something and every 'frozen' number in the document is wrong."
echo "k=5 should reproduce 79.22 (three-draw mean) to within the ~0.6 floor."
echo
echo "Then read the gaps against it:"
echo "  k=1  attn 62.15  full_ft 60.89  frozen ?"
echo "  k=5  attn 85.96  full_ft 83.24  frozen 79.27 (seed 0)"
echo "If attn and full_ft both sit near frozen at k=1, the small k=1 gap is a floor"
echo "effect and says nothing about the mechanism. If they are both well above it,"
echo "the shrinking gap is a genuine failed prediction and section 3.5 must say so."
