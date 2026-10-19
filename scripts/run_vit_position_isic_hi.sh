#!/usr/bin/env bash
# Close the one open grid edge in the ISIC positional sweep.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_isic_hi.sh \
#       > logs/vit_position_isic_hi.log 2>&1 < /dev/null &'
#
# ISIC block 0 tuned to 3e-3, the TOP of its grid, so 48.83 is a lower bound. Every
# other ISIC depth has an interior optimum.
#
# WHY THIS CANNOT WAIT, and it is rule 4 in RESULTS_VIT.md's standing cautions: the
# edge points in the direction that flatters the hypothesis I am about to write up.
# The ISIC curve currently reads as a flat plateau over blocks 0/2/5 (48.83, 48.99,
# 49.23, no pairwise difference settled) and then a fall at 8 and 11. My stated
# hypothesis is that ISIC's useful adaptation lives EARLY because dermoscopy diverges
# from ImageNet-21k early. If block 0 is undertuned, extending it can only raise it,
# which would convert the plateau into a shallow-peaked rise -- exactly the shape the
# hypothesis predicts. Writing the plateau up now and extending later would mean
# publishing the weaker claim and then "discovering" the stronger one from an edge I
# already knew about. Block 8 on EuroSAT cost half a point and a third of two margins
# to the same mistake.
#
# Note the inversion that makes this interesting rather than routine: on EuroSAT
# block 0 pinned at the BOTTOM of its grid (3e-5, wants almost no movement); on ISIC
# it pins at the TOP (3e-3, wants aggressive movement). Same selector, same depth,
# opposite ends of the same grid. That is itself evidence about where each shift
# bites, and it is worth stating in the write-up whichever way the extension lands.
#
# ~1 GPU-hour.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

HI_LRS=${HI_LRS:-"1e-2 3e-2"}
BASE="attn_block0_ft"
SWEEP="results/vit_isicpos0_lr_hi"
OUT="results/vit_isicpos0_v2"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# Refuse to run if the edge this script exists to close is not actually there --
# otherwise a stale copy of this script silently re-tunes a settled point.
$PY - results/vit_edges_isicpos0.json <<'PYEOF' || fail "ISIC block 0 has no top grid edge; nothing to extend"
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if not any(e["side"] == "top" for e in edges):
    print("!!! no TOP edge recorded for ISIC block 0")
    sys.exit(1)
print("[check] ISIC block 0 top edge confirmed; extending")
PYEOF

if [ ! -f "$SWEEP/.done" ]; then
    stage "ISIC block 0: extend upward over $HI_LRS"
    # shellcheck disable=SC2086
    $PY experiments/sweep_lr.py $ISIC_DATA --arch "$ARCH" \
        --k-shot "$SHOT" --episodes 50 \
        --baselines "$BASE" --lrs $HI_LRS \
        --inner-steps 20 $NOAMP --out "$SWEEP" \
        || fail "extension sweep failed"
    mkdir -p "$SWEEP" && touch "$SWEEP/.done"
fi

stage "merge the original grid with the extension"
$PY experiments/merge_sweeps.py results/vit_isicpos0_lr "$SWEEP" --dataset isic \
    --out results/vit_best_lr_isicpos0_v2.json \
    --edges-out results/vit_edges_isicpos0_v2.json \
    || fail "merge failed"

$PY - results/vit_edges_isicpos0_v2.json <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    for e in edges:
        print(f"\n!!! STILL AT A GRID EDGE: lr={e['lr']:.0e} ({e['side']})")
    print("!!! Two extensions in one direction means the regime is not where I think")
    print("!!! it is. Do NOT extend a third time silently -- record that the optimum")
    print("!!! for this depth is outside any learning rate I would call reasonable,")
    print("!!! and say so in the write-up as a property of the result.\n")
else:
    print("\nISIC block 0 now has an interior optimum.\n")
PYEOF

stage "re-measure at 100 episodes"
# shellcheck disable=SC2086
$PY experiments/run_baselines.py $ISIC_DATA --arch "$ARCH" \
    --split val --k-shot "$SHOT" --episodes 100 \
    --baselines "$BASE" --tuned results/vit_best_lr_isicpos0_v2.json \
    --inner-steps 20 $NOAMP --skip-existing --out "$OUT" \
    || fail "measurement failed"
mkdir -p "$OUT" && touch "$OUT/.done"

stage "done"
echo "Read results/vit_isicpos0_v2 in place of results/vit_isicpos0."
echo "ISIC depths for comparison, seed 0: block2 48.99, block5 49.23,"
echo "block8 47.23, block11 45.00; frozen 39.35, full_ft 52.49, attn_all 50.20."
