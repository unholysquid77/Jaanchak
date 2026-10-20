#!/usr/bin/env bash
# Mechanism prediction: does the attn-vs-full_ft crossover move with support-set size?
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_shots.sh \
#       > logs/vit_shots.log 2>&1 < /dev/null &'
#
# THE PREDICTION, on record before the measurement. §3.5 says full_ft loses on EuroSAT
# because 85.8M parameters taking 20 steps on 25 support images over-optimise where
# there is little to gain. §3.6 confirmed the first consequence: cutting the
# optimisation helps full_ft and hurts attn. The second consequence is about DATA, not
# steps -- if the problem is too many parameters for too few images, then
#
#   k=1  (5 support images):   full_ft should lose by MORE than at k=5
#   k=20 (100 support images): full_ft should close the gap, and may win
#
# At k=5 the gap is attn - full_ft = +2.50 ± 0.65 (three draws, §3.4). A monotone
# shrink from k=1 through k=20 is the prediction. If the gap is FLAT in k, the
# over-optimisation story loses its main remaining support and §3.5 has to be
# rewritten as a description rather than a mechanism -- which is a publishable
# negative, not a disaster, and §4 should record it either way.
#
# WHY BOTH ENDS. Only measuring k=20 would confound "more data helps full_ft" with
# "more data helps everything". Measuring k=1 as well gives the SLOPE of the gap in k,
# which is the quantity the mechanism actually predicts the sign of.
#
# TUNED SEPARATELY AT EVERY k, because the optimal lr depends on support-set size and
# reusing k=5's rates would hand the comparison to whichever baseline happens to suit
# them. One wide 6-point grid at each (baseline, k) cell per §4's standing rule --
# seven of my grids have been mispitched by assuming a window.
#
# MEMORY, and a prediction ARC_SPEC.md §1 got half right. That section sized the
# 10 GiB budget for ResNet-50 and said a 5-way 20-shot episode (175 images) "fits
# comfortably". It does -- for ResNet-50. ViT-B/16 with --no-amp and full_ft (every one
# of 85.8M parameters holding a gradient) does NOT: the first k=20 attempt died with
# CUDA OOM trying to allocate 462 MiB with 279 MiB free.
#
# The fix is --grad-checkpoint AT k=20 ONLY, which recomputes activations in the
# backward pass instead of storing them. Two things to be honest about:
#   * it is slower, which is why it is not applied at k=1 and k=5 where it is not needed
#   * recomputation can perturb floats, so k=20 runs a slightly different compute path
#     from the other two k values. That is well inside the ~0.6 between-run floor §4
#     records, but it means k=20 is not bit-comparable to k=1/k=5 and the write-up
#     should say so rather than present three points from one pipeline.
# The alternative -- enabling AMP for this cell -- was rejected: RESULTS_SUBSPACE.md §5
# found AMP was most of a confound on the ResNet arm, the whole ViT arm is --no-amp, and
# changing it for one cell would make that cell incomparable to everything else.
#
# ~6 GPU-hours (k=20 episodes are ~4x the forward cost of k=5, plus checkpointing).

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

SHOTS=${SHOTS:-"1 20"}
LRS=${LRS:-"1e-6 3e-6 1e-5 3e-5 1e-4 3e-4"}
PAIR="full_ft attn_only_ft"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

stage "preflight: baselines must resolve"
for base in $PAIR; do
    $PY - "$base" <<'PYEOF' || fail "baseline not registered"
import sys
from arc.train.baselines import BASELINES
if sys.argv[1] not in BASELINES:
    print(f"!!! {sys.argv[1]!r} is not in BASELINES"); sys.exit(1)
print(f"[preflight] {sys.argv[1]} -> ok")
PYEOF
done

run_k() {
    local k="$1"
    local sweep="results/vit_shots${k}_lr"
    local out="results/vit_shots${k}"

    if [ -f "$out/.done" ]; then
        echo "[skip] k=${k} already done"
        return
    fi

    # k=20 needs activation checkpointing to fit in 10 GiB; see the header.
    local ckpt=""
    if [ "$k" -ge 20 ]; then
        ckpt="--grad-checkpoint"
        echo "[mem] k=${k}: enabling --grad-checkpoint (175 images, no AMP, full_ft)"
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "k=${k}: tune $PAIR over $LRS"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$k" --episodes 50 \
            --baselines $PAIR --lrs $LRS \
            --inner-steps 20 $NOAMP $ckpt --out "$sweep" \
            || fail "k=${k} lr sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "k=${k}: merge and check for grid edges"
    $PY experiments/merge_sweeps.py "$sweep" --dataset eurosat \
        --out "results/vit_best_lr_shots${k}.json" \
        --edges-out "results/vit_edges_shots${k}.json" \
        || fail "k=${k} merge failed"

    $PY - "results/vit_edges_shots${k}.json" "$k" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! GRID EDGE at k={sys.argv[2]} -- these points are BOUNDS:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! The quantity here is a DIFFERENCE between two baselines, so an edge on")
    print("!!! EITHER row biases the gap. Extend before reading the slope in k.\n")
else:
    print(f"\nk={sys.argv[2]}: both baselines have interior optima.\n")
PYEOF

    stage "k=${k}: measure at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$k" --episodes 100 \
        --baselines $PAIR --tuned "results/vit_best_lr_shots${k}.json" \
        --inner-steps 20 $NOAMP $ckpt --skip-existing --out "$out" \
        || fail "k=${k} measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

# k=1 first: it is the cheaper end and the one where the mechanism predicts the
# LARGER effect, so a partial run still tests the prediction.
for k in $SHOTS; do
    run_k "$k"
done

stage "done"
echo "k=5 comparator, EuroSAT: attn 85.56, full_ft 83.07 (three-draw means, §3.4),"
echo "gap attn - full_ft = +2.50 +/- 0.65."
echo "Read k=1 and k=20 PAIRED per-episode within each k, then compare the three gaps."
echo "PREDICTED: gap shrinks monotonically from k=1 to k=20, possibly reversing at 20."
