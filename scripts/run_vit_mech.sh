#!/usr/bin/env bash
# Two things RESULTS_VIT.md §4 names as open, in one run.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_mech.sh \
#       > logs/vit_mech.log 2>&1 < /dev/null &'
#
# A: mlp_only_ft seed draws, to settle §3.3 (attention vs MLP).
# B: the inner-step test of the overfitting mechanism in §3.5.
#
# ~3 GPU-hours. Sentinel per stage.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"

ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# -- A: mlp seed draws --------------------------------------------------------
#
# §3.3 claimed attention and MLP are indistinguishable on both datasets. That was
# asserted with the quadrature test §3.4 showed to be wrong for paired data -- it
# understated the attn/full_ft margin by a factor of ~2.6. On EuroSAT the nominal
# attn - mlp gap is 1.20, which a paired test could resolve as real. The seed spread
# did not include mlp, so there is nothing to resolve it with.
#
# Same two seeds as scripts/run_vit_seeds.sh so the draws line up: the per-seed
# difference against attn is only meaningful if both were evaluated on the same
# episodes, which EpisodicSampler guarantees for a given seed.
run_mlp_seeds() {
    local tag="$1"
    local data="$2"
    local out="results/vit_seeds_${tag}"
    local tuned="results/vit_best_lr_${tag}.json"

    if [ -f "$out/.mlp_done" ]; then
        echo "[skip] $tag mlp seed draws already done"
        return
    fi
    [ -f "$tuned" ] || fail "$tag: $tuned missing -- run scripts/run_vit_2b.sh first"

    stage "$tag mlp_only_ft seeds 1 2"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $data --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 --seeds 1 2 \
        --baselines mlp_only_ft --tuned "$tuned" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "$tag mlp seed draws failed"
    touch "$out/.mlp_done"
}

run_mlp_seeds eurosat "$EURO_DATA"
run_mlp_seeds isic    "$ISIC_DATA"

# -- B: the inner-step test ---------------------------------------------------
#
# §3.5's mechanism candidate: full_ft opens 85.8M parameters and takes 20 steps on
# 25 support images, so where there is little headroom it overfits, and that is why
# attn_only_ft (28.3M) beats it by 2.50 on EuroSAT. Prediction: reduce the amount of
# optimisation and the gap should CLOSE, because the bigger group is the one being
# hurt.
#
# THE OBVIOUS VERSION OF THIS TEST IS CONFOUNDED, in two ways, and both are handled:
#
# 1. Fewer steps at a fixed learning rate means less total movement, so full_ft
#    getting worse would be uninterpretable -- less overfitting and less adaptation
#    predict the same sign. So the learning rate is RE-TUNED at each step count,
#    50-episode sweep, before anything is reported. The grid runs upward from the
#    20-step optimum because fewer steps want a larger one.
# 2. "full_ft improves" alone proves nothing: a shared improvement would just mean
#    20 steps was too many for everything. The quantity that tests the mechanism is
#    the attn - full_ft GAP, so attn_only_ft is re-tuned and re-measured at the same
#    step counts. The mechanism predicts the gap shrinks; it does not predict either
#    accuracy moves in a particular direction on its own.
#
# EuroSAT only. It is where the gap exists -- on ISIC full_ft already wins, so there
# is nothing for reduced optimisation to recover.
STEP_COUNTS="5 10"
FULL_LRS_HI="1e-5 3e-5 1e-4"
ATTN_LRS_HI="1e-4 3e-4 1e-3"

run_steps() {
    local steps="$1"
    local sweep="results/vit_steps${steps}_lr"
    local out="results/vit_steps${steps}"

    if [ -f "$out/.done" ]; then
        echo "[skip] ${steps}-step test already done"
        return
    fi

    if [ ! -f "$sweep/.done" ]; then
        stage "EuroSAT ${steps} inner steps: re-tune full_ft over $FULL_LRS_HI"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines full_ft --lrs $FULL_LRS_HI \
            --inner-steps "$steps" $NOAMP --out "$sweep" \
            || fail "${steps}-step full_ft sweep failed"
        stage "EuroSAT ${steps} inner steps: re-tune attn_only_ft over $ATTN_LRS_HI"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines attn_only_ft --lrs $ATTN_LRS_HI \
            --inner-steps "$steps" $NOAMP --out "${sweep}_attn" \
            || fail "${steps}-step attn sweep failed"
        mkdir -p "$sweep" && touch "$sweep/.done"
    fi

    stage "EuroSAT ${steps} inner steps: merge tuned settings"
    $PY experiments/merge_sweeps.py "$sweep" "${sweep}_attn" --dataset eurosat \
        --out "results/vit_best_lr_steps${steps}.json" \
        --edges-out "results/vit_edges_steps${steps}.json" \
        || fail "${steps}-step merge failed"

    # A grid-edge peak here would mean the re-tune did not find the optimum, and the
    # gap would then be measured at the wrong settings -- which is the whole thing
    # this stage exists to avoid. Report it loudly rather than quietly reporting a
    # number. Not fatal: the gap is still informative if BOTH groups are truncated
    # in the same direction, and the log says which.
    $PY - "results/vit_edges_steps${steps}.json" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print("\n!!! GRID EDGE at this step count -- the re-tune is a bound, not an optimum:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! extend that grid before quoting the gap as measured.\n")
else:
    print("\nboth groups have an interior optimum at this step count.\n")
PYEOF

    stage "EuroSAT ${steps} inner steps: full_ft vs attn at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines full_ft attn_only_ft \
        --tuned "results/vit_best_lr_steps${steps}.json" \
        --inner-steps "$steps" $NOAMP --skip-existing --out "$out" \
        || fail "${steps}-step measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

for n in $STEP_COUNTS; do
    run_steps "$n"
done

stage "ViT mechanism test complete"
echo "A: combine results/vit_seeds_{eurosat,isic} mlp rows with the attn rows for a"
echo "   PAIRED per-seed attn - mlp difference. The unpaired test is not valid here."
echo "B: the quantity that tests the mechanism is the attn - full_ft GAP at 5, 10 and"
echo "   20 steps (20 is results/vit_2b_eurosat, gap +2.72 at seed 0). Shrinking gap"
echo "   supports overfitting; a flat gap kills it."
