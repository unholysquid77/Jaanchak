#!/usr/bin/env bash
# The ViT arm. Stage 1 only: learning rates.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit.sh \
#       > logs/vit.log 2>&1 < /dev/null &'
#
# Everything measured so far is on ResNet-50. The headline finding -- every
# parameter group plateaus within 0.75 points across a 778x range of trainable
# parameters, on both ISIC and EuroSAT -- is a claim about few-shot adaptation that
# currently rests on one architecture. Four of five comparator papers are ViT and
# LN-Tune is the direct analogue of bn_only_ft, so this is where those numbers
# become comparable rather than asserted.
#
# ViT-B/16 augreg_in21k, because that is what SSF, LN-Tune and DIPA report. It fits:
# measured 4.50 GiB peak at 20-shot with gradient checkpointing, 1.63 GiB at 5-shot.
#
# WHY LEARNING RATES FIRST, ALONE. The smoke run put both full_ft and ln_only_ft at
# the top of a single-point grid, i.e. the ResNet settings (1e-4 full_ft on ISIC,
# 1e-5 on EuroSAT) are certainly wrong here and possibly wrong by orders of
# magnitude -- ViTs and ResNets do not share an optimiser regime, and on ISIC the
# BatchNorm groups wanted 3e-2 while full_ft wanted 1e-4, a 300x spread *within*
# one architecture. Running the displacement probe or the frontier before this is
# settled would measure a badly-tuned adaptation and the result would be garbage
# that looks like a finding.
#
# Stage 1 sweeps ONLY full_ft and ln_only_ft. Those two bracket the headroom
# (everything else in the parameter-type partition has to land between a frozen
# backbone and full fine-tuning), so they are what the later stages need in order
# to pick sensible grids. Sweeping all seven groups now would be 5x the GPU time to
# answer a question stage 2 can ask more cheaply once the ballpark is known.
#
# 50 episodes rather than the ResNet sweeps' 100: this stage picks a learning rate,
# it does not report an accuracy. Anything it produces is a tuning artefact and gets
# re-measured at full episode count later.
#
# Grid is 6 points from 1e-6 to 1e-1, deliberately coarse and deliberately wide.
# Refinement comes after, around whatever each baseline picks. A grid-edge peak is
# reported by sweep_lr.py as a warning and means this needs extending, not that the
# edge value is the answer.
#
# ~3 GPU-hours. Sentinel per dataset, so a kill resumes.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
EPISODES=${EPISODES:-50}
LRS="1e-6 1e-5 1e-4 1e-3 1e-2 1e-1"
BASES="full_ft ln_only_ft"

# --no-amp throughout. Measured on the ResNet arm: GradScaler skipping optimiser
# steps on inf/nan nondeterministically costs 1.99 accuracy points of spread
# between identical runs, against 0.56 with it off, and three draws later the real
# floor turned out to be +/-1.1. Tuning on a 2-point-noisy objective picks the
# noise, not the learning rate.
NOAMP="--no-amp"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# ISIC keeps its own 0.1/0.4/0.5 split and EuroSAT its own 0.2/0.1/0.7, matching the
# ResNet arm exactly. Reusing one dataset's fractions on the other would make the
# two arms incomparable for a reason unrelated to architecture.
run_sweep() {
    # Declared separately on purpose: bash expands every word of a `local`
    # statement before it assigns any of them, so `out="...${tag}"` on the same
    # line reads tag before it exists and dies under `set -u`.
    local tag="$1"
    local data="$2"
    local out="results/vit_lr_${tag}"
    if [ -f "$out/.done" ]; then
        echo "[skip] $tag learning-rate sweep already done"
        return
    fi
    stage "$tag learning rates: $BASES over $LRS"
    # shellcheck disable=SC2086
    $PY experiments/sweep_lr.py $data --arch "$ARCH" \
        --k-shot "$SHOT" --episodes "$EPISODES" \
        --baselines $BASES --lrs $LRS \
        --inner-steps 20 $NOAMP --out "$out" \
        || fail "$tag learning-rate sweep failed"
    mkdir -p "$out" && touch "$out/.done"
}

run_sweep isic    "--dataset isic --fractions 0.1 0.4 0.5"
run_sweep eurosat "--dataset eurosat --fractions 0.2 0.1 0.7"

# -- stage 1b: extend the LayerNorm grid upward --------------------------------
#
# Stage 1 on ISIC put ln_only_ft at 1e-1, the top of the grid, still rising
# monotonically: 39.28 / 39.15 / 39.55 / 42.03 / 44.91 / 49.36 across 1e-6..1e-1.
# "best = 1e-1" there means "largest tried", not "optimum", and sweep_lr.py says so
# in a warning. full_ft by contrast has a clean interior peak at 1e-4 and is done.
#
# So this sweeps ONLY ln_only_ft, and only upward. Values above 1 are not a typo: a
# LayerNorm affine group is ~0.06% of the backbone, so each parameter has to travel
# much further to change the function by as much, and the ResNet arm already saw the
# normalisation group want 300x full_ft's step size. If this also peaks at its own
# top end, extend again rather than accepting the edge -- that is the whole point of
# the warning.
LN_LRS_HI="3e-1 1e0 3e0 1e1"

run_ln_hi() {
    local tag="$1"
    local data="$2"
    local out="results/vit_lr_${tag}_lnhi"
    if [ -f "$out/.done" ]; then
        echo "[skip] $tag high-lr LayerNorm sweep already done"
        return
    fi
    stage "$tag LayerNorm high learning rates: $LN_LRS_HI"
    # shellcheck disable=SC2086
    $PY experiments/sweep_lr.py $data --arch "$ARCH" \
        --k-shot "$SHOT" --episodes "$EPISODES" \
        --baselines ln_only_ft --lrs $LN_LRS_HI \
        --inner-steps 20 $NOAMP --out "$out" \
        || fail "$tag high-lr LayerNorm sweep failed"
    mkdir -p "$out" && touch "$out/.done"
}

# ISIC only. EuroSAT's ln_only_ft is already settled with a clean interior peak at
# 1e-2 (84.32) and 1e-1 collapsing to 71.89, so sweeping 3e-1..1e1 there would spend
# half an hour of GPU confirming a decline that 1e-1 has already shown.
run_ln_hi isic "--dataset isic --fractions 0.1 0.4 0.5"

stage "ViT stage 1 complete -- read the grid-edge warnings before launching stage 2"
