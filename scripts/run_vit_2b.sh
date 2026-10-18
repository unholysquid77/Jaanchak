#!/usr/bin/env bash
# ViT stage 2b: the parameter-type table, each group at its own tuned learning rate.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_2b.sh \
#       > logs/vit_2b.log 2>&1 < /dev/null &'
#
# Run this only after scripts/run_vit.sh has finished. It reads that script's sweep
# output, so launching it early would tune on a half-finished grid.
#
# WHAT THIS DECIDES. Stage 1 found that on EuroSAT ln_only_ft ties full_ft (84.32 vs
# 84.29, 0.06% of the parameters) while on ISIC it does not (49.36 vs 53.07, a
# 3.71-point gap against a combined interval of +/-3.51 -- clearing zero by 0.2
# points, at 50 episodes). That split is the most interesting thing in the project
# and it is currently 0.2 points wide. 100 episodes halves the interval if the
# accuracies hold, and that is what settles it.
#
# Stage 2a then found, on ISIC, five groups spanning 2951x in parameter count all
# landing inside 49.36-49.95 while full_ft sat 3.7 points above every one of them --
# unlike the ResNet arm, where full_ft was *inside* the band. If that survives here
# it is a ceiling no proper subset of parameter types passes, which is a claim about
# joint movement across types rather than about parameter count. It is also exactly
# the kind of reading that three single-draw artefacts have already cost this
# project, so it stays out of the write-up until this stage agrees with it.
#
# VALIDATION ONLY. --split val, explicitly, on every call. run_baselines.py used to
# sample splits["test"] unconditionally; it now defaults to val and tests/
# test_no_hardcoded_test_split.py keeps it that way. Nothing here may read test.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}

# 100 episodes, double stage 2a's. This stage reports accuracies rather than picking
# a hyperparameter, so the tuning-artefact exemption does not apply to it.
EPISODES=${EPISODES:-100}
NOAMP="--no-amp"

ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

# The full partition plus both brackets. frozen_proto is the floor (no gradient steps
# at all) and full_ft the ceiling; without them a 0.6-point band across six groups
# cannot be read, because nothing says how much of the headroom that band sits in.
GROUPS="full_ft ln_only_ft ln_scale_ft ln_shift_ft attn_only_ft mlp_only_ft patch_embed_ft frozen_proto"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# -- stage 2a-ext3: EuroSAT patch_embed peaked at the BOTTOM of its grid ------
#
#   1e-6:80.13  1e-5:77.97  1e-4:76.24  1e-3:61.73  1e-2:51.12
#
# Monotone decline from the smallest step tried, so 1e-6 is "smallest tried", not an
# optimum -- the same defect as the four upward extensions, in the other direction.
# The natural reading is that adapting the patch embedding does nothing useful on
# EuroSAT and the best step size is whichever is closest to frozen, but that is a
# story about the shape of a curve, not a measurement. Two more points below settle
# it: if 1e-8 and 1e-7 climb towards frozen_proto, the group is simply inert here.
#
# ISIC needs no such extension: its patch_embed peaked interior at 1e-3 (46.24).
EURO_PATCH_LRS_LO="1e-8 1e-7"

run_patch_lo() {
    local out="results/vit_g_eurosat_patchlo"
    if [ -f "$out/.done" ]; then
        echo "[skip] EuroSAT patch_embed low-lr sweep already done"
        return
    fi
    stage "EuroSAT patch_embed low learning rates: $EURO_PATCH_LRS_LO"
    # shellcheck disable=SC2086
    $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
        --k-shot "$SHOT" --episodes 50 \
        --baselines patch_embed_ft --lrs $EURO_PATCH_LRS_LO \
        --inner-steps 20 $NOAMP --out "$out" \
        || fail "EuroSAT patch_embed low-lr sweep failed"
    mkdir -p "$out" && touch "$out/.done"
}

run_patch_lo

# -- the table ----------------------------------------------------------------
#
# Merge every sweep that touched this dataset into one tuned-settings file, then drop
# any group still peaking at the TOP of the merged grid. A top-edge learning rate is
# a lower bound, so reporting that group would understate it by an unknown amount --
# a hole in the table is honest, an understated row is not. Bottom edges are kept;
# frozen_proto is in the table precisely so a near-inert group is readable.
run_table() {
    local tag="$1"
    local data="$2"
    shift 2
    local out="results/vit_2b_${tag}"
    local tuned="results/vit_best_lr_${tag}.json"
    local edges="results/vit_edges_${tag}.json"

    if [ -f "$out/.done" ]; then
        echo "[skip] $tag parameter-type table already done"
        return
    fi

    stage "$tag merge sweeps -> $tuned"
    $PY experiments/merge_sweeps.py "$@" --dataset "$tag" \
        --out "$tuned" --edges-out "$edges" \
        || fail "$tag merge failed"

    local keep
    # shellcheck disable=SC2086
    keep=$($PY experiments/drop_edge_baselines.py "$edges" \
               --baselines $GROUPS --k-shot "$SHOT") \
        || fail "$tag edge filter failed"
    [ -n "$keep" ] || fail "$tag: every baseline was dropped as grid-edge"

    stage "$tag parameter-type table at $EPISODES episodes: $keep"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $data --arch "$ARCH" \
        --split val \
        --k-shot "$SHOT" --episodes "$EPISODES" \
        --baselines $keep --tuned "$tuned" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "$tag parameter-type table failed"
    mkdir -p "$out" && touch "$out/.done"
}

# ISIC first: it carries both open questions (does ln_only_ft stay behind full_ft,
# and does the 2951x band hold), so if only one dataset finishes it should be this
# one. vit_lr_isic_lnhi is the upward extension that bracketed ln_only_ft at 1e-1.
run_table isic "$ISIC_DATA" \
    results/vit_lr_isic results/vit_lr_isic_lnhi \
    results/vit_g_isic_lnhalves results/vit_g_isic_weights \
    results/vit_g_isic_weightshi results/vit_g_isic_patch

run_table eurosat "$EURO_DATA" \
    results/vit_lr_eurosat \
    results/vit_g_eurosat_lnhalves results/vit_g_eurosat_lnhalveshi \
    results/vit_g_eurosat_weights results/vit_g_eurosat_weightshi \
    results/vit_g_eurosat_patch results/vit_g_eurosat_patchlo

stage "ViT stage 2b complete"
echo "Read results/vit_2b_isic and results/vit_2b_eurosat with experiments/make_table.py."
echo "Any group named DROP in this log needs its grid extended before it can be reported."
