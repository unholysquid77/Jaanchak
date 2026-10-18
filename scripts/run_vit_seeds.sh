#!/usr/bin/env bash
# ViT seed spread: two more episode draws of the claims stage 2b turns on.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_seeds.sh \
#       > logs/vit_seeds.log 2>&1 < /dev/null &'
#
# Run after scripts/run_vit_2b.sh. It reuses that stage's tuned learning-rate files.
#
# WHY THIS AND NOT THE DISPLACEMENT PROBE. Stage 2b produced the strongest claim in
# the ViT arm -- attn_only_ft beats full_ft by 2.72 +/- 1.69 on EuroSAT -- and the
# weakest-but-live one, full_ft beating the best single type by 2.29 +/- 2.21 on
# ISIC. Both are single draws read against a within-run confidence interval, and
# RESULTS_SUBSPACE.md section 7 is the standing reason not to trust that: three
# draws there turned a believed +/-0.56 estimator floor into a measured +/-1.1,
# which is to say the within-run interval understated the real spread by 2x. Two of
# the three live claims in RESULTS_VIT.md section 4 sit inside that kind of error.
# A frontier built on top of unverified margins would inherit them.
#
# WHAT VARIES. --seeds moves the EPISODE sample only. --split-seed stays 0, so the
# train/val/test partition is identical across draws; varying both would confound
# estimator noise with split luck and make the draws incomparable. The learning
# rates are the ones tuned on seed 0 and are deliberately NOT re-tuned: the question
# is how much an accuracy moves between draws at a fixed configuration, so re-tuning
# per seed would measure something else and would also be tuning on the thing being
# measured.
#
# WHAT IS NOT SWEPT. Four baselines, not eight. full_ft and attn_only_ft are the two
# sides of both live claims, ln_only_ft is the 0.045% group that makes the parameter-
# count point, and frozen_proto is the floor every "gain over frozen" number divides
# by -- so an unstable floor would move every one of them. ln_scale, ln_shift, mlp
# and patch_embed are not in a claim that turns on a 1-3 point margin, and adding
# them would double a 3.5-hour run to answer nothing currently open.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
EPISODES=${EPISODES:-100}
SEEDS=${SEEDS:-"1 2"}
NOAMP="--no-amp"

# NOT named GROUPS: that is a bash special variable and assigning to it is silently
# ignored. See scripts/run_vit_2b.sh and tests/test_shell_special_vars.py.
SEED_BASELINES="full_ft attn_only_ft ln_only_ft frozen_proto"

ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

run_seeds() {
    local tag="$1"
    local data="$2"
    local out="results/vit_seeds_${tag}"
    local tuned="results/vit_best_lr_${tag}.json"

    if [ -f "$out/.done" ]; then
        echo "[skip] $tag seed spread already done"
        return
    fi
    [ -f "$tuned" ] || fail "$tag: $tuned missing -- run scripts/run_vit_2b.sh first"

    stage "$tag seeds $SEEDS at $EPISODES episodes: $SEED_BASELINES"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $data --arch "$ARCH" \
        --split val \
        --k-shot "$SHOT" --episodes "$EPISODES" --seeds $SEEDS \
        --baselines $SEED_BASELINES --tuned "$tuned" \
        --inner-steps 20 $NOAMP --skip-existing --out "$out" \
        || fail "$tag seed spread failed"
    mkdir -p "$out" && touch "$out/.done"
}

# EuroSAT first this time. It carries the claim that most needs checking, so if only
# one dataset finishes it should be the one where a reader will least expect the
# result.
run_seeds eurosat "$EURO_DATA"
run_seeds isic    "$ISIC_DATA"

stage "ViT seed spread complete"
echo "Combine with the seed-0 rows in results/vit_2b_{isic,eurosat} for three draws."
echo "The number that matters is the SPREAD ACROSS DRAWS, not the within-run ci95."
