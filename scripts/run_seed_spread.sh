#!/usr/bin/env bash
# How reproducible is the precomputed correction? Three extraction draws, one
# fixed evaluation set.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_seed_spread.sh \
#       > logs/seed_spread.log 2>&1 < /dev/null &'
#
# docs/RESULTS_SUBSPACE.md section 5 reports the precomputed correction at ~39.8
# on ISIC against frozen 38.04 and full_ft 49.6 -- about 15% of the headroom.
# That is ONE extraction draw, and the thing being retracted in that section was
# a claim built on one favourable draw. So it does not get to stand on one draw
# either. This runs three.
#
# --extract-seed varies only the extraction episodes; --seed stays 0 so all
# three draws are scored on identical evaluation episodes. Varying --seed would
# move the eval set too and the ~0.9-point eval-sampling noise at 200 episodes
# would hide the ~0.56-point extraction spread entirely.
#
# --no-amp throughout: with AMP on, GradScaler skipping steps nondeterministically
# costs 1.99 points of spread by itself and there would be nothing left to measure.
#
# --n-query 3 for extraction, 15 for evaluation. These are not arbitrary: the
# ISIC train split is 10% of the corpus and its two thinnest classes hold 14 and
# 11 images, so k_shot+n_query must stay under 11 or the sampler refuses to build
# an episode. The run being reproduced (results/precomputed_v2) used exactly
# these, and changing them would measure a different estimator.
#
# ~2 GPU-hours. Sentinel per seed, so a kill resumes.

set -uo pipefail

PY=.venv/bin/python
DATA="--dataset isic --fractions 0.1 0.4 0.5"
SHOT=${SHOT:-5}

for ES in 0 1 2; do
    OUT="results/precomputed_seed$ES"
    if [ -f "$OUT/.done" ]; then
        echo "[skip] extract-seed $ES already done"
        continue
    fi
    printf '\n========== extract-seed %s  (%s) ==========\n' "$ES" "$(date +%H:%M)"
    $PY experiments/precomputed_shift.py $DATA \
        --k-shot "$SHOT" --n-query 3 --eval-n-query 15 \
        --extract-episodes 100 --eval-episodes 200 \
        --extract-baseline full_ft --inner-lr 1e-4 --inner-steps 20 \
        --seed 0 --extract-seed "$ES" --no-amp \
        --out "$OUT" \
        || { echo "!!! ABORTING: extract-seed $ES failed"; exit 1; }
    mkdir -p "$OUT" && touch "$OUT/.done"
done

printf '\n========== three draws complete  (%s) ==========\n' "$(date +%H:%M)"
