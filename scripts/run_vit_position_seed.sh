#!/usr/bin/env bash
# Second and third draws of the EuroSAT depth curve.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_seed.sh \
#       > logs/vit_position_seed.log 2>&1 < /dev/null &'
#
# WHY. Two claims in RESULTS_VIT.md §3.3.2/§3.3.3 sit right on the edge of both bars
# and are currently written as "suggested", which is the correct reading of one draw:
#
#   block 8 - block 11       = +1.19 +/- 1.11   (the inverted U in depth)
#   block 8 - attn_last6_ft  = +1.00 +/- 0.96   (adding blocks to the best one costs)
#
# Both clear their per-episode interval and the ~0.6 between-run floor only just. §3.4
# establishes that the valid test across draws is the PER-SEED DIFFERENCE, so the fix
# is draws, not more episodes. Everything else about the depth result is far clear of
# the floor and does not need this; these two do, and they are the difference between
# "a monotone rise through block 8" and "an interior peak", which is a real difference
# in what the paper claims.
#
# SEEDS 1 AND 2, not just 1. With n=2 the t-multiplier is 12.7 and the interval is
# useless whatever the data says, so a second draw alone could not settle anything.
# n=3 matches §3.4's existing three-draw tables and makes the two curves comparable.
#
# NO RE-TUNING. Learning rates are the seed-0 tuned values, exactly as
# scripts/run_vit_seeds.sh does it: only the episode sample moves, so the comparison
# is like-with-like and a per-seed difference is meaningful. Note that block 8's lr is
# 3e-3, the value from the grid extension (results/vit_pos8_lr_hi), NOT the 1e-3 that
# the original truncated grid chose -- re-running the old value would measure a point
# nothing in the document claims any more.
#
# ~2.5 GPU-hours (6 baselines x 2 seeds x ~12 min).

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

SEEDS=${SEEDS:-"1 2"}
MERGED="results/vit_best_lr_poscurve.json"

# The six baselines that make up the depth curve plus the two suffix points the
# marginal claims compare against. attn_block11_ft is not a registered baseline --
# it IS attn_last1_ft (see scripts/run_vit_position_isic.sh for why) -- so block 11
# appears here under that name.
CURVE="attn_block0_ft attn_block2_ft attn_block5_ft attn_block8_ft attn_last1_ft attn_last6_ft"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# Merge the per-depth tuned files into one, because --tuned takes a single JSON.
# Done here rather than by hand so the merged file is reproducible and so a
# collision or a missing baseline aborts instead of silently falling back to
# --inner-lr, which would turn every row into a strawman at one shared rate.
stage "merge the per-depth tuned learning rates"
$PY - "$MERGED" $CURVE <<'PYEOF' || fail "tuned-lr merge failed"
import glob, json, sys

out_path, wanted = sys.argv[1], sys.argv[2:]
merged, seen = {}, {}
for path in sorted(glob.glob("results/vit_best_lr_pos[0-9].json")
                   + glob.glob("results/vit_best_lr_cap[0-9].json")):
    for base, by_k in json.load(open(path)).items():
        if base in seen:
            print(f"!!! {base} appears in both {seen[base]} and {path}")
            sys.exit(1)
        seen[base] = path
        merged[base] = by_k

missing = [b for b in wanted if b not in merged]
if missing:
    print(f"!!! no tuned lr found for: {', '.join(missing)}")
    sys.exit(1)

# Keep only what this sweep runs, so a stray baseline cannot ride along unnoticed.
merged = {b: merged[b] for b in wanted}
json.dump(merged, open(out_path, "w"), indent=2)
for b in wanted:
    for k, d in merged[b].items():
        print(f"[tuned] {b:20s} k={k} lr={d['lr']:g} (from {seen[b].split('/')[-1]})")
PYEOF

OUT="results/vit_poscurve_seeds"

# --seeds takes several episode seeds in one call and --skip-existing works per
# (baseline, k-shot, seed) cell, so this is resumable at the granularity of one
# measurement without a per-seed loop or per-seed sentinel.
#
# --split-seed STAYS 0 and is passed explicitly because it is load-bearing: it fixes
# the train/val/test partition and must match the seed the learning rates were tuned
# under. Driving it from the episode seed would evaluate each draw on a DIFFERENT
# validation set, which makes averaging across draws incoherent and silently
# invalidates every tuned lr. run_baselines.py keeps the two separate for this reason;
# the comment at its definition says so.
if [ -f "$OUT/.done" ]; then
    echo "[skip] seeds $SEEDS already done"
else
    stage "EuroSAT depth curve, seeds $SEEDS, 100 episodes each"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --seeds $SEEDS --split-seed 0 \
        --baselines $CURVE --tuned "$MERGED" \
        --inner-steps 20 $NOAMP --skip-existing --out "$OUT" \
        || fail "seed sweep failed"
    mkdir -p "$OUT" && touch "$OUT/.done"
fi

stage "done"
echo "Seed 0 rows live in results/vit_pos{0,2,5}, results/vit_pos8_v2 (NOT vit_pos8),"
echo "results/vit_cap1 and results/vit_cap6. Seeds 1-2 in results/vit_poscurve_seeds."
echo
echo "Read it as a PER-SEED DIFFERENCE: mean +/- t(2)*sd/sqrt(3), t(2)=4.303."
echo "The two claims this exists to settle, seed 0 values:"
echo "  block8 - block11(attn_last1) = +1.19   inverted U in depth"
echo "  block8 - attn_last6          = +1.00   adding blocks to the best one costs"
