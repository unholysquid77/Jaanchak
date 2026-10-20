#!/usr/bin/env bash
# Second and third draws of the ISIC depth curve.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_isic_seed.sh \
#       > logs/vit_position_isic_seed.log 2>&1 < /dev/null &'
#
# WHY. RESULTS_VIT.md §3.3.5 is single-draw at every depth, and §4 now calls it the
# largest hole in the arm. Three things rest on draws it does not have:
#
#   * the PLATEAU over blocks 0/2/5 (three accuracies inside 0.40, with no pairwise
#     difference settled) -- consistent with one flat line, and equally consistent
#     with a gentle curve at this noise level
#   * the FALL to blocks 8 and 11 (block5 − block11 = +4.23 ± 1.51 at one draw)
#   * the MIRRORED-CURVE claim, which compares an ISIC single draw against a EuroSAT
#     three-draw curve. §3.3.2a showed what draws do to a marginal link there:
#     block8 − block11 went +1.19 (one draw) → +1.54 ± 0.80 (three). The ISIC links
#     deserve the same treatment before the mirror is quoted as a result.
#
# NO RE-TUNING: seed-0 tuned lrs, --split-seed held at 0, only the episode sample
# moves, exactly as scripts/run_vit_position_seed.sh does it for EuroSAT. Block 0's
# lr is 3e-3 from results/vit_best_lr_isicpos0_v2.json -- the value AFTER its upward
# extension, not the pre-extension pick. The merge below prefers the _v2 file for
# that reason and says which file each lr came from.
#
# SEEDS 1 AND 2, not just 1: with n=2 the t-multiplier is 12.7 and the interval is
# useless whatever the data says.
#
# ~3 GPU-hours (5 baselines x 2 seeds x ~15 min; ISIC episodes run slower than
# EuroSAT's).

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

SEEDS=${SEEDS:-"1 2"}
MERGED="results/vit_best_lr_isicposcurve.json"
OUT="results/vit_isicposcurve_seeds"

# Depth 11 runs as attn_last1_ft: attn_block11_ft is deliberately unregistered
# because it duplicates it, and test_attn_block_11_is_attn_last_1 asserts the
# selectors are identical. Same mapping as scripts/run_vit_position_isic.sh.
CURVE="attn_block0_ft attn_block2_ft attn_block5_ft attn_block8_ft attn_last1_ft"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

stage "preflight: every baseline name must resolve"
for base in $CURVE; do
    $PY - "$base" <<'PYEOF' || fail "baseline not registered"
import sys
from arc.train.baselines import BASELINES
if sys.argv[1] not in BASELINES:
    print(f"!!! {sys.argv[1]!r} is not in BASELINES")
    sys.exit(1)
print(f"[preflight] {sys.argv[1]} -> ok")
PYEOF
done

stage "merge the per-depth tuned learning rates"
$PY - "$MERGED" $CURVE <<'PYEOF' || fail "tuned-lr merge failed"
import glob, json, os, sys

out_path, wanted = sys.argv[1], sys.argv[2:]

# Prefer a _v2 file where one exists: that is the post-extension tuning, and using
# the pre-extension pick would measure a learning rate the document no longer
# claims. Block 0 is the case in point (3e-3 after extension).
by_base, source = {}, {}
for path in sorted(glob.glob("results/vit_best_lr_isicpos*.json")):
    is_v2 = path.endswith("_v2.json")
    for base, by_k in json.load(open(path)).items():
        if base in by_base and not is_v2:
            continue            # already have something; only a _v2 may overwrite
        if base in by_base and is_v2:
            print(f"[merge] {base}: {os.path.basename(path)} supersedes "
                  f"{os.path.basename(source[base])}")
        by_base[base] = by_k
        source[base] = path

missing = [b for b in wanted if b not in by_base]
if missing:
    print(f"!!! no tuned lr found for: {', '.join(missing)}")
    sys.exit(1)

merged = {b: by_base[b] for b in wanted}
json.dump(merged, open(out_path, "w"), indent=2)
for b in wanted:
    for k, d in merged[b].items():
        print(f"[tuned] {b:20s} k={k} lr={d['lr']:g} (from {os.path.basename(source[b])})")
PYEOF

if [ -f "$OUT/.done" ]; then
    echo "[skip] seeds $SEEDS already done"
else
    stage "ISIC depth curve, seeds $SEEDS, 100 episodes each"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $ISIC_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --seeds $SEEDS --split-seed 0 \
        --baselines $CURVE --tuned "$MERGED" \
        --inner-steps 20 $NOAMP --skip-existing --out "$OUT" \
        || fail "seed sweep failed"
    mkdir -p "$OUT" && touch "$OUT/.done"
fi

stage "done"
echo "Seed 0 rows: results/vit_isicpos{2,5,8,11} and results/vit_isicpos0_v2"
echo "(NOT vit_isicpos0 -- block 0 was re-measured after its grid extension)."
echo "Seeds 1-2: $OUT. Depth 11 is filed under attn_last1_ft."
echo
echo "Read as a PER-SEED DIFFERENCE: mean +/- t(2)*sd/sqrt(3), t(2)=4.303."
echo "The links this exists to settle, seed-0 values:"
echo "  block5 - block11 = +4.23   the fall at the deep end"
echo "  block0 - block5  = -0.40   the shallow plateau (expected to stay unsettled)"
