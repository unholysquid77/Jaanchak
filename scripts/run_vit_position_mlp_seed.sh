#!/usr/bin/env bash
# Second and third draws of the MLP depth curve.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_position_mlp_seed.sh \
#       > logs/vit_position_mlp_seed.log 2>&1 < /dev/null &'
#
# WHY. §3.3.6 is single-draw at every depth, and §4 records that the type x depth
# interaction therefore inherits the weaker of the two curves' draw counts: the
# attention curve has three (§3.3.2a), the MLP curve has one. The interaction is now
# the organising claim of §3.3, so it should not rest on the weaker half.
#
# What draws would decide, in order of how close they sit to the floor:
#
#   * mlp_b8 - attn_b8 = +1.21 ± 0.96 at one draw -- the narrowest cell in the
#     interaction table, and the one that makes "MLP wins at block 8" rather than
#     "they tie there"
#   * mlp_b11 - attn_b11 = -1.33 ± 0.85 -- the SIGN FLIP, which is the whole point of
#     calling it an interaction. If this does not survive draws, the honest reading
#     collapses to "MLP is ahead at every depth measured, by a margin that shrinks
#     with depth", which is a weaker and quite different claim.
#   * mlp_b2 - mlp_b0 = -0.67 ± 1.17, unsettled at one draw -- whether MLP's shallow
#     region is flat like attention's
#
# §3.3.5a is the cautionary case: three draws DEMOTED ISIC's block11-block8 from
# settled to not settled, and §3.3.2a is the opposite, where three draws PROMOTED
# two marginal EuroSAT links. Single draws are uninformative near the floor in
# either direction, so the sign flip could go either way.
#
# NO RE-TUNING: seed-0 tuned lrs, --split-seed held at 0, only the episode sample
# moves. Block 5's lr comes from the POST-EXTENSION file
# (results/vit_best_lr_mlppos5_v2.json, 3e-3); the merge below prefers _v2 and logs
# the supersession, the same mechanism scripts/run_vit_position_isic_seed.sh uses.
#
# SEEDS 1 AND 2: with n=2 the t-multiplier is 12.7 and the interval is useless.
#
# ~2.5 GPU-hours (5 baselines x 2 seeds x ~13 min).

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

SEEDS=${SEEDS:-"1 2"}
MERGED="results/vit_best_lr_mlpposcurve.json"
OUT="results/vit_mlpposcurve_seeds"

# mlp_block11_ft IS registered, unlike attn_block11_ft -- there is no MLP suffix
# family for it to duplicate (see src/arc/train/baselines.py), so all five depths
# appear under their own names here.
CURVE="mlp_block0_ft mlp_block2_ft mlp_block5_ft mlp_block8_ft mlp_block11_ft"

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

# Prefer _v2 where it exists: that is the post-extension tuning. Using the
# pre-extension pick would measure a learning rate the document no longer claims.
by_base, source = {}, {}
for path in sorted(glob.glob("results/vit_best_lr_mlppos*.json")):
    is_v2 = path.endswith("_v2.json")
    for base, by_k in json.load(open(path)).items():
        if base in by_base and not is_v2:
            continue
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
        print(f"[tuned] {b:18s} k={k} lr={d['lr']:g} (from {os.path.basename(source[b])})")
PYEOF

if [ -f "$OUT/.done" ]; then
    echo "[skip] seeds $SEEDS already done"
else
    stage "EuroSAT MLP depth curve, seeds $SEEDS, 100 episodes each"
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
echo "Seed 0 rows: results/vit_mlppos{0,2,8,11} and results/vit_mlppos5_v2"
echo "(NOT vit_mlppos5 -- block 5 was re-measured after its grid extension)."
echo "Seeds 1-2: $OUT."
echo
echo "Read as a PER-SEED DIFFERENCE: mean +/- t(2)*sd/sqrt(3), t(2)=4.303."
echo "Attention three-draw means for the interaction (results/vit_poscurve_seeds):"
echo "  b0 78.76  b2 78.64  b5 81.78  b8 87.49  b11 85.95"
echo "The three links this exists to settle, seed-0 values:"
echo "  mlp_b8  - attn_b8  = +1.21   the narrowest cell"
echo "  mlp_b11 - attn_b11 = -1.33   THE SIGN FLIP"
echo "  mlp_b2  - mlp_b0   = -0.67   is MLP's shallow region flat too"
