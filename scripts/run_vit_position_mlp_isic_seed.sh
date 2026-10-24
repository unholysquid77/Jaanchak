#!/usr/bin/env bash
# Three draws of the ISIC MLP depth curve, so §3.10's single-draw curve can be read.
#
# WHAT IS AT STAKE. Seed 0 gave a shallow plateau falling to the deep end
# (50.85 / 50.63 / 49.91 / 47.69 / 43.44), with every shallow-vs-deep comparison
# settled and block 0 - block 5 UNSETTLED at +0.95. That is the mirror holding in a
# second parameter type, which answers §4's "the type x depth interaction is one
# dataset" entry -- but on one draw. Seven links in this project moved under
# additional draws: four promoted, three demoted. The big gaps here (+4.25 to +7.41)
# are far from the floor and are unlikely to move; the shallow plateau's internal
# ordering is exactly the kind of marginal structure that does move, and it is not
# claimed either way until these seeds land.
#
# NO RE-TUNING: seed-0 tuned lrs, --split-seed held at 0, only the episode sample
# moves. Re-tuning per seed would confound the draw with the learning rate.
#
# Block 5's lr comes from the POST-EXTENSION file
# (results/vit_best_lr_isicmlppos5_v2.json, lr=3e-3). Block 5 hit a top edge on the
# first pass; the extension added 1e-2 and 3e-2, found 1e-2 collapses to 38.00, and
# confirmed 3e-3 as the genuine optimum with no edge on the merged grid. The merge
# below prefers _v2 and logs the supersession -- verified by reading it, not assumed
# from this comment.
#
# SEEDS 1 AND 2 only: seed 0 already exists. With n=2 the t-multiplier is 12.7 and
# the interval is useless, so all three are needed before anything is quoted.
#
# ~2.5 GPU-hours (5 baselines x 2 seeds x ~13 min). Resumable via the OUT sentinel.
set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
ISIC_DATA="--dataset isic --fractions 0.1 0.4 0.5"

SEEDS=${SEEDS:-"1 2"}
MERGED="results/vit_best_lr_isicmlpposcurve.json"
OUT="results/vit_isicmlpposcurve_seeds"

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
for path in sorted(glob.glob("results/vit_best_lr_isicmlppos*.json")):
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
    stage "ISIC MLP depth curve, seeds $SEEDS, 100 episodes each"
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
