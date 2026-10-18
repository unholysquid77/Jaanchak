#!/usr/bin/env bash
# Close the attention grid in the inner-step test, then re-measure the gap.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_vit_mech2.sh \
#       > logs/vit_mech2.log 2>&1 < /dev/null &'
#
# Run after scripts/run_vit_mech.sh finishes.
#
# WHY. run_vit_mech.sh re-tuned both groups at 5 and 10 inner steps so the gap would
# not be measured at settings chosen for 20. full_ft moved as expected (1e-5 at 20
# steps -> 3e-5 at 5) and peaked interior. attn_only_ft did NOT: it peaked at 1e-4,
# the BOTTOM of the grid 1e-4/3e-4/1e-3, at both step counts. My "fewer steps want a
# larger step" reasoning held for full_ft and not for attention, so the attn grid was
# pitched too high and its reported accuracy is a bound.
#
# THIS MATTERS IN THE DIRECTION THAT WOULD FOOL ME. An undertuned attn makes attn
# look worse, which SHRINKS the attn - full_ft gap, which is exactly the signature
# the overfitting mechanism predicts. The 5-step numbers already in hand --
# attn 85.21, full_ft 84.71, gap +0.50 against +2.50 at 20 steps -- are therefore
# unreadable as evidence for the mechanism until attention is tuned properly. They
# may well survive; the point is that believing them now would be believing my own
# grid error.
#
# Extends downward to 1e-5 and 3e-5, re-merges, and re-measures into a FRESH output
# directory. The fresh directory is not cosmetic: run_baselines.py --skip-existing
# would otherwise keep the stale attn row computed at the bound.
#
# ~1 GPU-hour.

set -uo pipefail

PY=.venv/bin/python
export PYTHONPATH=src

ARCH="vit_base_patch16_224.augreg_in21k"
SHOT=${SHOT:-5}
NOAMP="--no-amp"
EURO_DATA="--dataset eurosat --fractions 0.2 0.1 0.7"

ATTN_LRS_LO="1e-5 3e-5"
STEP_COUNTS="5 10"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

run_fix() {
    local steps="$1"
    local lo="results/vit_steps${steps}_lr_attnlo"
    local out="results/vit_steps${steps}_v2"

    if [ -f "$out/.done" ]; then
        echo "[skip] ${steps}-step re-measurement already done"
        return
    fi

    if [ ! -f "$lo/.done" ]; then
        stage "EuroSAT ${steps} steps: extend attn downward over $ATTN_LRS_LO"
        # shellcheck disable=SC2086
        $PY experiments/sweep_lr.py $EURO_DATA --arch "$ARCH" \
            --k-shot "$SHOT" --episodes 50 \
            --baselines attn_only_ft --lrs $ATTN_LRS_LO \
            --inner-steps "$steps" $NOAMP --out "$lo" \
            || fail "${steps}-step attn downward sweep failed"
        mkdir -p "$lo" && touch "$lo/.done"
    fi

    stage "EuroSAT ${steps} steps: re-merge all three sweeps"
    $PY experiments/merge_sweeps.py \
        "results/vit_steps${steps}_lr" \
        "results/vit_steps${steps}_lr_attn" \
        "$lo" \
        --dataset eurosat \
        --out "results/vit_best_lr_steps${steps}.json" \
        --edges-out "results/vit_edges_steps${steps}.json" \
        || fail "${steps}-step re-merge failed"

    $PY - "results/vit_edges_steps${steps}.json" "$steps" <<'PYEOF' || true
import json, sys
edges = json.loads(open(sys.argv[1]).read())
if edges:
    print(f"\n!!! STILL AT A GRID EDGE at {sys.argv[2]} steps:")
    for e in edges:
        print(f"    {e['baseline']} lr={e['lr']:.0e} ({e['side']} edge)")
    print("!!! extend AGAIN before reading the gap. Two extensions in one direction")
    print("!!! means the regime is not where I think it is.\n")
else:
    print(f"\nboth groups interior at {sys.argv[2]} steps -- the gap is readable.\n")
PYEOF

    stage "EuroSAT ${steps} steps: re-measure full_ft vs attn at 100 episodes"
    # shellcheck disable=SC2086
    $PY experiments/run_baselines.py $EURO_DATA --arch "$ARCH" \
        --split val --k-shot "$SHOT" --episodes 100 \
        --baselines full_ft attn_only_ft \
        --tuned "results/vit_best_lr_steps${steps}.json" \
        --inner-steps "$steps" $NOAMP --skip-existing --out "$out" \
        || fail "${steps}-step re-measurement failed"
    mkdir -p "$out" && touch "$out/.done"
}

for n in $STEP_COUNTS; do
    run_fix "$n"
done

stage "inner-step test re-measured"
echo "Read results/vit_steps{5,10}_v2 against results/vit_2b_eurosat (20 steps)."
echo "The mechanism predicts the attn - full_ft gap SHRINKS as steps fall."
echo "Ignore results/vit_steps{5,10} -- attn there is at a grid bound."
