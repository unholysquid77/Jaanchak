#!/usr/bin/env bash
# The ISIC ladder, run sequentially on the compute box.
#
#   ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup bash scripts/run_isic.sh \
#       > logs/isic.log 2>&1 < /dev/null &'
#
# Why ISIC: EuroSAT cannot answer the question the paper asks. A frozen backbone
# already reaches 84.6% there at 5-shot, so every adaptation strategy lands within
# about three points of every other and the Level-1 marginals (-0.05, +0.04) are what
# measuring inside that gap looks like. Published 5-shot accuracy on ISIC is in the
# 40s. The probe run measured 31.3% (1-shot) and 37.5% (5-shot) for untuned
# BatchNorm-only adaptation, so there is real headroom here.
#
# Resumable: every stage has a sentinel, so re-running after an interruption skips
# what finished. Kill by PID, never with pkill -f (landmine 14).
#
# Stages 1-3 are tuning on the *validation* split and cost roughly 8-9 GPU-hours.
# Stage 4 (reporting on test, 600 episodes x 3 seeds) is deliberately gated behind
# REPORT=1: stage 3's outcome is what decides whether reporting is worth ~20 hours.

set -uo pipefail

PY=.venv/bin/python
DATA_ARGS="--dataset isic --fractions 0.1 0.4 0.5"
EPISODES=${EPISODES:-100}
GATE_EPISODES=${GATE_EPISODES:-75}
SEEDS=${SEEDS:-"0 1 2"}
REPORT_EPISODES=${REPORT_EPISODES:-600}
TUNED=results/best_lr_isic.json
GATE_COEF=results/gate_sweep_isic/best_gate_isic.json

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }
fail()  { printf '\n!!! ABORTING: %s\n' "$1"; exit 1; }

# Split fractions are 0.1/0.4/0.5, not EuroSAT's 0.2/0.1/0.7. ISIC's largest class is
# 58x its smallest (NV 6,705 images vs DF 115), and EpisodicSampler raises when any
# class in a split holds fewer than k_shot + n_query images. At a 10% validation
# fraction DF lands ~11 images and the run dies at sampler construction, after the
# backbone has loaded. Verified with experiments/diagnose_splits.py, which reports
# feasibility per (fractions, k-shot) before any GPU time is spent:
#
#   0.2/0.1/0.7   val DF  ~11   BLOCKED at every k-shot
#   0.2/0.2/0.6   val DF   18   BLOCKED from 5-shot
#   0.1/0.3/0.6   val DF   33   BLOCKED at 20-shot
#   0.1/0.4/0.5   val DF   43   OK at 1/5/10/20-shot   <- chosen
#
# The large validation split is also deliberate: ISIC accuracies sit near 40% with
# much wider per-episode variance than EuroSAT's 87%, so tuning needs the episodes.
# 10% is still reserved for train in case Level 3 needs a meta-training pool; Levels
# 0-2 never touch it.

# -- 1. tune baseline learning rates on val ---------------------------------
# Grid spans 3e-6..1e-2. Both ends matter and both were learned the hard way on
# EuroSAT: full fine-tuning optimised at 1e-5 while BatchNorm-only wanted 1e-3 to
# 3e-3, so a grid covering one leaves the other pinned to an edge and its "optimum"
# is only a lower bound (landmine 8). The ISIC probe already peaked bn_only_ft at the
# top of the old default grid, which is exactly that failure starting again.
if [ ! -f results/isic_sweep/.done ]; then
    stage "tuning baseline LRs on val, 1/5-shot, ${EPISODES} episodes per config"
    $PY experiments/sweep_lr.py $DATA_ARGS \
        --k-shot 1 5 --episodes "$EPISODES" \
        --lrs 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 \
        --inner-steps 5 20 \
        --out results/isic_sweep \
        || fail "baseline LR sweep failed"
    touch results/isic_sweep/.done
else
    echo "[skip] baseline LR sweep already done"
fi

# -- 2. merge and verify the grid -------------------------------------------
stage "merging and checking the tuning grid"
$PY experiments/merge_sweeps.py results/isic_sweep --dataset isic --out "$TUNED" \
    || fail "merge failed"

# merge_sweeps prints a warning for grid-edge optima but exits 0, so enforce it here.
# A zero exit from the sweep is not evidence that tuning is usable: a partially
# populated tuning file is precisely what caused three EuroSAT baselines to be
# reported at an untuned default (landmine 10).
$PY - <<'PYEOF' || fail "tuning incomplete or still truncated -- see above"
import json, sys
best = json.load(open("results/best_lr_isic.json"))
raw = json.load(open("results/isic_sweep/sweep_isic_raw.json"))
want = {"last_layer_ft", "stage4_ft", "bn_only_ft", "progressive_unfreeze", "full_ft"}
shots = ("1", "5")

missing = [(b, k) for b in sorted(want) for k in shots if k not in best.get(b, {})]
if missing:
    print("missing tuned settings:", missing); sys.exit(1)

grids = {}
for r in raw:
    grids.setdefault((r["baseline"], str(r["k_shot"])), set()).add(r["lr"])
edge = []
for b in sorted(want):
    for k in shots:
        g = sorted(grids[(b, k)])
        lr = best[b][k]["lr"]
        if lr in (g[0], g[-1]) and len(g) > 1:
            edge.append((b, k, lr, "top" if lr == g[-1] else "bottom"))
if edge:
    print("optima pinned to a grid edge -- extend the grid before reporting:")
    for b, k, lr, which in edge:
        print(f"  {b} {k}-shot lr={lr:.0e} ({which})")
    sys.exit(1)
print(f"tuning complete: {len(want)} baselines x {len(shots)} shots, all interior optima")
PYEOF

# -- 3. tune the adaptive gate on val ---------------------------------------
# Coefficient grids are 0/1/2, the scale appropriate to standardised signals. The
# script's old defaults (0/2/4, 0/4/8) belong to the raw-signal era and saturate the
# gate into a constant (landmine 5); they are now fixed at the source, and passed
# explicitly here so this run's configuration is legible from the script alone.
#
# Stage-1 LR grid is extended past EuroSAT's: the gate adapts the whole backbone, and
# every ISIC baseline tuned higher than its EuroSAT counterpart.
if [ ! -f results/gate_sweep_isic/.done ]; then
    stage "tuning the adaptive gate on val, ${GATE_EPISODES} episodes per config"
    $PY experiments/sweep_gate.py $DATA_ARGS \
        --k-shot 1 5 --episodes "$GATE_EPISODES" \
        --lrs 1e-5 3e-5 1e-4 3e-4 1e-3 \
        --biases 0 1 2 --utilities 0 1 2 --drifts 0 1 2 \
        --out results/gate_sweep_isic \
        || fail "gate sweep failed"
    touch results/gate_sweep_isic/.done
else
    echo "[skip] gate sweep already done"
fi

stage "tuning complete"
cat <<'MSG'

Tuning done. Read results/best_lr_isic.json and results/gate_sweep_isic/ before
spending ~20 hours on reporting: the marginal effect of each signal on val is what
decides whether Level 1 is rescued on a hard dataset or reported as a negative.

Report on test with:
    REPORT=1 bash scripts/run_isic.sh
MSG

# -- 4. report on test (gated) ----------------------------------------------
if [ "${REPORT:-0}" != "1" ]; then
    echo "[skip] stage 4 (test reporting) -- set REPORT=1 to run it"
    exit 0
fi

if [ ! -f results/isic_level0/.done ]; then
    stage "reporting baselines on test, ${REPORT_EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_baselines.py $DATA_ARGS \
        --k-shot 1 5 --episodes "$REPORT_EPISODES" --seeds $SEEDS \
        --tuned "$TUNED" --skip-existing --out results/isic_level0 \
        || fail "baseline reporting failed"
    touch results/isic_level0/.done
else
    echo "[skip] baseline reporting already done"
fi

if [ ! -f results/isic_gate/.done ]; then
    [ -f "$GATE_COEF" ] || fail "no tuned gate coefficients at $GATE_COEF"
    stage "reporting the gate on test, ${REPORT_EPISODES} episodes x seeds ${SEEDS}"
    $PY experiments/run_gate.py $DATA_ARGS \
        --k-shot 1 5 --episodes "$REPORT_EPISODES" --seeds $SEEDS \
        --coefficients "$GATE_COEF" --split test \
        --skip-existing --out results/isic_gate \
        || fail "gate reporting failed"
    touch results/isic_gate/.done
else
    echo "[skip] gate reporting already done"
fi

stage "complete"
