# ARC — Handoff

**Last updated:** 2026-10-16 (probes)
**State:** Level 0 complete and reportable on EuroSAT. Level 1 built and tuned, and
now producing a null result on **two** datasets — the signals are actively harmful on
the harder one. ISIC tuning complete (`docs/RESULTS_ISIC.md`); ISIC test reporting not
run. Level 2 not started.

Four probes (`docs/RESULTS_SUBSPACE.md`) now explain *why* Level 1 is null and point
at a different paper: adaptation is ~90% a per-channel affine, near rank-1 at the top
across 200 tasks, units are redundant, every parameter group plateaus at ~49.5, and
neither signal predicts within-episode which unit to adapt. The open question is
whether the correction can be precomputed offline and applied with no test-time
gradient steps.

`results/` is now tracked in git. It was ignored until 2026-10-16, which left ~60
GPU-hours of measurements with no durable copy.

> **§6 landmine 16:** `src/arc/data/` was absent from every commit until 2026-10-15 (a
> `.gitignore` pattern ate it). The original has since been recovered from the compute
> box and committed verbatim, so the recorded results are re-derivable again and the
> package now has tests. Nothing is outstanding; the entry is kept because the failure
> mode will recur.

This document is the single entry point for anyone (human or agent) picking the project
up. Read it before touching code. `docs/ARC_SPEC.md` holds the engineering design;
`docs/RESULTS_LEVEL0.md` and `docs/RESULTS_LEVEL1.md` hold the headline tables;
`docs/EXPERIMENT_LOG.md` holds every run ever performed with exact numbers, including
the void ones and why each was discarded; this file holds the state, the landmines, and
the open decisions.

---

## 1. Access — read this first

### The compute box
`prashant@100.103.26.120` (`JPD-2AB-114-D02`), RTX 3080 10 GiB, i9-10850K / 20 threads,
61 GiB RAM, ~330 GiB free. Project lives at `~/arc`, venv at `~/arc/.venv`
(uv, Python 3.12, torch cu126).

### ⚠ A cloud session cannot reach the compute box

`100.103.26.120` is a **Tailscale** address (the `100.64.0.0/10` CGNAT range). It is
only routable from inside the tailnet. A Claude cloud session runs in an Anthropic
sandbox with no Tailscale node and no credentials, so `ssh prashant@100.103.26.120`
will simply time out. This is not a permissions prompt that can be approved — the host
is not addressable from there.

Three ways round it, in order of preference:

1. **Split the work.** Cloud session does code, tests, analysis, writing — all of which
   run on CPU and need no GPU. A local session (or you, by hand) launches GPU jobs. The
   repo is already structured for this: every experiment is a detached `nohup` script
   that survives disconnection, and `scripts/sync.sh` moves code up and results down.
2. **Join the cloud session to the tailnet.** Install Tailscale in the sandbox and
   authenticate with an auth key from the Tailscale admin console. Workable, but it
   means putting a credential into a sandbox — your call, and generate a short-lived,
   single-use, tagged key if you do.
3. **Expose the box another way** (public IP + SSH on a forwarded port, or a tunnel).
   Least preferred: it puts an SSH endpoint on the open internet.

Everything in section 4 that is marked *CPU* works fine in a cloud session today.

---

## 2. Current state in one page

| Level | What | State |
|---|---|---|
| 0 | Six tuned baselines, EuroSAT, 1/5/10/20-shot | **Done.** `docs/RESULTS_LEVEL0.md` |
| 0 | Five tuned baselines, ISIC, 1/5-shot | **Tuned on val; test not run.** `docs/RESULTS_ISIC.md` |
| 1 | Drift + utility signals, adaptive gate | **Built, tuned, null on EuroSAT *and* ISIC — and now diagnosed.** §5, `docs/RESULTS_SUBSPACE.md` |
| 2 | Topology, stability, novelty; learned gate; ablations | Not started |
| 3 | FBSDE controller | Not started (explicit stretch goal) |

**Datasets:** EuroSAT wired and used throughout. **ISIC now run** — baseline LR sweep
(172 configs) and gate sweep (27 configs x 2 shots) complete on val, ~8.5 GPU-hours;
see `docs/RESULTS_ISIC.md`. ChestX loader written and unit-tested, data not obtainable
automatically (§10).

**Test suite:** 110 tests, CPU, ~35 s. The six `easyfsl` ProtoNet cross-validation
tests need `pip install -e ".[dev]"`; without it they skip as a single entry and the
count reads 104.

**The forensic datasets (FFM/GFA/PCF) still do not exist as far as we know.** This was
flagged on day one in `docs/DATA_REQUEST.md` and has never been resolved. Everything
built so far is dataset-agnostic, so it is not blocking, but the application claim in
the paper has no data behind it yet.

---

## 3. Repo map

```
src/arc/
  backbones/tapped.py     ResNet-50 split into 17 gated units + frozen twin.
                          THE core mechanism. Read its docstring first.
  data/base.py            FewShotSource protocol, leakage-safe splits, episodic sampler
  data/sources.py         EuroSAT, ImageFolder, TensorSource, transforms
  data/pathlist.py        corpus from explicit (path, label, group) lists
  data/medical.py         ISIC + ChestX loaders
  heads/protonet.py       ProtoHead (cosine default) + linear probe
  train/episodic.py       EpisodeRunner: per-episode adaptation. Everything runs here.
  train/objectives.py     two-view / leave-one-out / (degenerate) self objectives
  train/augment.py        GPU random-resized-crop + flip
  train/baselines.py      the six Level-0 baselines
  signals/base.py         LayerSignals state vector + frozen_batchnorm guard
  signals/drift.py        D_l: 1 - linear CKA
  signals/utility.py      U_l: gradient agreement (NOT the spec's definition — §7)
  controllers/gate.py     AdaptiveGate (Controller A)
experiments/              runnable scripts, each with a module docstring
scripts/                  bootstrap_server.sh, sync.sh, run_level0.sh, download_data.sh
tests/                    82 tests; every one corresponds to a real defect
```

---

## 4. How to run things

```bash
# CPU, works in a cloud session
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python experiments/make_table.py results/level0

# code up to the box / results down   (needs tailnet)
bash scripts/sync.sh
bash scripts/sync.sh pull

# GPU jobs — always detached, always survive disconnect
ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup .venv/bin/python <script> \
    > logs/<name>.log 2>&1 < /dev/null &'
```

Conventions worth keeping:
- Tune on **val**, report on **test**. Never read test while tuning.
- `--split-seed` fixes the partition; `--seeds` varies only the episode draw.
- Every result JSON carries a manifest (git SHA, config, env).
- Long runs are resumable: `--skip-existing` skips completed cells.

---

## 5. Results

### Level 0 — EuroSAT, 5-way, tuned baselines (final)

```
baseline                1-shot        5-shot       10-shot       20-shot    %params
frozen_proto         67.67±0.47   84.59±0.29   87.28±0.32   88.99±0.27      0.00
last_layer_ft        67.70±0.47   84.80±0.29   89.11±0.30   91.99±0.24     18.40
bn_only_ft           68.46±0.49   86.83±0.29   90.79±0.27   93.61±0.20      0.22
stage4_ft            69.06±0.48   87.73±0.27   91.11±0.26   93.51±0.21     62.52
progressive_unfreeze 68.51±0.48   87.66±0.27   91.06±0.26   93.64±0.20     93.21
full_ft              68.37±0.48   87.21±0.28   91.35±0.26   94.02±0.20     94.34
```

1-/5-shot pool 3 seeds (1800 episodes/cell); 10-/20-shot pool 2 (1200/cell).

Three things that set the bar for ARC:
1. **Where you adapt dominates how much.** `last_layer_ft` touches 18% of weights for
   zero gain; `stage4_ft` touches 62% for +3.1.
2. **Uniform adaptation is not optimal in the scarce regime**, and the crossover is
   clean: `stage4_ft` beats `full_ft` at 5-shot and loses at 20-shot.
3. **`bn_only_ft` is the baseline to beat, not full fine-tuning.** Within ~0.5 points
   of the best method everywhere, at 0.22% of parameters. ARC's "adapt selectively,
   modify less" pitch already has a strong incumbent.

### Level 1 — the adaptive gate: null result

Tuned on val, 75 episodes/config, after fixing both the BatchNorm bug and the
saturation bug (§6). Marginal effect of each signal, averaged over the other
coefficients — the unbiased way to ask "does this signal help?":

```
             1-shot    5-shot
utility      -0.05     +0.04
drift        -0.10     -1.20
```

Best 5-shot config is `bias=2, aU=0, bD=0` — **no signals at all**, i.e. a constant
gate. 88.14 on val, identical to the signal-free best.

**Honest reading: the adaptive gate, correctly implemented and tuned, gives no
measurable benefit over a constant gate on EuroSAT, and drift as a gate input is mildly
harmful at 5-shot.**

Two caveats before concluding the idea is dead:
- EuroSAT is the easy end of the benchmark. A frozen backbone already gets 84.6% at
  5-shot, so every method sits within ~3 points and there is very little room for a
  controller to demonstrate anything. **ISIC and ChestX are where this should be
  tested** — published 5-shot accuracy there is ~45% and ~25%.
- The gate never reaches 0, so it modulates *how much* each layer adapts, not *whether*
  it does. `fraction_modified` stays at 84–98% in every configuration, so the
  parameter-efficiency win ARC needs against `bn_only_ft` is not being produced either.
  A hard/sparse gate (Gumbel-sigmoid, or an explicit sparsity penalty) is untested.

### ISIC — the null replicates and strengthens (2026-10-16)

Full numbers in `docs/RESULTS_ISIC.md`. The three things that change the picture:

**1. Headroom is real here.** The `lr=3e-6` column stands in for a frozen backbone at
~38.0 at 5-shot against a best tuned baseline of 49.55 — **11.5 points**, versus 3.1 on
EuroSAT. The reason for moving datasets was sound. (1-shot on ISIC spans only 27.6–30.9
and cannot separate methods; treat it as uninformative.)

**2. `bn_only_ft` ties `full_ft`: 49.44 vs 49.55 at 5-shot, 0.22% of parameters versus
~94%.** Its optimum needed a grid extension to 3e-2 — an order of magnitude above
EuroSAT's. ARC's "adapt selectively, modify less" pitch no longer has a strong incumbent;
it has an incumbent holding the ceiling at 1/400th of the cost.

**3. Both gate signals are harmful, more so than on EuroSAT:**

```
marginal effect      EuroSAT 1 / 5      ISIC 1 / 5
utility              -0.05 / +0.04      -0.18 / -1.43
drift                -0.10 / -1.20      -0.30 / -3.33
```

And `fraction_modified` is 97.7–98.4% across every ISIC 5-shot configuration — worse
than EuroSAT's 84–98%. No efficiency win is produced on either dataset.

**The reversal that matters.** On EuroSAT at 5-shot `stage4_ft` (87.73) beat `full_ft`
(87.21): adapting only the top beat adapting everything, which is the observation ARC's
premise rests on. On ISIC it reverses — `full_ft` 49.55 > `stage4_ft` 46.52 — while
`bn_only_ft`, which adapts BatchNorm in *every* unit rather than selecting by depth,
matches the best. The axis that pays on ISIC is **parameter type, not layer depth**. If
that survives test, ARC's layer-wise granularity may be the wrong granularity for this
kind of shift, and a controller over parameter groups (BN / conv / downsample) is the
more promising object. That is a redesign, and a question for him before more compute
goes into the current formulation.

**Interpretation worth putting to Prof. Hemrajani:** drift rises monotonically with
depth (measured: 0.000 at the stem to 0.33 at `layer4_2`). Penalising drift therefore
freezes the top of the network and adapts the stem — the opposite of both standard
transfer practice and our own strongest baseline (`stage4_ft` = adapt the top). The
methodology's sign convention on `D` may simply be wrong for accuracy, or drift may
belong in the loss as §8's `λ₁·D_repr` penalty rather than as a gate input. Neither has
been tested.

---

## 6. Landmines — bugs found, and why each mattered

Every one of these produced plausible-looking numbers while being wrong. They are
listed because the same classes of error will recur.

**Silent-numerics family**
1. **Metric saturation.** ProtoNet's squared-Euclidean over 2048-d ImageNet features
   gives distances of 5–419; at temperature 1 the softmax saturates to exactly 1.0 and
   cross-entropy underflows to **exactly 0.0** in fp32 — zero gradient at any learning
   rate, under any objective. Head now defaults to cosine. *Symptom: an entire LR sweep
   returned identical accuracy across a 300× LR range.*
2. **Degenerate inner objective.** Classifying the support set against prototypes built
   from that same set pins the correct-class logit at the metric maximum; at k=1 the
   point *is* its own prototype. Fixed with a two-view augmented objective.
3. **BatchNorm pollution (inner loop).** In train mode BN recomputes running statistics
   on the *forward* pass, independent of the optimiser. At `lr=0`: parameter drift 0.0,
   BN stat drift 162.1, accuracy falling 61.2% → 51.2% purely with step count.
4. **BatchNorm pollution (controller).** Same bug, different door: the controller
   restored mode with `.train()`, overriding `bn_mode='eval'`. It fired only when drift
   was enabled, which made **drift look harmful** (−9 points at 1-shot). Now
   `signals/base.py::frozen_batchnorm`.
5. **Gate saturation.** Raw signals (~0.1 magnitude) against a bias of 4 gave a gate of
   0.997 at *every* unit — spread 0.004. ARC was running as exactly full fine-tuning
   while appearing to work. Fixed by standardising signals across units.

**Experimental-validity family**

6. **`bn_only_ft` was full fine-tuning.** The runner re-applied its own unit selection
   after the caller narrowed to BN parameters. Two baselines were the same experiment
   under different names.
7. **Seed coupling.** One `--seed` drove both the data split and the episode sampler,
   so multi-seed runs would have evaluated on *different test sets*.
8. **Grid-edge truncation.** Several baselines peaked at the top of the LR grid, making
   the reported optimum a lower bound. `merge_sweeps.py` now re-checks this on the
   merged grid and the driver aborts if tuning is incomplete.
9. **Selection bias in "signals are worth +X".** Comparing max-over-18 signal-enabled
   configs against max-over-9 disabled ones favours the larger set by construction.
   Use marginal effects.
10. **Pipeline continued past a failed stage.** `set -uo pipefail` without `-e` let the
    merge and reporting stages run on a tuning file covering 7 of 10 cells; three
    baselines were silently reported untuned, and the run looked successful.

**Infrastructure family**

16. **`.gitignore` ate the entire data package.** `data/` is unanchored, so git matched
    it at *every* depth — including `src/arc/data/`. The package was therefore in no
    commit at all: a fresh clone could not `import arc.train.episodic`, every
    experiment script failed at its first import, and `pytest` collected none of the
    six test files that reach it. It survived only because both machines that ever ran
    the code had the files on disk already, untracked — and the box is a plain
    tar-synced tree with no git repo of its own, so nothing there would ever have
    flagged it. This is landmine 11 a second time through a different door, and it went
    unnoticed for the whole project.

    Fixed by anchoring the patterns (`/data/`, `/results/`, `/logs/`) and by recovering
    the original four modules from `~/arc/src/arc/data/` over the tailnet. They are
    committed **verbatim**, so every number in `docs/EXPERIMENT_LOG.md` is re-derivable.
    The package had no tests; it now has 28 (`tests/test_data.py`), covering the two
    guarantees nothing downstream can recover from — no specimen in two splits, and
    identical episodes across configurations — plus the stable-hash property that makes
    a split survive a reordering of the corpus.

    A reconstruction built from the test suite and call sites was committed first, while
    the box was unreachable, and is superseded. It differed in four ways that each
    would have moved the numbers, which is worth knowing as a measure of how little of
    this is inferable from an API: the eval transform (`Resize(size*1.14)` +
    `CenterCrop` versus a direct square resize), `group_disjoint` assignment (stable
    `blake2b` hash of `(seed, group)` versus an RNG permutation), class selection order
    within an episode, and whether a class too thin for `k_shot + n_query` raises or is
    silently dropped. Two of its judgement calls remain open and are logged in §9.6.

11. `--exclude data` in the sync tar matched `src/arc/data/`, shipping a package with a
    hole in it.
12. `print()` is block-buffered to a file while `logging` is not, so a healthy run looks
    stalled for hours. All experiment scripts now flush.
13. CRLF line endings from writing shell scripts on Windows → `$'\r': command not
    found`. `.gitattributes` forces LF.
14. `pkill -f run_level0.sh` matched the ssh wrapper issuing it, killing the session
    before the kill landed — a stale 11-hour job survived and shared the GPU with its
    replacement. **Kill by PID.**
15. Backslashes inside a shell heredoc get collapsed before Python sees them, so a
    patch searching for `\n` silently matches nothing. Edit files directly instead.

---

## 7. Deliberate departures from `ARC_SPEC.md`

These are design decisions, not drift. Each is documented at its definition site.

| Spec says | We do | Why |
|---|---|---|
| `U_l = ⟨∇L_support, ∇L_query⟩` | gradient agreement between two augmented draws of the support set | **Query labels are not available at test time.** Using them is label leakage that would inflate every ARC number invisibly. Spec's version kept as `oracle_utility()` for meta-training and as an upper bound. |
| squared-Euclidean ProtoNet head | cosine, learnable scale | Squared-Euclidean saturates to zero loss at 2048-d (§6.1). `sqeuclidean` retained, dimension-normalised, as an ablation. |
| gate consumes raw signals | signals standardised across units per update | Raw signals saturate the gate into a constant (§6.5). Raw behaviour kept behind `normalise=False`. |
| (unstated) group-disjoint splitting | the object partition is **stratified by class** | A single global shuffle leaves most classes out of a small split once a class has only a handful of physical objects — the projected shape of the forensic sets. Each object is assigned its modal class and still moves to exactly one split, so the no-object-spans-a-split guarantee is untouched. Introduced during the §6.16 reconstruction. |
| (unstated) episodes may share objects between support and query | same, by default; `EpisodicSampler(group_disjoint_queries=True)` separates them | The published cross-domain protocol does not separate them, and matching it is what keeps our numbers comparable. But on ISIC 45% of images share a lesion, so a support/query pair from one lesion makes an episode easier for reasons unrelated to adaptation. Available as a reported sensitivity check; see §9.5. |

---

## 8. What is valid, what is void

**Valid and reportable:** the entire Level-0 table; the baseline LR sweeps (merged grid,
222 configs, all interior optima); the ProtoNet cross-validation against easyfsl; the
ISIC loader verified against the real corpus.

**Void, kept only for provenance:**
- `results/sweep_INVALID_bn_pollution/` — first LR sweep, metric saturation + BN
- `results/gate_sweep_bnbug/` — first gate sweep, controller BN bug
- `results/gate_sweep_saturated/` — second gate sweep, gate saturated to a constant
- `results/discarded/` — baseline cells run at untuned or truncated settings

`results/gate_sweep/` is the current, valid gate sweep.

**Reproducibility caveat added 2026-10-15.** Everything above remains valid as a
*measurement* — it was produced by the code that was on the box at the time, and the
result JSONs carry their manifests. What is no longer guaranteed is *re-derivability
from this repository*, because the `arc.data` package those runs used was never
committed (§6.16) and the copy here is a reconstruction. Recovering
`~/arc/src/arc/data/` from the box and committing it closes this; until then, treat a
re-run that disagrees with the tables as a difference in the sampler, not a new
finding, and check the diff first.

---

## 9. Open decisions (need a human)

1. **Do the forensic datasets exist?** Unanswered since day one.
   `docs/DATA_REQUEST.md` is written to be forwarded to Prof. Hemrajani as-is. The
   specimen-structure question is the one that matters — ISIC turned out to have 45% of
   images sharing a lesion with another, which is exactly the hazard flagged there.
2. **Is the drift sign convention right?** See §5. This is his signal and his equation;
   the empirical structure suggests penalising drift freezes the wrong layers.
3. **Where does ARC get tested?** ISIC tuning is done and vindicated the move (11.5
   points of headroom vs 3.1). Open part: **should ISIC test reporting run now?** The
   recommendation in `docs/RESULTS_ISIC.md` §5 is yes for the baselines (~9-10 GPU-hours;
   "0.22% of parameters matches all of them on a hard target" is a contribution in its
   own right and the bar every ARC variant must clear) and **not yet for the gate**,
   which would buy a more precise measurement of a negative already held twice. ChestX
   still blocked on data (§10).
4. **Does Level 1 get rescued or reported as a negative?** (a) is now done and the
   answer is no — the null replicates on ISIC with the signals actively harmful. The
   remaining options, re-ordered by what the ISIC numbers imply rather than by cost:
   (c) **hard/sparse gate** — the only one attacking the axis ARC can still win on,
   since the accuracy ceiling is already held by `bn_only_ft` at 0.22% of parameters.
   Until the gate can reach zero, "modify less" has no mechanism, let alone evidence;
   (e) **a parameter-type controller instead of a layer controller** — new, implied by
   the §5 reversal, not suggested by the EuroSAT data;
   (b) **drift as a loss penalty** (`λ₁·D_repr`) — drift plainly carries signal, it is
   strong enough to cost 3.3 points when weighted; it looks like the wrong sign or the
   wrong place;
   (d) **invert the drift sign** — cheap, but ISIC weakens the prior: inverting pushes
   adaptation toward the high-drift top, which here is `stage4_ft`, and `stage4_ft`
   *loses* to `full_ft` on this corpus.
   A clean negative is publishable — "lightweight representation-feedback gating does
   not beat a tuned constant gate, and BatchNorm-only adaptation is a very strong
   baseline" is a useful result — but it is a different paper from the one proposed.

---

## 10. ChestX access

Not obtainable automatically. The NIH Box links every community script uses return 404
(verified 2026-10-12), as does the folder landing page. No Kaggle credentials on either
machine. `scripts/download_data.sh chestx` documents both routes and fails with
instructions rather than pretending. Cheapest path: a Kaggle API token at
`~/.kaggle/kaggle.json` on the box, then run that script (~42 GB).

---

## 11. Immediate next step if you want one command

ISIC is downloaded, verified, and has never been run. The whole Level-0 ladder plus the
gate can be pointed at it with the existing scripts:

```bash
ssh prashant@100.103.26.120 'cd ~/arc && setsid nohup .venv/bin/python \
  experiments/sweep_lr.py --dataset imagefolder --data-root data/isic2018 \
  --k-shot 1 5 --episodes 100 --out results/isic_sweep \
  > logs/isic_sweep.log 2>&1 < /dev/null &'
```

Note `run_baselines.py`/`sweep_lr.py` currently expose `eurosat` and `imagefolder`;
`run_gate.py` and `sweep_gate.py` already accept `--dataset isic` directly. Wiring
`isic` into the first two is a ~5-line change and is the obvious first task.
