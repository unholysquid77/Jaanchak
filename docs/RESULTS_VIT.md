# The ViT arm

**Validation split only. No test numbers here.** ViT-B/16 `augreg_in21k` (what SSF,
LN-Tune and DIPA report), 5-shot, 20 inner steps, `--no-amp` throughout, each
dataset on its own split fractions matching the ResNet arm. Driver:
`scripts/run_vit.sh`. Raw output under `results/vit_lr_*`.

Why this arm exists: `docs/RESULTS_SUBSPACE.md` establishes that on ResNet-50,
which parameters you adapt is nearly irrelevant — a 0.75-point band across a 778x
range of trainable parameters, replicated on ISIC and EuroSAT. That is a claim
about few-shot adaptation resting on one architecture. Four of five comparator
papers are ViT, and LN-Tune (Basu et al. 2023) is the direct analogue of the
`bn_only_ft` baseline the whole ResNet result turns on.

Two things are structurally better here, and both are reasons to trust this arm
over the ResNet one where they disagree:

* **LayerNorm has no running statistics.** The `bn_mode="train"` transductive-
  BatchNorm confound that every ResNet number carries does not exist on a ViT.
* **Attention and MLP are separable.** A ResNet cannot express that split at all,
  so it is the one genuinely new question rather than a port.

---

## 1. Stage 1: learning rates

Learning rates were swept first and alone, before any probe. The ResNet settings
are not transferable — and the size of the gap is the point. On ISIC with
ResNet-50, `bn_only_ft` wanted `3e-2` while `full_ft` wanted `1e-4`, a 300x spread
*within* one architecture. Running a displacement probe or a frontier on a
mis-tuned adaptation produces numbers that look like findings.

6-point grid, `1e-6` to `1e-1`, 50 episodes. **50 episodes, not 100, because this
stage picks a hyperparameter and does not report an accuracy.** Everything in this
section is a tuning artefact and gets re-measured at full episode count in stage 2.

### 1.1 ISIC

| lr | `full_ft` | `ln_only_ft` |
|---|---|---|
| 1e-6 | 41.20 ± 2.32 | 39.28 ± 1.87 |
| 1e-5 | 45.23 ± 2.11 | 39.15 ± 1.90 |
| **1e-4** | **53.07 ± 2.38** | 39.55 ± 1.97 |
| 1e-3 | 34.75 ± 2.06 | 42.03 ± 2.15 |
| 1e-2 | 29.76 ± 1.56 | 44.91 ± 2.17 |
| **1e-1** | 27.44 ± 1.64 | **49.36 ± 2.58** ← grid edge |

**`full_ft` is settled at `1e-4`.** A clean interior peak with a sharp collapse on
both sides: 53.07 at `1e-4` against 45.23 one decade below and 34.75 one decade
above. Nothing ambiguous.

**`ln_only_ft` is NOT settled and `1e-1` is not its answer.** It rises
monotonically across the whole grid and is still rising at the top. The reported
"best = 1e-1" is the largest value tried, not an optimum, and `sweep_lr.py`
emitted its grid-edge warning accordingly. Extending upward is the next run.

The shape is informative even unconverged: LayerNorm-only adaptation does
*nothing* below `1e-3` (39.28 → 39.55 across three decades, flat inside its own
confidence interval) and then climbs steeply. So the ResNet observation that the
normalisation-affine group wants a learning rate orders of magnitude above
`full_ft`'s reappears on ViT, and more extremely: `1e-4` against at least `1e-1`
is a factor of 1000 or more, against ResNet's 300.

Mechanically this is unsurprising — a per-channel affine has ~0.06% of the
parameters, so each one must move much further to change the function by the same
amount — but it is worth stating because it means **a parameter-type comparison
run at a single shared learning rate is not a comparison of parameter types.** It
is a comparison of how well each group happens to suit one arbitrary step size.
Any paper reporting LN-Tune against full fine-tuning at one lr is measuring the
wrong thing.

### 1.2 Interim: ViT-B/16 is better than ResNet-50 on ISIC

`full_ft` reaches 53.07 ± 2.38 here against ResNet-50's 49.6 — about 3.5 points,
roughly at the edge of the confidence interval and from 50 episodes, so treat it as
suggestive rather than measured until stage 2 re-runs it at 100.

### 1.3 EuroSAT — settled, and the plateau replicates

| lr | `full_ft` | `ln_only_ft` |
|---|---|---|
| 1e-6 | 78.69 ± 1.91 | 79.89 ± 1.93 |
| **1e-5** | **84.29 ± 1.69** | 79.81 ± 1.93 |
| 1e-4 | 81.15 ± 1.92 | 79.68 ± 1.90 |
| 1e-3 | 49.25 ± 2.89 | 79.79 ± 1.80 |
| **1e-2** | 40.13 ± 1.91 | **84.32 ± 1.69** |
| 1e-1 | 39.47 ± 2.01 | 71.89 ± 2.90 |

Both are **clean interior peaks** and both are settled: `full_ft` at `1e-5`,
`ln_only_ft` at `1e-2` with `1e-1` collapsing to 71.89. No grid-edge warning.

**`ln_only_ft` ties `full_ft`: 84.32 against 84.29.** A 0.03-point difference, far
inside either confidence interval, with **0.06% of the parameters** against 100%.
This is the ResNet plateau reproducing on a transformer — and reproducing on the
dataset where the ResNet arm found it hardest (EuroSAT's parameter-type band was
0.75 points across a 778x range).

So the plateau is not an artefact of BatchNorm, and specifically not of transductive
BatchNorm statistics: LayerNorm has no running statistics, so the `bn_mode`
confound that every ResNet number carries cannot explain this. That was the main
methodological reason to build this arm and it has paid off.

The step-size spread also replicates and is now confirmed on two datasets:
`ln_only_ft` wants `1e-2` where `full_ft` wants `1e-5`, a factor of **1000**,
matching the ≥1000x seen on ISIC. On ResNet the same comparison was 300x.

### 1.4 ISIC — settled too, and it does NOT tie

Stage 1b extended `ln_only_ft` upward on ISIC. `1e-1` turns out to be the real
peak, now bracketed on both sides:

| lr | 1e-3 | 1e-2 | **1e-1** | 3e-1 | 1e0 | 3e0 | 1e1 |
|---|---|---|---|---|---|---|---|
| `ln_only_ft` | 42.03 | 44.91 | **49.36 ± 2.58** | 29.97 | 28.45 | 26.51 | 23.12 |

The collapse above the peak is abrupt — 49.36 to 29.97 across a factor of 3 — so
the optimum is sharp rather than a plateau in lr. Both ISIC learning rates are now
interior and settled: `full_ft` `1e-4`, `ln_only_ft` `1e-1`. **Exactly 1000x apart,
and exactly 1000x apart on EuroSAT too** (`1e-5` / `1e-2`). Two datasets agreeing
on that ratio to one significant figure is more than I expected from a 6-point
logarithmic grid.

**The headline comparison, both datasets, both groups at their own tuned lr:**

| | `full_ft` (100%) | `ln_only_ft` (0.06%) | gap |
|---|---|---|---|
| EuroSAT | 84.29 ± 1.69 | 84.32 ± 1.69 | **−0.03** |
| ISIC | 53.07 ± 2.38 | 49.36 ± 2.58 | **+3.71** |

EuroSAT is a clean tie. **ISIC is not.** The 3.71-point gap against a combined 95%
interval of ±3.51 (√(2.38² + 2.58²)) excludes zero — but *barely*, by 0.2 points,
at 50 episodes. So this is the second of the two outcomes §1.4 previously laid out,
and it is suggestive rather than established:

> the plateau holds under mild shift (EuroSAT, ~3-point headroom) but weakens or
> breaks under strong shift (ISIC, ~14-point headroom on ViT).

That rhymes with §6.2 of `RESULTS_SUBSPACE.md`, where on ResNet the extra capacity
bought 3.5 points on ISIC and nothing at all on EuroSAT. Two independent
experiments — a basis-rank frontier and a parameter-type comparison, different
architectures — pointing at the same shift-magnitude interaction is the most
interesting thing in this project so far.

It is also exactly the kind of margin that has burned this project three times, and
0.2 points past a confidence interval is not a finding. Stage 2b re-measures at 100
episodes, which halves the interval if the accuracies hold, and that is what
decides it.
---

## 2. What is NOT claimed yet

Explicitly, so none of this gets quoted as a result:

* **The plateau is confirmed on EuroSAT only, and only for one group.**
  `ln_only_ft` ties `full_ft` there (84.32 / 84.29), which is real and is an
  interior peak on both sides. But that is one of six groups in the partition at a
  tuning episode count; `ln_scale`, `ln_shift`, `attn`, `mlp` and `patch_embed` are
  not run, so "every group plateaus" is not established on ViT. On ISIC nothing is
  established: `ln_only_ft` is unconverged and currently 3.7 points behind, with
  overlapping intervals (±2.38, ±2.58).
* **No attention-vs-MLP result.** Not run.
* **No displacement or frontier numbers.** Not run. When they are, the code's
  notion of a "diagonal" correction has to be checked against LayerNorm affine
  rather than BatchNorm affine before any of it is trusted.

## 3. Standing cautions inherited from the ResNet arm

Three of the first four results in this project were single-draw artefacts, and
`RESULTS_SUBSPACE.md` §7 establishes a ±1.1-point estimator noise floor on
anything derived from an extracted displacement. Before any number here is
written up: check how many draws it rests on, whether its margin clears that
floor, and that the selector opened the parameter set it was supposed to. The last
of those has already failed twice — `bn_only_ft` once ran as full fine-tuning for
a whole sweep, and `last_layer_ft` resolved on ViT to the 1.5k-parameter pre-head
LayerNorm, which would have been a near-frozen baseline reporting a plausible
accuracy. Both are now asserted in `tests/test_selectors_vit.py`.
