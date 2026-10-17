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

### 1.3 EuroSAT

In progress at time of writing; `full_ft` peaks at `1e-5` (84.29 ± 1.69), an
interior peak, with `1e-6` at 78.69 and `1e-4` at 81.15. `ln_only_ft` is flat and
mildly *decreasing* across `1e-6`–`1e-4` (79.89 / 79.81 / 79.68), consistent with
ISIC's finding that LayerNorm does nothing at small step sizes. The three
high-lr configs are what matter and are not in yet.

---

## 2. What is NOT claimed yet

Explicitly, so none of this gets quoted as a result:

* **No parameter-type plateau claim.** The headline ResNet finding is that every
  group ties within 0.75 points. On ISIC, `ln_only_ft` at its (unconverged) best
  sits 3.7 points below `full_ft`, which would be a *departure* from the plateau —
  but the grid is not settled, 50 episodes is a tuning budget, and the two
  confidence intervals (±2.38, ±2.58) overlap at that margin. This is not evidence
  of anything yet, in either direction.
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
