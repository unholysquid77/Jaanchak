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

## 2. Stage 2a: the parameter-type partition

Every group at **its own** tuned learning rate, 50 episodes, every peak now
bracketed on both sides. The partition is exact — LayerNorm + attention + MLP +
patch-embedding = 85,798,656 parameters, each owned once, asserted by parameter name
in `tests/test_selectors_vit.py`.

| group | params | % | lr | val acc |
|---|---|---|---|---|
| `full_ft` | 85,798,656 | 100.000 | 1e-4 | **53.07 ± 2.38** |
| `ln_scale_ft` | 19,200 | 0.022 | 1e-1 | 49.95 ± 2.37 |
| `ln_shift_ft` | 19,200 | 0.022 | 1e-1 | 49.92 ± 2.44 |
| `attn_only_ft` | 28,348,416 | 33.041 | 3e-4 | 49.65 ± 2.24 |
| `mlp_only_ft` | 56,669,184 | 66.049 | 1e-4 | 49.65 ± 2.28 |
| `ln_only_ft` | 38,400 | 0.045 | 1e-1 | 49.36 ± 2.58 |
| `patch_embed_ft` | 742,656 | 0.866 | 1e-3 | 46.24 ± 2.23 |

### 2.1 A band, and `full_ft` sitting above all of it

Five of the six groups land in **49.36–49.95**, a 0.59-point band, across a
**2,951× range** in trainable parameters (19,200 to 56,669,184). That part is the
ResNet plateau reproducing: parameter count buys almost nothing, and neither does
parameter kind.

What does **not** reproduce is where `full_ft` sits. On ResNet-50 it was *inside* the
band. Here it is **3.12 points above the top of it** (53.07 against 49.95) and 3.71
above the bottom. So on ISIC with a ViT there appears to be a ceiling that no proper
subset of the parameter-type partition passes — not a parameter-count ceiling, since
`mlp_only_ft` has two thirds of the backbone open and lands with the 19,200-parameter
LayerNorm scale group. If that holds, the gain is coming from parameters of
*different types moving together*, which is a different claim from anything in the
ResNet arm.

Three reasons this is not stated as a result yet. It is 50 episodes, the tuning
count. The 3.12-point margin against `ln_scale_ft`'s ±2.37 and `full_ft`'s ±2.38
(combined ±3.36) does **not** clear zero — only the comparison against `ln_only_ft`
does, and that one by 0.2 points. And the whole shape rests on one draw. Stage 2b at
100 episodes is what decides it.

### 2.2 Attention against MLP: the comparison a ResNet cannot express

A ResNet has no attention, so this split exists only on this arm. It is also the one
place where the fraction-of-backbone argument predicts a visible difference: `mlp` has
exactly twice `attn`'s parameters (66.0% against 33.0%).

The difference is **zero**. 49.65 ± 2.28 against 49.65 ± 2.24 — the same 1862 of 3750
correct predictions, which is a coincidence of counts rather than evidence of
identical behaviour, but the point survives rounding either way: doubling the
parameter budget and changing which sublayer moves does nothing measurable.

Their learning rates differ by a factor of 3 (`attn` 3e-4, `mlp` 1e-4), in the
direction fraction-of-backbone predicts — the smaller group wants the larger step —
so the two were tuned apart and still landed together.

### 2.3 Patch embedding is the one group below the band

46.24 ± 2.23, a clear 3.1 points below `ln_only_ft` and the only group outside the
band. Its peak is interior, so this is
not a tuning artefact: the grid is 42.21 / 44.51 / **46.24** / 41.44 / 36.45 across
1e-5 to 1e-1.

It is also the only group that cannot change how tokens *interact*, only how pixels
become tokens. Suggestive, and left as that: one dataset, one draw, and EuroSAT's
`patch_embed_ft` behaves differently enough (peaking at the smallest step tried,
i.e. closest to frozen) that a single story does not yet cover both.

### 2.4 What settled the question this section was waiting on

`attn_only_ft` peaked at 3e-4, the top of its original grid, which left open the
possibility that attention alone reaches `full_ft` and the ceiling story is wrong.
The upward extension closed it: above 3e-4 attention collapses hard — 49.65 at 3e-4,
then **35.60** at 1e-3, 32.40 at 3e-3, 28.61 at 1e-2. `mlp` does the same from 1e-4
(49.65 → 42.08 → 33.47 → 25.23). Both peaks are now interior, so the band is real
rather than an artefact of grids that stopped too early.

### 2.5 The same table on EuroSAT, and the ceiling does **not** reproduce

Same protocol, same partition, every group at its own tuned learning rate, every
peak bracketed on both sides.

| group | % params | lr | val acc |
|---|---|---|---|
| `mlp_only_ft` | 66.049 | 1e-4 | **85.57 ± 1.55** |
| `ln_scale_ft` | 0.022 | 3e-2 | 85.20 ± 1.73 |
| `attn_only_ft` | 33.041 | 1e-4 | 84.96 ± 1.75 |
| `ln_only_ft` | 0.045 | 1e-2 | 84.32 ± 1.69 |
| `full_ft` | 100.000 | 1e-5 | 84.29 ± 1.69 |
| `ln_shift_ft` | 0.022 | 3e-2 | 83.79 ± 1.94 |
| `patch_embed_ft` | 0.866 | 1e-6 | 80.13 ± 1.95 |

`patch_embed_ft` needed a downward extension, which closed it: **79.92 at both
1e-8 and 1e-7**, identical to two decimals because at that step size nothing moves.
So its 80.13 peak at 1e-6 is interior, and it is 0.21 points above doing nothing at
all. Adapting the patch embedding on EuroSAT is not weak adaptation; it is no
adaptation, and `frozen_proto` from stage 2b should land on roughly the same number.

**`full_ft` is fourth of seven.** Three groups sit above it, including the
19,200-parameter `ln_scale_ft` at 85.20. The whole six-group band spans 1.78 points
(83.79–85.57) against intervals of ±1.6 to ±1.9, so the honest reading is that none
of these groups is distinguishable from any other — `full_ft` included. That is the
ResNet result reproducing exactly, on a transformer, with LayerNorm in place of
BatchNorm and therefore without the transductive-statistics confound.

So §2.1's ceiling is **not** architectural. It appears on ISIC and not on EuroSAT,
which makes it a property of the shift rather than of the backbone:

| | band width | `full_ft` vs band |
|---|---|---|
| EuroSAT | 1.78 | **inside** (4th of 7) |
| ISIC | 0.59 | **3.12 above** |

The natural way to describe the difference is headroom — how much there is to gain
over a frozen backbone — but **this arm has no `frozen_proto` number yet**, on either
dataset, so the headroom figures quoted in §1.4 are carried over from the ResNet arm
and are not measurements of this one. Stage 2b runs `frozen_proto` as the floor
bracket on both datasets, which is what makes "little to gain" and "a lot to gain"
quantities rather than adjectives.

This is the third independent experiment in this project to land on the same
shift-magnitude interaction. `RESULTS_SUBSPACE.md` §6.2 found extra basis capacity
bought 3.5 points on ISIC and nothing on EuroSAT, on ResNet-50. §1.4 here found
`ln_only_ft` ties `full_ft` on EuroSAT and not on ISIC, on ViT. Now the full
parameter-type partition says the same thing a third way. Different architectures,
different interventions, one pattern: **where there is little to gain, every
parameter group gets all of it; where there is a lot, only joint movement across
types does.**

That is a real claim and it is the one worth building the paper around, but all
three legs are at 50-episode tuning counts or single draws. Stage 2b is what turns
the third leg into a measurement.

### 2.6 Attention and MLP tie on EuroSAT too

84.96 ± 1.75 against 85.57 ± 1.55 — a 0.61-point difference against a combined
±2.33. Indistinguishable, same as ISIC's zero. Two datasets, both saying that
doubling the parameter budget and swapping which sublayer moves changes nothing
measurable. Both learning rates landed on 1e-4 here, so unlike ISIC they were not
even tuned apart.

---

## 3. What is NOT claimed yet

Explicitly, so none of this gets quoted as a result:

* **Every accuracy in §1 and §2 is at 50 episodes, the tuning count.** Nothing
  here is a reported number. Stage 2b re-measures the whole partition at 100
  episodes with `frozen_proto` as the floor bracket, and the §2.1 ceiling in
  particular needs it: the 3.12-point margin between `full_ft` and the top of the
  band does not clear the combined interval (±3.36) at this episode count.
* **§2.5's shift-magnitude claim rests on two datasets.** Two points do not
  establish a monotone relationship between shift magnitude and whether joint
  movement matters, and "headroom" here is a proxy for shift magnitude, not a
  measurement of it. ChestX, the third and largest shift, is unobtainable
  (`RESULTS_SUBSPACE.md` §8), so this may not be resolvable on BSCD-FSL at all.
* **§2.1's ceiling has no mechanism behind it.** "Gain comes from different
  parameter types moving together" is the shape of the data, not something measured.
  Nothing here separates it from an optimisation story (seven adapters tuned
  separately, one tuned jointly) and the project has already had one mechanism die
  for exactly that reason (`RESULTS_SUBSPACE.md` §6).
* **§2.2's attention/MLP tie is one draw.** Two groups landing on the same accuracy
  to four digits is a coincidence of integer counts, not a measurement of
  equivalence; the claim that survives is "indistinguishable at ±2.3", and only on
  ISIC.
* **No ViT seed spread.** `RESULTS_SUBSPACE.md` §7 cost three draws to establish a
  ±1.1-point floor on the ResNet arm. Nothing equivalent exists here, so every
  margin in this document is being read against a within-run confidence interval
  that may understate the real noise.
* **No displacement or frontier numbers.** Not run. When they are, the code's
  notion of a "diagonal" correction has to be checked against LayerNorm affine
  rather than BatchNorm affine before any of it is trusted.

## 4. Standing cautions inherited from the ResNet arm

Three of the first four results in this project were single-draw artefacts, and
`RESULTS_SUBSPACE.md` §7 establishes a ±1.1-point estimator noise floor on
anything derived from an extracted displacement. Before any number here is
written up: check how many draws it rests on, whether its margin clears that
floor, and that the selector opened the parameter set it was supposed to. The last
of those has already failed twice — `bn_only_ft` once ran as full fine-tuning for
a whole sweep, and `last_layer_ft` resolved on ViT to the 1.5k-parameter pre-head
LayerNorm, which would have been a near-frozen baseline reporting a plausible
accuracy. Both are now asserted in `tests/test_selectors_vit.py`.
