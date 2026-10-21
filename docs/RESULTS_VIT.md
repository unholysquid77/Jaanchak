# The ViT arm

**Validation split only. No test numbers here.** ViT-B/16 `augreg_in21k`
(**CORRECTED 2026-10-20**: this was described here and in the draft as "what SSF,
LN-Tune and DIPA report". That is false. Checked against the papers: SSF (Lian et al.,
NeurIPS 2022) does use a ViT-B/16 ImageNet-21k timm checkpoint but evaluates on FGVC
and VTAB-1k, which are not episodic few-shot at all; LN-Tune (Basu et al., AAAI 2024)
reports on Meta-Dataset and ORBIT; DIPA (Perera & Halgamuge, CVPR 2024) uses
**ViT-S/16 with DINO self-supervised pretraining**, a different size *and* a different
pretraining objective. No comparator shares both our checkpoint and our protocol. The
checkpoint choice is still fine -- a strong, standard, publicly available supervised
IN-21k ViT-B/16 -- but it is *our* choice, not an inherited comparability guarantee,
and no number of theirs is directly comparable to ours. See §4.), 5-shot, 20 inner steps, `--no-amp` throughout, each
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
* **Depth is a clean axis.** Twelve blocks, identical in shape and parameter count,
  so "adapt block *i* and nothing else" varies position with size held exactly
  fixed. A ResNet's stages differ in width, depth and resolution, so the same
  question there cannot be asked without confounding position with size. §3.3.2
  turns out to be where the largest effect on this arm lives, and it is an
  experiment the ResNet arm structurally could not run.

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

> **Every accuracy in this section is at 50 episodes and exists to pick a learning
> rate.** §3 re-measures the whole partition at 100 with both brackets, and on
> EuroSAT it does not merely tighten this section's reading, it changes it. Read
> §3 for the numbers; read this section for how the learning rates were settled
> and which grids had to be extended.


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

## 3. Stage 2b: the measured tables

100 episodes, every group at its own merged best learning rate, validation split,
both brackets present. No group was dropped as grid-edge — every peak in §2 is
interior on the merged grid, so every row below is a tuned measurement rather than
a bound.

`frac_mod` is the fraction of backbone parameters an episode actually changed,
reported per episode by `run_baselines.py`. It is an independent check that each
selector opened what it was supposed to, and this project has twice shipped a
baseline that silently selected the wrong set, so it is quoted rather than trusted.

### 3.1 ISIC

| group | lr | val acc | gain over frozen | frac_mod |
|---|---|---|---|---|
| **`full_ft`** | 1e-4 | **52.49 ± 1.50** | **+13.14** | 97.35% |
| `attn_only_ft` | 3e-4 | 50.20 ± 1.62 | +10.85 | 32.74% |
| `ln_shift_ft` | 1e-1 | 50.05 ± 1.72 | +10.70 | 0.0224% |
| `mlp_only_ft` | 1e-4 | 50.01 ± 1.31 | +10.66 | 64.46% |
| `ln_only_ft` | 1e-1 | 49.60 ± 1.76 | +10.25 | 0.0447% |
| `ln_scale_ft` | 1e-1 | 49.28 ± 1.43 | +9.93 | 0.0223% |
| `patch_embed_ft` | 1e-3 | 45.95 ± 1.50 | +6.60 | 0.8616% |
| `frozen_proto` | — | 39.35 ± 1.24 | — | 0.0000% |

`full_ft` is top, and **every single parameter type reaches 76–83% of its gain**
(`ln_scale` 75.6%, `attn` 82.6%) — including the 19,200-parameter LayerNorm scale
group, which is 0.022% of the backbone.

**The margin shrank.** At 50 episodes `full_ft` sat 3.12 above the band's top
against a combined ±3.36, which did not clear zero. At 100 it sits **2.29 above
`attn` against a combined ±2.21** — clearing by 0.08, which is on the line, not
settled. Both halves moved: `full_ft` 53.07 → 52.49 while the band drifted up
(`attn` 49.65 → 50.20, `ln_shift` 49.92 → 50.05). Doubling the episode count
narrowed the gap roughly as much as it narrowed the interval.

> **Superseded by §3.4.** The combined-interval test used in this paragraph treats
> the two accuracies as independent estimates. They share their episodes, so the
> correct test is the paired per-seed difference — and across three draws
> `full_ft` − `attn` is +2.29 / +2.07 / +2.25, mean **+2.20 ± 0.30**. The
> comparison is settled; it was the statistic that was wrong, not the data.

### 3.2 EuroSAT — `full_ft` is not just inside the band, it **loses**

| group | lr | val acc | gain over frozen | vs `full_ft`'s gain | frac_mod |
|---|---|---|---|---|---|
| **`attn_only_ft`** | 1e-4 | **85.96 ± 1.06** | **+6.69** | **168%** | 32.33% |
| `mlp_only_ft` | 1e-4 | 84.76 ± 1.29 | +5.49 | 138% | 64.57% |
| `ln_only_ft` | 1e-2 | 84.75 ± 1.15 | +5.48 | 138% | 0.0443% |
| `ln_scale_ft` | 3e-2 | 84.51 ± 1.35 | +5.24 | 132% | 0.0223% |
| `full_ft` | 1e-5 | 83.24 ± 1.31 | +3.97 | 100% | 83.28% |
| `ln_shift_ft` | 3e-2 | 82.92 ± 1.31 | +3.65 | 92% | 0.0223% |
| `patch_embed_ft` | 1e-6 | 79.44 ± 1.44 | +0.17 | 4% | 0.3351% |
| `frozen_proto` | — | 79.27 ± 1.42 | — | — | 0.0000% |

`full_ft` is **fifth of seven**, and `attn_only_ft` beats it by **2.72** — a margin
that §3.4 confirms on all three draws (mean **+2.50 ± 0.65**; the combined-interval
figure of ±1.69 quoted here is the wrong test for paired data and understates it).
That is not a tie and not a plateau: on
EuroSAT, adapting attention alone is *significantly better* than fine-tuning the
whole backbone, with a third of the parameters. Four of the six single types beat
`full_ft` outright.

The 50-episode reading in §2.5 ("fourth of seven, nothing distinguishable from
anything") was the weaker version of this. Tighter intervals did not flatten the
table, they resolved it.

`patch_embed_ft` lands 0.17 above `frozen_proto`, confirming §2.5 exactly: on
EuroSAT, adapting the patch embedding is not weak adaptation, it is none. Its
`frac_mod` of 0.3351% against ISIC's 0.8616% says the same thing mechanically —
at 1e-6 most of its parameters do not move at float precision. On ISIC the same
group reaches 50% of headroom, so the two datasets genuinely differ here.

### 3.3 Where you adapt beats which parameters you adapt

This section used to be called "Attention against MLP". That comparison is still
here and still holds, but it is no longer the finding — it is one coarse slice
through a bigger effect, and the slice averages the effect away.

The finding is this. Take five interventions that are *identical in size and in
parameter type* — attention in exactly one transformer block, 2.75% of the backbone,
2.4M parameters — and vary only **which** block. On EuroSAT they span

> **78.80 at block 0 to 87.69 at block 8: a range of 8.89 points.**

For comparison, the entire parameter-*type* axis — patch-embed, LayerNorm,
attention, MLP, everything — spans 6.5 points, and the headline `attn` − `mlp` gap
inside it is 1.18. **Depth is a larger axis than type, by a factor of about seven
against the type comparison and still clearly larger than the whole type range.**
Both endpoints are now interior optima on their own grids, so this is a measurement
rather than a bound (§3.3.2 point 4).

**The organising fact is a depth effect with a type interaction on top of it, not
attention.** §3.3.6 runs the identical sweep in the block MLP and gets the same
shape — flat shallow region, steep rise, interior peak at block 8, fall at 11 —
with a comparable span (8.21 against 8.89) and a *better* best cell (88.91, the
highest number in this document, from 5.5% of the backbone). So "attention is the
active ingredient", which earlier versions of this section claimed, is **retracted**;
see §3.3.4 reading (d). What generalises is that **one well-placed block of either
type beats every larger configuration, and a badly placed block of either type does
not**. Type still matters once depth is fixed: MLP is ahead at every depth measured
and settledly behind at none, by a margin that varies 3.6× across depth (§3.3.6b).
An earlier reading had that margin *changing sign* at block 11; three draws demoted
it to unsettled, so the interaction is in the magnitude and not the sign.

**And read §3.3.5 before quoting any of it.** The same sweep on ISIC has depth
mattering there too and pointing the *other* way — the shallow half is best, the deep
end is worst, and no single block beats `full_ft` at all. So the part of this that is
specific to EuroSAT is the part where a 2.75% slice beats everything bigger; the part
that generalises is that *where* you adapt is a first-order choice. The two curves
are roughly mirrored, which makes depth a lens on the shift rather than a property of
the backbone.
Every number that follows in §3.3.1 and §3.3.3 is a quantity averaged over depth,
and §3.3.2 is why that averaging is the thing to be suspicious of.

§3.3.4 records three readings of this data that I published and then retracted. They
are kept because the sequence is informative about the failure mode, not because the
earlier readings are live.

**A note on the test used in this section.** §3.4 establishes that the valid test
for a multi-draw comparison is the per-seed difference, never within-run intervals
combined in quadrature. §3.3.2 and §3.3.3 are **one draw each** — there is no
per-seed difference to take. For those I report the *per-episode paired* difference
instead (same seed, hence literally the same 100 episodes, so the pairing is exact),
which is the right test for "did these two differ on these episodes" but does **not**
cover run-to-run nondeterminism. §4 records that floor at ~0.6 points between
byte-identical reruns. So a single-draw gap is only claimed here when it clears
*both* its own paired interval and that 0.6 floor with room. Gaps between those two
bars are written as suggestive and nothing more.

### 3.3.1 The type partition: attention beats MLP, with half the parameters

This is the comparison a ResNet cannot express — it has no attention — so it is the
one genuinely new question on this arm rather than a port. `mlp` has exactly twice
`attn`'s parameters (66.0% of the backbone against 33.0%).

Three draws each, paired per seed, per §3.4:

| | seed 0 | seed 1 | seed 2 | mean | sd | 95% CI (n=3) |
|---|---|---|---|---|---|---|
| EuroSAT `attn` − `mlp` | +1.20 | +1.00 | +1.35 | **+1.18** | 0.17 | **[+0.76, +1.60]** |
| ISIC `attn` − `mlp` | +0.19 | +0.84 | +0.28 | +0.44 | 0.35 | [−0.43, +1.31] |

**On EuroSAT attention beats MLP, with half the parameters.** Every draw positive,
interval well clear of zero. On ISIC the two are indistinguishable.

> **What this comparison is actually averaging over, per §3.3.6.** Whole-type
> adaptation touches all twelve blocks, so `attn − mlp` moves type *and* the depth
> distribution together. Measured depth by depth over **three draws** (§3.3.6b),
> MLP − attention runs **+1.51, +2.03, +4.66, +1.29, −0.96** at blocks 0, 2, 5, 8,
> 11 — settled and positive at three of them, unsettled at the other two, and
> **settledly negative at none**. So whole-type attention beats whole-type MLP while
> attention never demonstrably beats MLP at any single depth. The resolution is that
> whole-type MLP adapts 66% of the backbone against attention's 33% and so suffers
> more from §3.3.3's interference term; that penalty is what this average measures.
> So **`attn − mlp` is a fact about adapting all twelve blocks at once and carries no
> information about which type is better to adapt at a given depth.** It is the last of the four averaging errors §3.3.4 collects, and the
> only one caught before it was promoted to a claim about mechanism.

The whole type axis on EuroSAT, three-draw means where available:

| group | % of backbone | EuroSAT |
|---|---|---|
| `frozen_proto` | 0 | 79.22 |
| `patch_embed_ft` | 0.87 | 79.44 (1 draw) |
| `ln_only_ft` | 0.045 | 84.18 |
| `mlp_only_ft` | 66.0 | 84.38 |
| **`attn_only_ft`** | **33.0** | **85.56** |
| `full_ft` | 100 | 83.07 |

Two things to take from it, and only two. Attention is the best type; and 0.045% of
the backbone (LayerNorm affine) already gets within 1.4 points of the best, so
whatever the useful adaptation is, it is *small*. The ordering is **not** a capacity
curve — see §3.3.4, reading (b).

### 3.3.2 Position: the dominant axis

`select_attn_block(i)` adapts attention in block *i* alone. The tests in
`tests/test_selectors_vit.py` assert that the parameter count is identical at every
depth — load-bearing here, because otherwise an accuracy difference could be about
size rather than position — and that `select_attn_block(11)` is by name the same
selector as `select_attn_last_n(1)`, so the n=1 row of §3.3.3 *is* block 11.

EuroSAT, seed 0, 100 episodes, 20 inner steps, each block at its own tuned learning
rate. `scripts/run_vit_position.sh`.

| block | lr | EuroSAT | frac_mod |
|---|---|---|---|
| 0 | 3e-5 **(bottom grid edge)** | 78.80 | 2.636 |
| 2 | 3e-4 | 78.87 | 2.732 |
| 5 | 3e-4 | 82.04 | 2.736 |
| **8** | 3e-3 | **87.69** | 2.751 |
| 11 | 3e-4 | 86.51 | 2.743 |
| — `frozen_proto` | — | 79.27 | 0.000 |
| — `full_ft` | 1e-5 | 83.24 | 83.279 |

`frac_mod` confirms the sizes match to within a tenth of a point across all five
rows; block 0 reads slightly low because at 3e-5 a few weights do not move far
enough to change their float, which is itself a hint that its grid is mispitched.

Per-episode paired differences, all on the same 100 episodes:

| comparison | difference | clears 0.6 floor? |
|---|---|---|
| block 8 − `frozen_proto` | **+8.43 ± 1.22** | yes |
| block 8 − block 5 | +5.65 ± 1.40 | yes |
| block 8 − `full_ft` | **+4.45 ± 1.24** | yes |
| block 11 − block 5 | +4.47 ± 1.29 | yes |
| block 5 − block 2 | +3.17 ± 1.19 | yes |
| block 8 − `mlp_only_ft` | +2.93 ± 1.29 | yes |
| block 5 − `frozen_proto` | +2.77 ± 1.20 | yes |
| block 8 − `attn_only_ft` (all 12 blocks) | **+1.73 ± 1.07** | yes, but 2.9× floor |
| block 8 − block 11 | +1.19 ± 1.11 | **marginal both ways** |
| block 8 − `attn_last6_ft` (contains block 8) | +1.00 ± 0.96 | **marginal both ways** |
| block 2 − `frozen_proto` | −0.40 ± 1.29 | **not settled** |
| block 0 − `frozen_proto` | −0.47 ± 0.70 | **not settled** |

Read in order:

1. **Depth rises steeply and then falls: the peak is interior.** 78.8, 78.9, 82.0,
   87.7, 86.5. The rise is enormous — block 8 beats block 5 by 5.65 ± 1.40 and block
   5 beats block 2 by 3.17 ± 1.19. The *fall* from block 8 to block 11 was the one
   marginal link in the chain at one draw (+1.19 ± 1.11, twice the floor); **§3.3.2a
   settles it at three draws: +1.54 ± 0.80, every draw positive.** So the inverted U
   in depth is established, not suggested. Note that closing block 8's grid edge
   (point 4) had moved this gap *down* from +1.73 to +1.19 — the edge had been
   inflating it — and the honest three-draw value sits between the two.
2. **Early-block attention is indistinguishable from doing nothing.** Blocks 0 and 2
   sit within half a point of `frozen_proto` and neither difference is settled. An
   earlier commit message on this data said early blocks are *below* frozen; that
   was an overclaim read off the point estimates, and the paired test does not
   support it. What is supported is the weaker and still interesting statement:
   2.4M adaptable parameters buy *nothing* at the bottom of the stack and ~9 points
   at three-quarters height.
3. **One well-placed block beats every other configuration measured on this arm.**
   Block 8 beats all of attention (+1.73 ± 1.07), MLP (+2.93 ± 1.29), full
   fine-tuning (+4.45 ± 1.24) and frozen (+8.43 ± 1.22). The best ViT number
   anywhere in this document is a 2.75% intervention. The margin against `full_ft`
   is the robust one; the margin against *all of attention* is real but only three
   times the floor at one draw, so the specific claim "less attention beats more
   attention" rests on §3.3.2a rather than on this row alone.
4. **Block 8's grid edge is closed, and closing it cost half a point.** The first
   grid ran 3e-5 → 1e-3 and peaked on 1e-3, the top, with the tuning curve still
   climbing (80.29, 83.09, 86.72, 87.49 on the 50-episode tuning subset). The
   extension (`scripts/run_vit_position_hi.sh`, `results/vit_pos8_lr_hi`) added 3e-3
   and 1e-2: **88.85 then 86.72**, so the optimum is interior at 3e-3 and the edge
   report is empty. Re-measured at the proper learning rate, block 8 is **87.69**
   against 88.24 at the undertuned 1e-3 — and those two differ by 0.55 ± 0.93, i.e.
   not distinguishable from the between-run floor, which is a useful incidental check
   that the floor is where §4 says it is. Every block in the sweep now has an
   interior optimum except block 0, which sits at its *bottom* edge.

   Worth noting what the edge was doing: it was flattering the finding. The bound
   read 88.24 and made block 8 − block 11 look like +1.73; the measurement reads
   87.69 and makes it +1.19. A grid edge is not a neutral uncertainty — the peak it
   hides is the one you went looking for.

Block 0's grid is at its *bottom* edge and is now the only unclosed one. It cuts the
other way — a lower learning rate might lift it — but it would have to lift it ~9
points to disturb the shape, and the three grid points above it fall away
monotonically, so this is a known hole rather than a live threat.

### 3.3.2a Three draws of the depth curve, and the two marginal claims settle

The depth result's large links were far clear of the noise floor at one draw, but two
were not: the fall from block 8 to block 11, and block 8 against the six-block suffix
containing it. Both are the difference between "a monotone rise" and "an interior
peak", so both got draws rather than argument. Seeds 1 and 2 at the **seed-0 tuned
learning rates** — only the episode sample moves, `--split-seed` held at 0 so every
draw sees the same validation partition. `scripts/run_vit_position_seed.sh`.

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| block 0 | 78.80 | 78.43 | 79.07 | 78.76 | 0.32 |
| block 2 | 78.87 | 78.32 | 78.75 | 78.64 | 0.29 |
| block 5 | 82.04 | 81.33 | 81.97 | 81.78 | 0.39 |
| **block 8** | 87.69 | 87.63 | 87.16 | **87.49** | 0.29 |
| block 11 | 86.51 | 85.81 | 85.53 | 85.95 | 0.50 |
| last-6 suffix | 86.69 | 86.08 | 85.92 | 86.23 | 0.41 |

Per-seed differences, mean ± t(2)·sd/√3:

| comparison | 3-draw | per-draw | |
|---|---|---|---|
| block 8 − block 5 | **+5.71 ± 1.38** | +5.65, +6.29, +5.19 | settled |
| block 11 − block 5 | **+4.17 ± 1.31** | +4.47, +4.48, +3.56 | settled |
| block 5 − block 2 | **+3.14 ± 0.28** | +3.17, +3.01, +3.23 | settled |
| **block 8 − block 11** | **+1.54 ± 0.80** | +1.19, +1.81, +1.63 | **settled** |
| **block 8 − last-6** | **+1.26 ± 0.68** | +1.00, +1.55, +1.24 | **settled** |
| block 2 − block 0 | −0.12 ± 0.48 | +0.07, −0.11, −0.32 | not settled |

**Both marginal claims settle, in the direction the single draw suggested.** The
inverted U in depth is established: the peak is interior, not a plateau at the top.
And adding attention blocks to the best single block costs accuracy — so the
interference term §3.3.3 infers is real and negative, not merely "whatever makes the
suffix curve flat".

**The depth span replicates tightly.** 8.89, 9.31, 8.41 across the three draws. The
shallow pair stays indistinguishable (blocks 0 and 2 differ by −0.12 ± 0.48), which
is the third independent confirmation that early-block attention does nothing here.

**A note on how much the pairing buys across draws, because it is less than within
one.** Individual accuracies have sd ≈ 0.37 across draws; the per-seed differences
have sd ≈ 0.33. If the two accuracies in a difference were independent the difference
sd would be √2 × 0.37 ≈ 0.52, so pairing across draws gains a factor of **1.6**,
implying a correlation of about ρ ≈ 0.6 between draws. That is a real gain and the
reason these two claims resolve at n=3, but it is far smaller than the within-seed
episode pairing that §3.4 is about, where the episodes are *literally identical*. The
two should not be quoted as the same effect: seeds share a validation split and tuned
learning rates but not an episode sample.

### 3.3.3 Capacity within attention: flat, and why that flatness was misleading

`select_attn_last_n(n)` adapts attention in the last *n* of 12 blocks — same
parameter type, nested, exactly 2.75% of the backbone per block, every peak interior
on its own grid, no grid edges anywhere in the sweep. EuroSAT, seed 0, 100 episodes,
each *n* at its own tuned learning rate.

| n blocks | % of backbone | lr | EuroSAT | frac_mod |
|---|---|---|---|---|
| 0 (`frozen_proto`) | 0 | — | 79.27 | 0.000 |
| **1** (= block 11) | **2.75** | 3e-4 | **86.51** | 2.743 |
| 2 | 5.51 | 3e-4 | 86.09 | 5.484 |
| 3 | 8.26 | 3e-4 | 86.48 | 8.223 |
| 4 | 11.01 | 3e-4 | 86.60 | 10.962 |
| **6** | **16.52** | 1e-4 | **86.69** | 16.279 |
| 8 | 22.02 | 1e-4 | 85.72 | 21.641 |
| 12 (`attn_only_ft`) | 33.04 | 1e-4 | 85.96 | 32.33 |

**n = 1 through 6 is one flat line** — spread 0.60 across those five points, exactly
the floor. Taken alone this says a six-fold increase in adapted parameters changes
nothing, and that is what an earlier version of this section concluded: capacity
within attention is free, so one block is enough and the active ingredient is the
type.

**§3.3.2 shows why that conclusion does not follow.** Every row of this table is a
*suffix*: n=6 is blocks 6–11, which **contains block 8**, the best single block
measured. And n=6 scores 86.69 against block 8's 87.69 — paired, block 8 − n=6 is
**+1.00 ± 0.96** at one draw, which clears its interval and the floor only just.
**§3.3.2a settles it at three draws: +1.26 ± 0.68, every draw positive.** So adding
five more attention blocks to the best single one *does* cost accuracy. What the suffix table cannot be read as is a
capacity curve: whatever the sign of the interference term, n=6 contains block 8, so
the flat line is a large positive depth effect and some other term summing to
something flat, not evidence that capacity is free.

That also disposes of the suffix family as an instrument: it cannot separate "how
many blocks" from "which blocks", because growing n changes both at once. The
positional sweep is the design that can, and it is the one to extend.

What survives from this table, independent of the confound:

- A single block of attention somewhere in the top half beats `full_ft` by 3.3 and
  frozen by 7.2. Both far clear of the floor, and §3.3.2 only strengthens it.
- The learning rate moves with n as a step, not a slope: 3e-4 for n ≤ 4, 1e-4 for
  n ≥ 6. The fraction-of-backbone heuristic that mispitched seven of my grids is
  real but much coarser than I was treating it — a 4× change in adapted parameters
  moved the optimum not at all, and a 12× change moved it one grid point.
- n=6 − `attn_only_ft` is +0.73 ± 1.06, which clears neither bar. There is still no
  evidence that capacity *above* ~17% hurts, only that it does not help.

### 3.3.4 Three retracted readings of this data

All three were published in this document and all three were wrong. The sequence
matters more than any of them:

**(a) "It does not matter which sublayer you move."** `attn` and `mlp` were called a
tie, on the strength of combining the two within-run intervals in quadrature — the
test §3.4 shows is invalid for paired data, and which understated this margin by
about 4×. The nominal gap (1.20 at seed 0) was visible at the time and was dismissed
as noise. It is not noise; §3.3.1 has it at +1.18 [+0.76, +1.60]. **Cause: wrong
statistical test.**

**(b) "An inverted U in capacity, peaking near a third of the backbone."** Read off
the type ordering in §3.3.1, which is not monotone in parameter count. §3.3.3
measured capacity directly and found a flat line from 2.75% upward, with no peak at
33% and no climb to reach it. The apparent optimum was an artefact of 33% being the
smallest non-trivial point a *type*-based partition can offer — the next point down
is LayerNorm at 0.045%, and there is nothing in between to reveal that the curve had
levelled off long before 33%. **Cause: reading a curve off four points that were
never a curve.**

**(c) "Attention is the active ingredient and one block of it is enough."** Replaced
(b), and read the flat suffix curve as evidence that size does not matter within
attention while type does. §3.3.2 shows the suffix curve confounds depth with count
and that the flatness is two effects cancelling: one well-placed block beats six
containing it by 1.00 ± 0.96, and five same-size blocks span 8.89 points. **Cause: an
instrument that varied two things at once, trusted as though it varied one.**

**(d) "Attention is the active ingredient."** Survived (c) and was the live reading
of §3.3 until §3.3.6 ran the same positional sweep inside the block MLP. The MLP
curve has the same shape, a comparable span, and a higher peak — and a single MLP
block beats `full_ft` by 5.67, against a pre-registered prediction that no single
MLP block would beat it by anything like 5. So depth is the axis and attention is
not privileged. **Cause: a sweep run in one parameter type and read as a fact about
the backbone.** This one was caught by running the experiment the previous reading
implied rather than by a later contradiction, which is the only reason it cost a
retraction instead of a result.

The common thread is that each reading took a quantity averaged over an unmeasured
axis and interpreted it as though the axis were not there. (a) averaged over the
episode pairing, (b) and (c) over depth. The live reading in §3.3 is itself an
average over *seed*, which §4 names as the next thing to break it. It was also an
average over *task* until §3.3.5, and that one did break: the depth effect holds on
ISIC but with the curve mirrored and the single-slice win gone, so the claim had to
be split into a part that generalises and a part that does not. Three retractions
from reading past an unmeasured axis, and then a fourth axis measured and the claim
narrowed before it was quoted — the pattern is the point.

### 3.3.5 The same sweep on ISIC: depth matters there too, and it points the other way

§3.3.2 is EuroSAT, where `full_ft` loses. §4 called the same sweep on ISIC — twice
the headroom, `full_ft` winning — the single most informative unrun experiment on
this arm. This is it: the same five depths, the same selector, the same protocol,
one draw each at 100 episodes and 20 inner steps, each depth tuned separately over
one wide six-point grid. `scripts/run_vit_position_isic.sh`.

| block | lr | ISIC | frac_mod | EuroSAT (§3.3.2) |
|---|---|---|---|---|
| 0 | 3e-3 | 48.83 | 2.751 | 78.80 |
| 2 | 1e-3 | 48.99 | 2.745 | 78.87 |
| **5** | 1e-3 | **49.23** | 2.746 | 82.04 |
| 8 | 1e-3 | 47.23 | 2.747 | **87.69** |
| 11 | 3e-4 | 45.00 | 2.741 | 86.51 |
| — `frozen_proto` | — | 39.35 | 0.000 | 79.27 |
| — `attn_only_ft` (all 12) | 3e-4 | 50.20 | 32.3 | 85.96 |
| — `full_ft` | 1e-4 | **52.49** | 83.3 | 83.24 |

Every ISIC optimum is interior (block 0's upward extension is below), and `frac_mod`
confirms all five interventions are the same size to within a hundredth of a point.

Paired per-episode differences on ISIC, same seed so literally the same 100 episodes:

| comparison | difference | |
|---|---|---|
| block 0 − `frozen_proto` | **+9.48 ± 1.41** | settled |
| block 5 − `frozen_proto` | **+9.88 ± 1.81** | settled |
| block 5 − block 11 | **+4.23 ± 1.51** | settled |
| block 5 − block 8 | +2.00 ± 1.70 | settled |
| block 11 − block 8 | −2.23 ± 1.28 | **superseded — see §3.3.5a** |
| block 0 − `full_ft` | −3.67 ± 1.60 | settled |
| block 2 − `full_ft` | −3.51 ± 1.54 | settled |
| block 5 − `full_ft` | −3.27 ± 1.65 | settled |
| block 11 − `full_ft` | −7.49 ± 1.46 | settled |
| block 0 − `attn_only_ft` | −1.37 ± 1.64 | **not settled** |
| block 2 − `attn_only_ft` | −1.21 ± 1.69 | **not settled** |
| block 5 − `attn_only_ft` | −0.97 ± 1.60 | **not settled** |
| block 0 − block 2 | −0.16 ± 1.57 | **not settled** |
| block 2 − block 5 | −0.24 ± 1.48 | **not settled** |
| block 0 − block 5 | −0.40 ± 1.46 | **not settled** |

**1. Depth matters on ISIC, and the two curves are roughly mirrored.** ISIC is one
flat plateau over blocks 0/2/5 — no pairwise difference among them settled, spread
0.40 — and then falls away: block 5 beats block 11 by 4.23 ± 1.51 at this draw, and
**+3.28 ± 2.07 over three draws** (§3.3.5a). The fall is one step down from the
plateau to the deep end — §3.3.5a shows blocks 8 and 11 are *not* distinguishable
from each other once draws are taken. EuroSAT is flat
over blocks 0/2 and then *rises*, block 8 beating block 5 by 5.65 ± 1.40. The useful
depths are the shallow half on ISIC and the deep end on EuroSAT.

The reversal is not a magnitude difference that the noise floor could produce.
**block 5 − block 11 is +4.23 ± 1.51 on ISIC against −4.47 ± 1.29 on EuroSAT**:
opposite signs, both settled, same selector at the same two depths under the same
five-way 5-shot protocol.

**2. The depth effect is about half the size under strong shift.** Spread across the
five depths is 4.23 on ISIC against 8.89 on EuroSAT. Depth is real in both regimes;
it is not equally decisive in both.

**3. No single block beats `full_ft` on ISIC, and that is the cross-regime result.**
The best ISIC depth loses to `full_ft` by 3.27 ± 1.65, where on EuroSAT the best
depth *beat* it by 4.45 ± 1.24. So the headline of §3.3.2 — a 2.75% slice beating all
85.8M parameters — is a **mild-shift** phenomenon, not a property of the backbone.
Under strong shift, joint movement across parameter types earns its capacity.

That is the same shape as §3.5's claim about shift magnitude, arrived at through a
completely different partition, which is the strongest thing in this section: §3.5
reached it by comparing parameter *types* across two datasets, and this reaches it by
comparing *depths*. Two independent decompositions, same conclusion.

**4. On ISIC one block LOSES to all twelve — corrected from a single-draw tie.**
At seed 0, blocks 0, 2 and 5 are each indistinguishable from `attn_only_ft` (−1.37,
−1.21, −0.97, none settled), and an earlier version of this point read that as a tie
and concluded the ISIC shortfall is *not* about attention capacity. **Three draws
overturn it**: `block 5 − attn_only_ft = −1.52 ± 1.30`, every draw negative
(−0.97, −2.01, −1.56), settled.

So under strong shift, twelve blocks of attention **do** buy something over one
well-placed block — about a point and a half — and the honest statement is that the
ISIC shortfall has *two* components, not one: part is attention capacity, and part is
the other parameter types (one block still loses to `full_ft` by 3.72 ± 1.03, three
draws, which is more than the 1.52 that extra attention recovers). That is a weaker
and more complicated claim than the tie, and it narrows §3.3.1's open question less
than I said it did.

This was caught by `paper/make_figures.py` recomputing every quoted number from
`results/` rather than copying it, which is the argument for generating figures from
source. It is the **seventh** single-draw link to move under draws (four promoted,
three demoted — see §3.3.6b), and the first found by a consistency check rather than
by running the draws for their own sake.

**5. Block 0's grid edge, and what closing it cost the hypothesis.** Block 0 first
tuned to 3e-3, the top of its grid, so 48.83 was a lower bound — and raising it was
exactly what would have turned the plateau into the shallow-peaked rise that the
reading in point 6 wanted. `scripts/run_vit_position_isic_hi.sh` extended it to 1e-2
and 3e-2: **44.77 then 39.07**, both well below 3e-3's 50.05 on the tuning subset, so
the optimum was already interior at 3e-3 and the edge report is now empty. The
plateau is real and 48.83 stands. Per §3.3.2 point 4 this was checked *before* the
section was written, not after.

The extension's re-measurement is also a free replication, and it is reported here
rather than quietly averaged in: `results/vit_isicpos0` and
`results/vit_isicpos0_v2` are the **same configuration** — same selector, same lr
(3e-3), same `frac_mod` (2.751), same 100 episodes — and give **48.83 and 48.48**, a
gap of 0.35. Every paired difference in this section is computed against the first
of those, so that is the number the table quotes; the pair is a third independent
confirmation that the between-run floor §4 records at ~0.6 is the right order of
magnitude (the other two: block 8 on EuroSAT at two learning rates, 0.55 ± 0.93, and
MLP block 5 before and after its extension, 0.21).

Worth recording for its own sake: **on EuroSAT block 0 pinned at the BOTTOM of its
grid (3e-5, wanting almost no movement); on ISIC the same block pins near the TOP
(3e-3).** Same selector, same depth, opposite ends of the same grid. That is a direct
read on where each shift bites, and it does not depend on any accuracy comparison.

**6. A hypothesis, and the form it survives in.** The obvious reading is that the
depth at which adaptation pays tracks the depth at which the pretrained
representation stops transferring: dermoscopy diverges from ImageNet-21k early, so
ISIC needs early adaptation; EuroSAT is overhead RGB where low-level features carry
over and only the task-specific top needs moving. That predicted ISIC blocks 2 and 0
would be *at least as good* as block 5.

They are — but as a **tie, not a rise**, so the prediction passes in its weak form
only. The defensible statement is "on ISIC useful adaptation is available anywhere in
the shallow half and is lost at the top", **not** "it concentrates early". The
learning-rate inversion in point 5 is independent evidence for the mechanism and is
the more interesting half of the case for it.

**7. One methodological note, because it cost a retraction.** The first ISIC depth
measured was block 8, chosen because it is EuroSAT's peak, and on that single point
the conclusion drawn was that the depth advantage simply does not transfer. Block 5
then came in 2.00 above block 8 and the reading had to be withdrawn. Picking the
*other* dataset's optimum is a systematically bad way to test whether an optimum
transfers: if the peak moves at all, that depth is close to the worst available
probe, and the error runs toward "no effect here". Measure the curve, not the other
dataset's argmax.

### 3.3.5a Three draws of the ISIC curve: the mirror holds, one sub-step does not

§3.3.5 was single-draw at every depth, and §4 called that the largest hole in the
arm. Seeds 1 and 2 at the seed-0 tuned learning rates, `--split-seed` held at 0 (block
0 runs at its **post-extension** 3e-3, not the pre-extension pick — the driver's
merge prefers `_v2` files and logs the supersession).
`scripts/run_vit_position_isic_seed.sh`.

| | seed 0 | seed 1 | seed 2 | mean | sd | frac_mod |
|---|---|---|---|---|---|---|
| block 0 | 48.83 | 48.45 | 47.32 | **48.20** | 0.78 | 2.751 |
| block 2 | 48.99 | 47.75 | 47.16 | 47.96 | 0.93 | 2.745 |
| block 5 | 49.23 | 47.07 | 47.04 | 47.78 | 1.25 | 2.746 |
| block 8 | 47.23 | 45.09 | 45.21 | 45.84 | 1.20 | 2.747 |
| block 11 | 45.00 | 44.43 | 44.05 | 44.49 | 0.48 | 2.741 |

Per-seed differences, mean ± t(2)·sd/√3:

| comparison | 3-draw | per-draw | |
|---|---|---|---|
| block 2 − block 11 | **+3.47 ± 1.14** | +3.99, +3.32, +3.11 | settled |
| block 5 − block 11 | **+3.28 ± 2.07** | +4.23, +2.64, +2.99 | settled |
| block 0 − block 8 | **+2.36 ± 2.25** | +1.60, +3.36, +2.11 | settled |
| block 5 − block 8 | **+1.93 ± 0.23** | +2.00, +1.97, +1.83 | settled |
| block 8 − block 11 | +1.35 ± 1.98 | +2.23, +0.67, +1.16 | **not settled** |
| block 0 − block 2 | +0.24 ± 1.09 | −0.16, +0.71, +0.16 | not settled |
| block 2 − block 5 | +0.19 ± 1.15 | −0.24, +0.68, +0.12 | not settled |
| block 0 − block 5 | +0.42 ± 2.24 | −0.40, +1.39, +0.28 | not settled |

**1. The plateau is confirmed as a plateau, in the way that actually counts.** All
three internal comparisons across blocks 0/2/5 stay unsettled at three draws, and
the *argmax moves between draws* — block 5 at seed 0, block 0 at seeds 1 and 2. A
curve I had merely failed to resolve would have kept its ordering; one whose ordering
reshuffles between draws is flat. §3.3.5's "flat plateau over 0/2/5" was the right
reading of the single draw and survives the test designed to break it.

**2. One sub-step is DEMOTED.** §3.3.5 quoted `block 11 − block 8 = −2.23 ± 1.28`
from seed-0 per-episode pairing and called it settled. At three draws it is
**+1.35 ± 1.98, not settled** — the per-draw values are +2.23, +0.67, +1.16, so seed 0
was the most favourable of the three. **The ISIC decline is therefore one step down
from the shallow plateau to the deep end, not a graded fall through it.** Blocks 8
and 11 are not distinguishable from each other here.

That is the fifth time a single-draw link has moved under draws (§3.3.2a has two
going the other way, where three draws *promoted* marginal claims). The pattern is
not "single draws are optimistic" — it is that single draws are *uninformative about
links near the floor*, in either direction.

**3. The mirror is now three draws on both sides, and it holds.** Every
shallow-versus-deep comparison has the opposite sign on the two datasets, all eight
settled:

| comparison | ISIC | EuroSAT |
|---|---|---|
| block 2 − block 11 | **+3.47 ± 1.14** | **−7.31 ± 1.13** |
| block 5 − block 11 | **+3.28 ± 2.07** | **−4.17 ± 1.31** |
| block 0 − block 8 | **+2.36 ± 2.25** | **−8.73 ± 1.42** |
| block 5 − block 8 | **+1.93 ± 0.23** | **−5.71 ± 1.38** |

Shallow beats deep on ISIC; deep beats shallow on EuroSAT. Same selector, same
depths, same five-way 5-shot protocol, three draws each. This was the weakest link in
§3.3.5's cross-dataset reading and it is now the best-supported claim in the section.

**4. The depth span is smaller than the single draw suggested.** 3.71 on the
three-draw means against 4.23 at seed 0, versus EuroSAT's 8.85. So ISIC's depth
effect is real and settled but **less than half** the size of EuroSAT's — consistent
with §3.5's reading that mild shift is where placement buys the most.

**5. Draw-to-draw variance is larger on ISIC than on EuroSAT**, which is worth
recording for anyone sizing a future sweep: accuracy sd across draws runs 0.48–1.25
here against 0.29–0.50 on EuroSAT (§3.3.2a). ISIC's 7 classes and 1.3 images per
group make its episode sample noisier, so an ISIC link needs a bigger true effect
than a EuroSAT one to clear the same bar. Two of the settled rows above have
intervals above 2 points for that reason.

### 3.3.6 The same sweep in MLP: the effect is DEPTH, and type interacts with it

§3.3.2 varies depth inside attention. §4 recorded that nothing varied depth inside
any other type, and that the possibility of a well-placed MLP block also beating
`full_ft` was what would decide whether the finding is about attention or about
depth. `select_mlp_block`'s docstring recorded both predictions **before this data
existed**:

> *"about attention"* — MLP stays near its whole-type 84.76 at every depth, with no
> comparable spread and no single MLP block beating `full_ft` by anything like 5.
> *"about depth"* — the same shape, and a single MLP block beats `full_ft`.

EuroSAT, seed 0, 100 episodes, 20 inner steps, each cell tuned separately over one
wide six-point grid. Every optimum interior (block 5's top edge was extended and
closed, §3.3.6a). `frac_mod` is 2.64–2.75 on every attention row and 5.45–5.50 on
every MLP row — exactly the 1:2 size ratio the selectors claim.
`scripts/run_vit_position_mlp.sh`.

| block | attn lr | attention | MLP lr | MLP | MLP − attn |
|---|---|---|---|---|---|
| 0 | 3e-5 | 78.80 | 3e-4 | 81.36 | **+2.56 ± 1.28** |
| 2 | 3e-4 | 78.87 | 3e-4 | 80.69 | **+1.83 ± 1.15** |
| 5 | 3e-4 | 82.04 | 3e-3 | 86.12 | **+4.08 ± 1.20** |
| **8** | 3e-3 | 87.69 | 1e-3 | **88.91** | **+1.21 ± 0.96** |
| 11 | 3e-4 | 86.51 | 1e-3 | 85.17 | **−1.33 ± 0.85** |
| — | | `attn_only` 85.96 | | `mlp_only` 84.76 | |
| — | | `full_ft` 83.24 | | `frozen` 79.27 | |

**1. The "about attention" branch is dead.** MLP in block 8 alone scores 88.91 — the
best number anywhere in this document, from 5.5% of the backbone — and beats
`full_ft` by **+5.67 ± 0.99**, where the prediction said no single MLP block would
beat it by anything like 5. MLP in block 5 also beats `full_ft` (+2.88 ± 1.19). So
the depth result is not about attention.

**2. Depth is about as decisive inside MLP as inside attention.** Span 8.21 against
8.89; `mlp_b8 − mlp_b0 = +7.55 ± 1.15`. Within MLP the chain is
`b5 − b2 = +5.43 ± 1.27`, `b8 − b5 = +2.79 ± 1.00`, `b8 − b11 = +3.73 ± 0.94`, all
settled, and the shallow pair ties (`b2 − b0 = −0.67 ± 1.17`). **Same shape as
attention: a flat shallow region, a steep rise, an interior peak at 8, a fall at
11.** Two parameter types, two independent sweeps, one shape. That is what makes
depth a property of the backbone rather than a fact about attention.

**3. Type is not free once depth is fixed — but the sign flip does NOT survive
draws.** At one draw MLP wins at blocks 0, 2, 5, 8 and loses at block 11, all five
settled, which read as an interaction that changes sign. **§3.3.6b corrects this**:
with three draws MLP − attention at block 11 is −0.96 ± 1.61, *not settled*, and
blocks 0 and 11 are both unsettled. The surviving claim is that **MLP is ahead at
every depth measured and never settledly behind, with a margin that varies by a
factor of 3.6 across depth** (+1.29 to +4.66 among settled cells). The interaction
is in the magnitude, not the sign. Read §3.3.6b before quoting any cell of this
column.

> **This is what §3.3.1's `attn − mlp` = +1.18 was averaging over.** That comparison
> moves both type *and* the implicit depth distribution at once: whole-type
> adaptation touches all twelve blocks, so it averages a quantity that runs from
> +4.08 to −1.33 across depth. The sign of the average is not wrong, but it is an
> average over an interaction, and reading it as "attention is the better type" is
> the same error as the three retractions in §3.3.4 — a quantity averaged over an
> unmeasured axis read as though the axis were not there. The difference is that
> this time the axis was measured before the claim was promoted.

**4. The shallow end corrects a claim I had been repeating.** Attention in blocks 0
and 2 is indistinguishable from `frozen_proto`, which earlier commits of mine
generalised to "early-block adaptation does nothing here". **MLP in block 0 beats
frozen by +2.09 ± 1.30 and in block 2 by +1.43 ± 1.32.** So the shallow null is a
fact about *attention at those depths*, not about those depths. What both types share
is a flat shallow *region* — MLP's sits a little above frozen, attention's sits on
it.

**5. One well-placed block of either type beats everything larger; a badly placed
block of either type does not.** Block 8 beats `full_ft` by 5.67 (MLP) and 4.45
(attention); block 0 loses to it by 1.88 (MLP) and 4.44 (attention). The win is not a
property of small interventions as such — it is a property of *well-placed* ones, and
placement is doing more work than either size or type.

**6. The learning rates move with depth, not with size.** MLP block 5 wants 3e-3
while MLP block 11 wants 1e-3, at identical parameter counts; attention block 8 wants
3e-3 while attention block 11 wants 3e-4. Across the two types the same depth often
wants a similar rate despite a 2× size difference, which is further evidence against
the fraction-of-backbone heuristic §4 records as having mispitched seven grids.

### 3.3.6a Block 5's grid edge, and which way it leaned

MLP block 5 first tuned to 3e-3, the top of its grid, so 85.91 was a lower bound. The
extension (`scripts/run_vit_position_mlp_hi.sh`, 1e-2 and 3e-2) left the optimum
interior at 3e-3 and the re-measurement gives **86.12** — 0.21 from the bounded
value, i.e. inside the ~0.6 floor, so nothing moved except its status.

Worth recording because it is the mirror of block 8's edge on the attention curve.
Block 5 is the depth with MLP's **largest** margin over attention, so an understated
row there would have **shrunk** the interaction this section describes; block 8's
attention edge **inflated** that curve's headline. The two lean in opposite
directions and both were closed on the same rule. Choosing which edges to chase by
how they lean is how a result stops being a measurement — so the driver reads the
edge report rather than a depth list, skips interior optima, and refuses to extend a
*bottom* edge upward rather than guessing at it.

### 3.3.6b Three draws of the MLP curve: the sign flip does not survive

§3.3.6 was single-draw, so §4 recorded that the type × depth interaction inherited
the weaker of the two curves' draw counts. Seeds 1 and 2 at the seed-0 tuned learning
rates, `--split-seed` 0, block 5 at its post-extension 3e-3.
`scripts/run_vit_position_mlp_seed.sh`.

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| block 0 | 81.36 | 79.57 | 79.88 | 80.27 | 0.96 |
| block 2 | 80.69 | 80.31 | 81.03 | 80.68 | 0.36 |
| block 5 | 86.12 | 86.83 | 86.39 | 86.44 | 0.36 |
| **block 8** | 88.91 | 88.69 | 88.76 | **88.79** | 0.11 |
| block 11 | 85.17 | 84.48 | 85.32 | 84.99 | 0.45 |

**The MLP curve's own structure is settled at every link**, and more tightly than
attention's: `b5 − b2 = +5.77 ± 1.62`, `b8 − b5 = +2.34 ± 1.14`,
`b8 − b11 = +3.80 ± 0.97`, `b8 − b0 = +8.52 ± 2.11`. Span 8.52 against attention's
8.85, and **the argmax is block 8 in all three draws** — unlike ISIC's plateau, whose
argmax moved (§3.3.5a). So the interior peak at block 8 is now established in *two*
parameter types independently.

Now the interaction, three draws on both types:

| depth | MLP | attn | MLP − attn | per-draw | |
|---|---|---|---|---|---|
| 0 | 80.27 | 78.76 | +1.51 ± 2.30 | +2.56, +1.15, +0.81 | **not settled** |
| 2 | 80.68 | 78.64 | **+2.03 ± 0.57** | +1.83, +1.99, +2.28 | settled |
| 5 | 86.44 | 81.78 | **+4.66 ± 1.84** | +4.08, +5.49, +4.41 | settled |
| 8 | 88.79 | 87.49 | **+1.29 ± 0.68** | +1.21, +1.07, +1.60 | settled |
| 11 | 84.99 | 85.95 | −0.96 ± 1.61 | −1.33, −1.33, −0.21 | **not settled** |

> **The sign flip at block 11 is NOT settled, and §3.3.6 point 3 is corrected here.**
> At one draw it read −1.33 ± 0.85, settled. Three draws give **−0.96 ± 1.61**: all
> three per-draw values are negative, but seed 2's −0.21 widens the interval through
> zero. So "MLP loses to attention at the last block" is **suggested and not
> established**, and the claim that the interaction *changes sign* with depth is
> withdrawn.

**What the interaction actually is, stated at the strength the data supports.** MLP
is **ahead at every depth measured** — settled at blocks 2, 5 and 8, unsettled at 0
and 11, and **never settledly behind**. What varies with depth is the *margin*, and
it varies a lot: among the settled cells it runs **+1.29 to +4.66, a factor of
3.6**, peaking in the middle of the stack and shrinking at both ends. So the
interaction is in the **magnitude**, not the sign. That is a weaker claim than
§3.3.6 made and a different one: there is no depth at which the better parameter
type demonstrably reverses.

**This makes §3.3.1's puzzle sharper rather than softer.** Whole-type attention beats
whole-type MLP by +1.18 ± 0.42 (three draws, §3.3.1) — yet at **no single depth does
attention settledly beat MLP**, and at three of five depths MLP settledly wins. A
whole-type comparison and a per-depth comparison of the same two types point opposite
ways. The resolution is not that one is wrong: whole-type MLP adapts 66% of the
backbone against attention's 33%, so it suffers more from §3.3.3's interference term,
and that penalty is what the whole-type average measures. **`attn − mlp` is a fact
about adapting all twelve blocks at once, and carries no information about which type
is better to adapt at a given depth.**

**Sixth link to move under draws, and the tally is now even.** §3.3.2a promoted two
marginal EuroSAT links; §3.3.5a demoted one ISIC link; this demotes one MLP link.
Three up, three down. The lesson §3.3.5a drew stands and is worth repeating because
it is counter-intuitive: **single draws are uninformative about links near the floor
in either direction**, so a marginal single-draw result should be written as marginal
rather than guessed at, and the guess should not be the optimistic one.

**What this costs and what it does not.** §3.3's framing is unaffected: depth is still
the dominant axis, the peak at block 8 is now established in two types, and one
well-placed block of either type still beats every larger configuration. What is lost
is the claim that type *reverses* with depth — a detail of the interaction, not its
existence.

### 3.4 Three draws: both claims survive, and the test I was using was the wrong one

Seeds 1 and 2 of `full_ft`, `attn_only_ft`, `ln_only_ft` and `frozen_proto`, at the
learning rates tuned on seed 0, 100 episodes, identical split. Only the episode
sample moves. `scripts/run_vit_seeds.sh`.

| | seed 0 | seed 1 | seed 2 | mean | sd | range |
|---|---|---|---|---|---|---|
| **EuroSAT** | | | | | | |
| `attn_only_ft` | 85.96 | 85.41 | 85.32 | **85.56** | 0.35 | 0.64 |
| `ln_only_ft` | 84.75 | 83.76 | 84.03 | 84.18 | 0.51 | 0.99 |
| `full_ft` | 83.24 | 82.85 | 83.11 | 83.07 | 0.20 | 0.39 |
| `frozen_proto` | 79.27 | 78.93 | 79.45 | 79.22 | 0.26 | 0.52 |
| **ISIC** | | | | | | |
| `full_ft` | 52.49 | 51.15 | 50.85 | **51.50** | 0.87 | 1.64 |
| `attn_only_ft` | 50.20 | 49.08 | 48.60 | 49.29 | 0.82 | 1.60 |
| `ln_only_ft` | 49.60 | 48.48 | 48.64 | 48.91 | 0.61 | 1.12 |
| `frozen_proto` | 39.35 | 37.44 | 37.99 | 38.26 | 0.98 | 1.91 |

**Both live claims hold on every single draw:**

| paired difference | seed 0 | seed 1 | seed 2 | mean | sd | 95% CI (n=3) |
|---|---|---|---|---|---|---|
| EuroSAT `attn` − `full_ft` | +2.72 | +2.56 | +2.21 | **+2.50** | 0.26 | [+1.85, +3.15] |
| ISIC `full_ft` − `attn` | +2.29 | +2.07 | +2.25 | **+2.20** | 0.12 | [+1.90, +2.50] |

Three for three in both directions, with margins roughly ten times their
across-draw standard deviation. The ISIC comparison that looked marginal in §3.1 —
2.29 clearing a combined ±2.21 by 0.08 — is not marginal at all.

**And that is the methodological point, which is a correction to my own test.** I
had been comparing two baselines by combining their within-run intervals in
quadrature, √(ci₁² + ci₂²). That assumes the two accuracies are independent
estimates. They are not. `EpisodicSampler` seeds per-episode rather than
per-sampler, so within a seed **every baseline is evaluated on literally the same
episodes** — the docstring says so and it is why the class was written that way.
Episode difficulty is shared, so it cancels in the difference. The paired
difference's spread (sd 0.12–0.26) is three to seven times tighter than either
accuracy's own spread (sd 0.20–0.98), exactly as pairing predicts.

So the quadrature test was not merely conservative, it was answering a question
nobody asked: "could these two numbers have come from independent runs of different
things?" The question that matters is "does A beat B on the same tasks?", and the
per-seed difference answers it directly. Every margin in §3 compared this way is
stronger than §3.1 and §3.2 state, and §3.1's "on the line, not settled" was an
artefact of the wrong statistic rather than of thin data.

**What this does *not* license.** The across-draw spread of a single accuracy is
real and is not small on ISIC — `frozen_proto` moves 1.91 points across three draws
— so any *unpaired* comparison, such as a number quoted against one from a different
run or a different paper, still needs that spread. The headroom figures are
unpaired in exactly that way, and they move: ISIC 13.14 → **13.24**, EuroSAT 6.69 →
**6.34**, with `attn` at **165%** of `full_ft`'s gain rather than 168%. §3.5's
framing is unchanged by that, which is the point of checking.

One expectation was wrong and worth recording. I ran this because
`RESULTS_SUBSPACE.md` §7 found a believed ±0.56 floor was really ±1.1 — the
within-run interval understating true spread by 2×. That happened to a quantity
derived from an *extracted displacement*, and it does not transfer to episodic
accuracy: here the across-draw sd of 0.2–1.0 sits comfortably inside the within-run
±1.0–1.8. The lesson from §7 was right to act on and wrong about which direction
the error would come from.

### 3.5 What the two tables say together

Headroom is now measured on this arm rather than carried over from the ResNet one:

Means over the three draws of §3.4, since these are unpaired quantities:

| | `frozen_proto` | best | headroom | `full_ft`'s rank | best single type vs `full_ft` |
|---|---|---|---|---|---|
| ISIC | 38.26 | 51.50 (`full_ft`) | **13.24** | 1 of 7 | 83% of its gain |
| EuroSAT | 79.22 | 85.56 (`attn`) | **6.34** | 5 of 7 | **165%** of its gain |

ISIC has roughly twice the headroom, and that is where joint movement across
parameter types pays. Where there is half as much, it does not merely stop paying —
it goes **negative**, and a single parameter type does better than all of them
together.

That sharpens §2.5's claim rather than restating it. The earlier version was "where
there is little to gain, every group gets all of it." The measured version is:

> **The value of adapting parameter types jointly scales with the size of the
> shift, and under mild shift it is actively harmful relative to adapting one
> type.**

This also hands the mechanism question a candidate that §2.5 did not have.
`full_ft` opens 85.8M parameters and takes 20 steps on 25 support images. Under
strong shift there is enough signal to justify that capacity; under mild shift
there is not, and it overfits. That is a testable story rather than a restatement
of the data — and unlike `RESULTS_SUBSPACE.md` §6's dead mechanism it does not
need a quantity that cannot be separated from depth.

It is still two datasets, one draw each, with no seed spread on this arm.

**§3.3.2 moves the EuroSAT row of this table, and the direction matters.** The
table's "best" is the best of the *type* partition over three draws. The best
configuration now known on EuroSAT is attention in block 8 alone, at 87.69 on one
draw — so headroom there is **8.47, not 6.34**, and the best-single-intervention
figure against `full_ft` rises from 165% to **220%** of its gain. That does not weaken
the claim in the box above; it strengthens it, because the configuration that now
holds the ceiling adapts 2.75% of the backbone rather than 33%. But it does mean
the ISIC/EuroSAT headroom *ratio* this section leans on is measured with a type
partition on one side and a positional one on the other.

**§3.3.5 closes that gap and the ratio survives.** The ISIC positional sweep is now
run: its best depth is 48.20 over three draws (§3.3.5a; 49.23 at seed 0), so
positional headroom there is 8.85 against EuroSAT's 8.47 — i.e. **near-identical**, where the type-partition figures (13.24 against 8.47) made
them look two-to-one apart.
What separates the two datasets is therefore *not* mainly how much a single
well-placed intervention can get; it is whether anything beyond one is worth adding.
On EuroSAT the best single depth beats `full_ft` by 4.45; on ISIC it loses by 3.27.
That is a sharper version of the claim in the box above, and it is now measured with
the *same* partition on both sides.


### 3.6 The inner-step test: the mechanism survives, and the headline strengthens

§3.5 proposes that `full_ft` loses on EuroSAT because 85.8M parameters taking 20
steps on 25 support images over-optimise where there is little to gain. The
prediction is that reducing the optimisation **closes the gap**, because the larger
group is the one being hurt.

EuroSAT, seed 0, 100 episodes, validation. Both groups **re-tuned at each step
count** — a fixed learning rate would confound less-overfitting with
less-adaptation, since both predict the same sign for `full_ft`. Every peak below is
interior on its merged grid.

| inner steps | `attn_only_ft` | `full_ft` | gap |
|---|---|---|---|
| 5 | 84.67 (1e-4) | 84.01 (3e-5) | **+0.65** |
| 10 | 85.33 (1e-4) | 84.64 (3e-5) | **+0.69** |
| 20 | **85.96** (1e-4) | 83.24 (1e-5) | **+2.72** |

**The two groups respond to optimisation differently, and that is the result.**

* `attn_only_ft` rises monotonically with steps: 84.67 → 85.33 → 85.96. Total +1.29
  over three points, each step up, against the ~0.6 run-to-run noise §4 records.
* `full_ft` does not. It is **1.40 points worse at 20 steps than at 10**, which
  clears that noise. More optimisation makes full fine-tuning worse.

So the 20-step gap is mostly `full_ft` over-optimising rather than attention pulling
ahead, which is what §3.5 predicted. The mechanism is supported.

**And the headline gets stronger, not weaker.** The obvious worry was that
`attn_only_ft`'s advantage is an artefact of a step count that happens to suit it.
Give `full_ft` its *best* step count — 10, at 84.64 — and compare against attention
at *its* best — 20, at 85.96 — and attention still wins by **1.32**. Best against
best, adapting a third of the backbone beats adapting all of it.

**What this is not.** Single draw at 5 and 10 steps, so these are compared against
seed 0 at 20 steps (+2.72) rather than the three-draw mean (+2.50), like with like.
The 5-versus-10 difference for either group (0.66 and 0.63) sits at the noise floor
and nothing should be read into it; "`full_ft` peaks at 10" is not established, only
"`full_ft` at 20 steps is worse than at 10". The step counts are also a crude probe:
20 → 10 → 5 is three points on one axis, and the mechanism's other predictions —
more shots, weight decay, the crossover moving with support-set size — remain
unrun.

---

### 3.7 Support size: the decision rule, written before the result

**This section was written at 2026-10-21 02:25 box, while the k=20 job was on
config 11 of 12 and no measurement existed.** It is a pre-registration, and it
exists because the tuning rows already visible are suggestive enough that reading
them is a temptation, and because the rule that governs them depends on something
that has not happened yet.

The prediction under test (§3.5): if the attention advantage comes from full
fine-tuning over-optimising a small support set, the advantage should **shrink as
k grows**, because a larger support set gives the large parameter group something
real to fit.

What is already measured:

| k | attn | full_ft | frozen | attn − full_ft | draws |
|---|------|---------|--------|----------------|-------|
| 1 | 62.15 | 60.89 | *not run* | +1.25 ± 1.18 | 1 |
| 5 | 85.56 | 83.07 | 79.22 | +2.50 ± 0.64 | 3 |

So k=1 already went the **wrong way** — the gap is smaller at k=1, not larger.
Note the k=5 gap is +2.50 ± 0.64 over three draws; the +2.72 quoted in earlier
notes is the seed-0 single draw, and the two must not be mixed.

**The rule, fixed in advance:**

1. **If the k=20 attention grid is at its top edge** — argmax at lr=3e-04, the
   last point — the k=20 number is **not readable** and no conclusion about
   support size may be drawn from it. The grid is extended upward and the cell is
   re-run first. This is live: at the time of writing, `attn_only_ft` reads 0.9344
   at 3e-05 and **0.9421 at 1e-04, still rising**, with one point left. `full_ft`
   peaked at 3e-05 (0.9448) and fell to 0.9219 at 1e-04, so it is bracketed.
   Note which way this leans: an attention grid cut off while still rising
   **understates attention**, which biases the comparison toward "full_ft wins at
   k=20", which is the direction that would make the prediction fail. The edge
   would flatter the conclusion. §2 fact 4, and the one that cost us a headline
   once already.
2. **If the grid is bracketed and the gap at k=20 is smaller than at k=5**, the
   support-size prediction has failed in direction at both ends. The relationship
   is then not monotone in k, which no version of the over-optimisation story
   predicts. §3.5 gets rewritten as a **description** of where the effect is
   found, not a mechanism for why — and is reported as such, not quietly reframed.
3. **If the gap at k=20 is larger than at k=5**, the prediction is confirmed at
   the top end and k=1's failure is most likely a floor effect — but that reading
   is only available once `run_vit_shots_ref.sh` supplies frozen_proto at k=1. If
   attn and full_ft both sit near frozen at k=1, there was no gap to have there.
   Until that run exists, k=1 stays recorded as a failure in direction.
4. **Either way the depth result is untouched.** It needs `full_ft` to lose at
   k=5, which it does across three draws on both datasets. It does not need an
   explanation of why. Losing §3.5 costs one section, not the headline.

#### 3.7a The gate opened clean: no grid edge at k=20

Resolved 2026-10-21 02:55 box, from the sweep's own merge step, which ran before
the measurement stage and so was readable before any accuracy was.
**`results/vit_edges_shots20.json` is `[]`.** Both arms peak in the interior:

| lr | 1e-6 | 3e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 |
|----|------|------|------|------|------|------|
| `full_ft`      | .8803 | .9024 | .9283 | **.9448** | .9219 | .8219 |
| `attn_only_ft` | .8648 | .8901 | .9157 | .9344 | **.9421** | .9216 |

The worry in §3.7 point 1 was real but did not land: attention was still rising at
1e-04 with one point left, and 3e-04 came in at .9216, below it. So attention's
argmax is 1e-04, bracketed on both sides, and full_ft's is 3e-05, likewise.
**Branch 1 does not apply; the k=20 cell is readable.** Note also that attention's
optimum sits one grid step ABOVE full_ft's, which is the same lr/size relationship
§1 found at k=5 and §2 fact 3 warns about assuming backwards.

What the tuning rows suggest, with the caveats that matter:

`attn_only_ft − full_ft` at k=20 reads **−0.27 points** (.9421 vs .9448). That is
**not** "attention loses at k=20": 0.27 is well inside the ~0.6 noise floor (§2
fact 2), these are 50-episode rows rather than the 100-episode measurement, they
are a single draw, and the two optima were found on separate runs so the quoted
CIs are unpaired. Nothing is established from them.

What they do indicate is the thing branch 2 turns on: the gap has **collapsed**
from +2.50 ± 0.64 at k=5 (three draws) to indistinguishable from zero at k=20. The
100-episode paired measurement is running now and is the number that counts.

If it holds, the support-size prediction has failed in direction at **both** ends
— +1.25 at k=1, +2.50 at k=5, ~0 at k=20 — which is not a monotone relationship in
k at all, and no version of the over-optimisation story predicts a peak in the
middle. Per branch 2, §3.5 then becomes a description of where the effect is found
rather than a mechanism for why, and is reported that way. Per branch 4, the depth
result is unaffected either way.

Margins: single-draw k=1 and k=20 are per-episode paired, t(99)=1.9842. They are
one draw each, so §2 fact 6 applies in full — a single draw near the floor is
uninformative in *either* direction, and the k=20 reading will need its own draws
before anything in §3.5 is rewritten on the strength of it.

## 4. What is NOT claimed yet

Explicitly, so none of this gets quoted as a result:

* **Nothing here is comparable to SSF, LN-Tune or DIPA, and the header's old claim
  that we share their checkpoint was false** (checked against the papers
  2026-10-20; see the correction at the top of this file). SSF (Lian et al.,
  NeurIPS 2022) uses a ViT-B/16 IN-21k checkpoint but evaluates on FGVC and
  VTAB-1k, which are not episodic few-shot; LN-Tune (Basu et al., AAAI 2024)
  reports on Meta-Dataset and ORBIT; DIPA (Perera & Halgamuge, CVPR 2024) uses
  ViT-S/16 with DINO self-supervised pretraining. Backbone, pretraining
  objective, benchmark and episode protocol all differ, so a table of their
  numbers beside ours would compare four things while looking like one. Closing
  this needs re-running all three under our protocol; until then the positioning
  claim is internal only — our own ResNet replication is the stand-in for "what
  the type literature implies". **How this got in: the sentence was written once,
  early, from memory, and then copied into two more files and the draft without
  anyone opening a paper.** Same family as every other entry in the standing
  lessons — an unverified claim occupying the slot where a checked one belongs.
* **Both headline margins now rest on three draws, which is the floor, not a
  luxury.** §3.4 settles their *direction* decisively (every draw, both datasets).
  It does not make three draws many: `frozen_proto` on ISIC moved 1.91 points
  across them, so anything quoted unpaired — the headroom figures, anything
  compared against another paper — carries that spread.
* **§3.5's shift-magnitude claim rests on two datasets.** Two points do not
  establish a monotone relationship between shift magnitude and whether joint
  movement helps or hurts, and headroom is a proxy for shift magnitude rather than
  a measurement of it — now a measured proxy (13.24 against 6.34, three draws) but
  a proxy.
  ChestX, the third and largest shift, is unobtainable (`RESULTS_SUBSPACE.md` §8),
  so this may not be resolvable on BSCD-FSL at all.
* **The overfitting mechanism has passed one test of several.** §3.6 confirms the
  inner-step prediction: `full_ft` is 1.40 worse at 20 steps than at 10 while
  `attn_only_ft` improves monotonically to 20. Its other predictions — more shots,
  weight decay, the crossover moving with support-set size — are unrun, and one
  axis probed at three points is not a mechanism established. It is now a
  hypothesis with evidence rather than a story.
* **§3.3.3's plateau is single-draw at every point.** Five accuracies inside 0.60
  is consistent with one flat line and that is how it is read, but five draws of one
  number and five points on a gentle curve are not distinguishable at this noise
  level. The n=8 and n=12 dip (0.97 and 0.73 below n=6) is at the floor and is
  explicitly not claimed as a decline.
* **The capacity and positional sweeps are EuroSAT only, and attention-only.** Whether a single block
  is also sufficient on ISIC — where `full_ft` wins and headroom is twice as large —
  is untested, and it is the comparison that would most change the story. Nor does
  anything here sweep capacity within MLP, and LayerNorm has no positional sweep at
  all. **Position within MLP is now ANSWERED by §3.3.6** — same shape, comparable
  span, higher peak, and the pre-registered "about attention" branch dead — so the
  open part is narrower: LayerNorm affine is 0.045% of the backbone and gets within
  1.4 points of the best whole-type baseline, and nothing measures whether *its*
  usefulness is also concentrated at a particular depth. If it is, the claim becomes
  a property of the block index independent of what is adapted there, which is
  stronger than anything currently written.
* **ANSWERED, and the answer was narrower than the single draw suggested.** §3.3.6b
  gives the MLP curve three draws. The sign flip at block 11 is **demoted to
  unsettled** (−0.96 ± 1.61), so the interaction is in the *magnitude* (+1.29 to
  +4.66 among settled cells, a factor of 3.6) and not the sign; MLP is ahead at every
  depth measured and settledly behind at none. What remains open: **the interaction
  has not been measured on ISIC at all**, so it is one dataset; both curves are five
  of twelve depths; and whether MLP's margin genuinely falls at block 11 or merely
  stops rising needs either a fourth draw or a depth between 8 and 11.
* **CORRECTED: the strong-shift shortfall is partly attention capacity after all.**
  §3.3.5 point 4 used to say twelve blocks of attention buy nothing over one on ISIC,
  on a single-draw tie. Three draws make it −1.52 ± 1.30, settled, so extra attention
  recovers ~1.5 of the ~3.7 points one block gives away to `full_ft` there. The
  remainder is still unexplained and still points at the other parameter types, but
  the clean "it is not capacity" reading is gone.
* **Nothing here explains why other types are harmful alongside attention.**
  `mlp_only_ft` (84.76) and `ln_only_ft` (84.75) each beat frozen comfortably on
  their own, yet adding them to attention costs 2.72. Interference between
  simultaneously-adapted types is a description of that, not a mechanism, and
  nothing here measures it. The obvious probe — attention plus MLP, attention plus
  LayerNorm, pairwise — has not been run.
* **Whether the depth curve has an interior peak is NOT settled — only that it
  rises.** §3.3.2 answers the "one block means the LAST block" caveat this bullet
  used to raise, and block 8's grid edge is now closed (87.69 at an interior 3e-3).
  But the fall from block 8 to block 11 is +1.19 ± 1.11 and block 8 over the
  six-block suffix containing it is +1.00 ± 0.96: both clear their intervals and the
  ~0.6 floor only just. **The monotone rise through block 8 is established; the
  inverted U is not.** Block 0's grid remains at its *bottom* edge, unextended. Five
  of twelve blocks are measured, one draw each, so the curve between them is
  interpolation and whether the peak sits at 8 specifically or anywhere in 6–10 is
  open. A second seed on the five depths is the cheapest thing that would settle it
  and is the highest-value unrun item after the two sweeps below.
* **ANSWERED by §3.3.5, and the answer was "both".** Depth matters on ISIC too, but
  the curve is mirrored (shallow best, deep worst) and no single depth beats
  `full_ft` there. So the *depth effect* is a property of the backbone and the
  *single-slice win* is a mild-shift phenomenon — the two halves of this bullet were
  not alternatives. What remains open is narrower and now stated as its own entry:
  five of twelve depths on each dataset, one draw each, and two datasets is still two
  points on a shift axis that §4 already warns is a proxy.
* **PARTLY ANSWERED: EuroSAT's curve now has three draws, ISIC's still has one.**
  §3.3.2a settled both EuroSAT marginal claims and **§3.3.5a now gives the ISIC
  curve three draws too**: the plateau is confirmed (all three internal comparisons
  unsettled, argmax moving between draws), the mirror holds with four
  shallow-versus-deep comparisons of opposite sign on the two datasets, and ISIC's
  `block 8 − block 11` is **demoted** from settled to not settled. What remains:
  both curves are five of twelve depths, so whether EuroSAT's peak sits at 8
  specifically or anywhere in 6–10 is interpolation; **the MLP curve is still
  single-draw**, so the type × depth interaction inherits that; and EuroSAT block 0's
  bottom grid edge is the last unclosed edge.
* **Nothing explains what `full_ft` has that attention does not, under strong shift.**
  §3.3.5 point 4 sharpens the question: on ISIC one well-placed block ties all twelve
  blocks of attention, and both lose to `full_ft` by ~3.3. So the shortfall is not
  attention capacity at any depth or any count — it is the other parameter types. The
  pairwise probe below is the experiment, and it is still unrun.
  `scripts/run_vit_position_isic.sh` exists and is unrun. Both it and the MLP sweep
  use one wide six-point grid at every depth rather than a window centred on the
  fraction-of-backbone heuristic, because the EuroSAT sweep showed that heuristic
  gets the early end backwards — a single early block tuned *down* to 3e-5 while a
  single late block tuned *up* past 1e-3, and an undertuned block is indistinguishable
  from the depth effect being measured.
* **The seed spread covers four baselines, not eight.** `ln_scale`, `ln_shift`,
  `mlp` and `patch_embed` have one draw each, so every §3 row outside
  {`full_ft`, `attn`, `ln_only`, `frozen_proto`} is still single-draw. The band
  *orderings* among those four are not established.
* **Runs are not reproducible, and the cause is not seeded away.** Two
  `run_baselines.py` invocations with byte-identical configuration — same tuned lr,
  same inner steps, same episode seed, same split seed, `--no-amp` — disagree:
  `attn_only_ft` 85.21 against 84.67 and `full_ft` 84.71 against 84.01 at 5 steps,
  with **85 of 100 episodes giving different per-episode accuracies**. Ruled out as
  causes: task sampling (`EpisodicSampler` seeds per episode from numpy, so the
  tasks are fixed) and stochastic regularisation (every dropout and stochastic-depth
  rate in the adapted branch is 0, a train-mode forward consumes no global torch
  RNG, and two consecutive train-mode forwards are bit-identical on CPU — all
  verified). What is left is GPU floating-point nondeterminism in the backward pass,
  amplified through 5–20 Adam steps whose updates are normalised. Nothing calls
  `torch.use_deterministic_algorithms`.

  **What follows.** Every *individual* accuracy in this document carries roughly 0.6
  points of between-run noise on top of the episode-sample spread §3.4 measures. The
  paired differences look much more stable (the 5-step gap moved 0.50 → 0.66 while
  its two accuracies moved 0.55 and 0.69), which is a fourth reason to use the
  paired test, but that is one comparison and not a measurement. The three-draw
  margins in §3.2–§3.4 are unaffected in substance: each draw already carried this
  noise, so the measured spreads are honest error bars, and 2.50 / 2.20 / 1.18 are
  all large against 0.6. What this does kill is reading any single-draw difference
  below about a point, and quoting an unpaired accuracy to two decimals as if it
  were repeatable.
* **No displacement or frontier numbers.** Not run. When they are, the code's
  notion of a "diagonal" correction has to be checked against LayerNorm affine
  rather than BatchNorm affine before any of it is trusted.

### 4.1 An infrastructure bug that cost idle GPU time, recorded because it recurs

A pattern passed to `pgrep -f` or `pkill -f` **matches the command line of the
process doing the matching**, because that pattern is literally present in it. This
has now bitten three times in one session:

* `pkill -f tailscaled` killed the shell issuing it (the command line contains
  "tailscaled"), returning exit 143 before the daemon was restarted.
* A watcher polling `pgrep -cf run_vit_position_isic_hi` over ssh never saw the count
  reach 0, because its own ssh wrapper matched. The job had finished 30 minutes
  earlier and the GPU sat idle that whole time.
* `pkill -f "scratchpad/chain.sh"` then killed the shell that was about to launch the
  replacement job, so the launch silently did not happen.

**The rule.** Use `pkill -x <name>` to match an executable name exactly. To wait on a
job, poll the **sentinel file** the drivers already write (`results/<dir>/.done`) or
`nvidia-smi` utilisation, never a process pattern — a sentinel cannot match the
process asking about it. When a process pattern is unavoidable, confirm the finding
against something else (the log's mtime, the output file) before acting on it.

The deeper pattern is the same one §3.3.4 is about: a measurement that includes the
measurer. There it was a statistic averaging over the axis being tested; here it is a
process query matching the querying process. Both read as a clean signal.

### 3.7b The k=20 measurement, and a branch the pre-registration missed

Measured 2026-10-21 05:25 box. Both arms at their own tuned optimum (attention
1e-04, full_ft 3e-05, both `tuned: true`, both 100 episodes, seed 0, val split),
paired per-episode within the draw, t(99)=1.9842.

| k | attn_only_ft | full_ft | gap | ±95% | draws | verdict |
|---|------|------|------|------|-------|---------|
| 1  | 62.15 | 60.89 | +1.25 | 1.18 | 1 | marginal |
| 5  | 85.56 | 83.07 | **+2.50** | 0.64 | 3 | settled |
| 20 | 94.00 | 93.76 | +0.24 | 0.53 | 1 | **not settled** |

At k=20 the gap clears neither its interval nor the 0.6 floor. The sequence
+1.25, +2.50, +0.24 rises and then collapses, peaking at k=5.

**The branch §3.7 did not anticipate: a ceiling.** Branch 2 said a smaller gap at
k=20 with a bracketed grid means the relationship is not monotone in k, therefore
§3.5 is a description rather than a mechanism. That inference is **not available
from these numbers**, because both arms sit at **94%** on a 5-way task. Room above
is ~6 points. A gap cannot be large where neither method has anywhere to go, so
"the mechanism's prediction failed at k=20" and "k=20 is nearly saturated, so no
method difference of any kind could show up" are *observationally identical here*
— the same shape of ambiguity that made k=1 unreadable, at the other end of the
axis, and I did not see it coming when writing the branches.

This is the §3.3.4 failure family again, and worth naming precisely: the gap
averages over an axis — **headroom** — that the comparison does not hold fixed.

**What adjudicates it:** `frozen_proto` at k=20, which is running now
(`run_vit_shots_ref.sh`, launched 05:30 box).

* If frozen is also ≈93–94, then at k=20 adaptation buys nothing at all on this
  dataset, the comparison between two methods of adapting is vacuous, and the
  k=20 cell says nothing about the mechanism either way. §3.5 survives untested
  at the top end rather than refuted.
* If frozen is well below (≈85 or less), then adaptation is doing real work at
  k=20 and the two methods genuinely tie. *That* is branch 2, and §3.5 becomes a
  description rather than a mechanism.

Until that number exists, the honest statement is: **the attention advantage is
established at k=5 and is not detectable at k=20, for reasons not yet
distinguished.** No claim about monotonicity in k, and specifically not the
claim I was leaning toward two ticks ago, which the ceiling would have handed me
for free.

Unchanged either way, per branch 4: the depth result needs full_ft to lose at
k=5, which it does across three draws on both datasets. It does not need the
mechanism. And note that k=5 — where the headline lives — is nowhere near a
ceiling at 83–86%.

### 3.7c The frozen reference at k=1: the prediction was mis-specified, not refuted

Measured 2026-10-21 05:35 box. **Gate first:** `frozen_proto` reports
`fraction_modified = 0.0000` with ci95 `0.0` across all 100 episodes, so it
genuinely adapts nothing and every "frozen" number in this document stands.
(Cross-checks from the same field: `attn_block8_ft` = 0.02751, matching the 2.75%
the selector is supposed to open; `attn_only_ft` = 0.3233.)

Note on the gate itself: the queue item said to "check `frac_mod` reads 0.000".
**There is no `frac_mod` field** — it is `fraction_modified`, per episode and in
`summary`. A check written against a key that does not exist passes vacuously or
errors, and mine errored, which is the lucky outcome. The shorthand came from my
own notes. Same family as §4.2, and the lesson is the same: a check must name the
field the writer actually wrote.

k=1, 100 episodes, seed 0, val, all three sharing the same episodes:

| comparison | diff | ±95% | verdict |
|---|---|---|---|
| `attn_only_ft` − `frozen_proto` | **+0.05** | 1.42 | not settled |
| `full_ft` − `frozen_proto` | **−1.20** | 1.42 | not settled |
| `attn_only_ft` − `full_ft` | +1.25 | 1.18 | settled |

**The k=1 gap is not attention winning. It is full fine-tuning losing to a frozen
backbone, with attention tying frozen to within 0.05 points.** At k=1, adapting
attention buys nothing whatsoever over not adapting at all.

This settles the ambiguity §3.7 flagged, and it settles it against the reading I
had been carrying. The k=1 row was recorded as "the support-size prediction failed
in direction". **It did not fail; the prediction was mis-specified.** §3.5's story
is that full fine-tuning over-optimises a small support set. The quantity that
story predicts is each method's *gain over frozen* — and that quantity behaves
exactly as predicted:

| k | attn − frozen | full_ft − frozen |
|---|---|---|
| 1 | +0.05 | **−1.20** |
| 5 | +6.34 | +3.85 |
| 20 | *pending* | *pending* |

Full fine-tuning's gain from adapting grows with k, and at k=1 it is *negative* —
adapting actively harms, which is over-optimisation doing precisely what the story
says. What does not grow is the **difference** between the two gains, and the
reason is now obvious: the gap is bounded above by how much attention can gain,
and at k=1 attention cannot gain anything either, because one example per class
does not support learning. Both gains go to roughly zero at k=1, so their
difference must too, whatever the mechanism.

So the prediction "the gap shrinks as k grows" was the wrong prediction to
pre-register. It is not a test of over-optimisation, because the gap is a
difference of two quantities that are *both* floored at small k and — per §3.7b —
both ceilinged at large k. The gap is the wrong statistic for this axis, at both
ends, which is why both ends came out unreadable.

**Status of the mechanism scoreboard, stated carefully.** §3.6's inner-step test
stands as a confirmation. The support-size test is now **void rather than failed**
— it measured a statistic that cannot answer the question. The gains-over-frozen
table above is the test it should have been, it points the right way at k=1 and
k=5, and it is two rows from being a real result. Recording it as "void" rather
than "confirmed" matters: I designed this replacement analysis *after* seeing the
k=1 numbers, so it is a hypothesis fitted to the data it explains, not a
prediction. Its k=20 row — frozen_proto at k=20, running now — is the first point
it has not already seen, and is therefore the only honest test of it.

Pre-registered here, before that number arrives: over-optimisation predicts
`full_ft − frozen` keeps **rising** with k, so k=20 should exceed +3.85. If it
comes in flat or lower, this replacement story is wrong too, and §3.5 goes from
void to refuted.

Unchanged: the depth result, which needs `full_ft` to lose at k=5 and not a reason.

### 3.7d The mechanism holds. The gap closes because full fine-tuning catches up

Measured 2026-10-21 06:10 box. **Gate:** `frozen_proto` reports
`fraction_modified = 0.0000`, ci95 `0.0000`, n=100 at **every** k. It adapts
nothing anywhere, so all three frozen references are sound.

All paired per-episode within each k, t(99)=1.9842, same episodes across arms:

| k | frozen | attn | full_ft | attn − frozen | full_ft − frozen | attn − full_ft |
|---|--------|------|---------|---------------|------------------|----------------|
| 1  | 62.09 | 62.15 | 60.89 | +0.05 ± 1.42 | **−1.20** ± 1.42 | +1.25 (settled) |
| 5  | 79.27 | 85.96 | 83.24 | **+6.69** ± 1.21 | +3.97 ± 1.10 | +2.72 (settled) |
| 20 | 86.35 | 94.00 | 93.76 | **+7.65** ± 0.98 | **+7.41** ± 1.04 | +0.24 (not settled) |

**The pre-registered test passes.** §3.7c fixed, before this number existed, that
over-optimisation requires `full_ft − frozen` to keep rising with k and to exceed
the k=5 value. It reads **+7.41 ± 1.04** at k=20 against +3.97 at k=5 and −1.20 at
k=1 — monotone increasing, decisively clear of the threshold.

(Threshold bookkeeping: §3.7c wrote the k=5 figure as +3.85, computed from the
three-draw means 83.07 − 79.22. The properly paired seed-0 value against the
seed-0 frozen run is +3.97. Different draw counts, same comparison; +7.41 clears
either, and the two must not be quoted interchangeably.)

**§3.7b's ceiling explanation is refuted.** Frozen at k=20 is **86.35**, not
93–94. Adaptation is doing substantial work at k=20 — both methods gain over
7 points — so the k=20 null is not a saturation artefact. The two methods gain
*the same amount*, which is a real finding rather than a measurement failure.

So the whole shape resolves, and it resolves in favour of the mechanism:

* **k=1** — one example per class supports no learning. Attention gains nothing
  (+0.05). Full fine-tuning is *actively harmed*, landing 1.20 **below** a frozen
  backbone. That is over-optimisation at its most extreme.
* **k=5** — attention gains a lot (+6.69); full fine-tuning gains much less
  (+3.97) because it over-fits the support set. The headline +2.72 is this
  difference.
* **k=20** — attention has saturated (+6.69 → +7.65, a further +0.96 only);
  full fine-tuning has **caught up** (+3.97 → +7.41) now that the support set is
  large enough to fit its parameter count. The gap closes to nothing.

**The gap closes because full fine-tuning catches up, not because attention
degrades.** Attention never gets worse at any k. That is precisely what
over-optimisation predicts: the penalty for carrying many adaptable parameters
shrinks as the support set grows, and vanishes once there is enough data to
constrain them.

Note what this does to the original prediction. "The gap shrinks as k grows" was
**correct for k=5 → k=20** and the k=1 point was a floor artefact, not a
counterexample — both gains are near zero there, so their difference must be too.
The gap is non-monotone because two *different* effects bound it at the two ends,
which is why §3.5 looked refuted from the gap alone and looks confirmed from the
gains.

**Mechanism scoreboard: 2 for 2, with one honest qualification.** §3.6's
inner-step test and this support-size test both confirm. The qualification is
that the gains-over-frozen statistic was chosen *after* seeing k=1 and k=5, so
only the **k=20 row** was out-of-sample for it — one pre-registered point,
confirmed. That is a genuine test, not a fitted one, but it is one point, and a
second dataset would make it an argument rather than an instance.

**What this buys the paper beyond a rescued section.** It converts §3.5 from a
hand-wave into a statement about *when the method is worth using*: the advantage
is specific to the regime where the support set is too small to constrain full
fine-tuning but large enough for a small, well-placed group to learn from. Below
that window nothing adapts; above it, everything does, equally. That is a usable
claim — it tells a practitioner which regime they are in — and it predicts the
method's own obsolescence at large k, which is the kind of boundary a method
paper is usually accused of hiding.

**Still not established:** this is EuroSAT only. ISIC's support-size behaviour is
unmeasured, and §3.3.5 already showed ISIC inverts the depth story, so it cannot
be assumed to share this one. One dataset, three k values, one draw each at k=1
and k=20.

### 3.7e The noise floor is entirely in the optimiser path

Found by accident, worth keeping. Loading the frozen references tripped the
figure loader's duplicate check: `frozen_proto` at EuroSAT k=5 seed 0 exists in
**both** `vit_2b_eurosat` and `vit_shotsref5`. Two independent runs of a
byte-identical configuration, months apart in wall-clock, on the same card.

They agree to **+0.00 ± 0.00**, paired, bit-for-bit across all 100 episodes.

That is a fourth independent probe of reproducibility, and it is not the same
quantity as the earlier three (0.55, 0.35, 0.21). `frozen_proto` takes **no
gradient steps**. So this measures forward pass plus episode sampling alone, and
finds them **exactly** reproducible. Which locates the ~0.6-point floor
precisely: it is **entirely in the backward/optimiser path** — gradient reduction
order amplified through Adam — and none of it is in the sampler or the forward
pass.

Three things follow:

1. **Any frozen number in this document is exactly reproducible.** The frozen
   references are not noisy quantities and do not need the 0.6 allowance.
2. **The episode sampler is perfectly deterministic**, which is the assumption
   every paired comparison in this document rests on (§2 fact 1). It had been
   argued from the code; it is now measured.
3. **Queue item 8 (determinism) has a sharper target.** The question is not
   "make the pipeline deterministic" but "make the backward pass deterministic",
   which is exactly what `torch.use_deterministic_algorithms` plus
   `CUBLAS_WORKSPACE_CONFIG` address, and it is now known that nothing else needs
   touching. The cost measurement still has to be run and the choice is still
   AJ's.

The duplicate itself is resolved by not loading `vit_shotsref5`: the main corpus
already has k=5 frozen over three draws, which is better than one, and the exact
agreement proves nothing is lost by dropping the second. The guard that caught it
is the same one added after ISIC silently overwrote EuroSAT — this is the second
time it has earned its place, and the first time what it caught was benign.

### 3.8 Weight decay, part 1: the wd=0 gate passes

Checked 2026-10-21 07:30 box, with the sweep still running. The gate exists
because adding per-parameter-group weight decay changed how the inner optimiser
is constructed, and `test_zero_weight_decay_is_a_single_group` asserts that at
wd=0 it collapses back to the single-group Adam every existing number was
measured with. An assertion about construction is not a measurement of
behaviour, hence this check.

Paired per-episode against `vit_2b_eurosat`, same seed, same episodes, same
tuned learning rate on both sides:

| baseline | wd=0 | reference | paired difference | verdict |
|---|---|---|---|---|
| `full_ft` | 83.56 | 83.24 | +0.32 ± 0.94 | indistinguishable |
| `attn_only_ft` | 85.36 | 85.96 | **−0.60 ± 1.03** | indistinguishable |

**The gate passes and the param-group change is safe.** Every number measured
before it stands.

Worth recording how nearly this went the other way for the wrong reason. The
gate as written in the queue said "must reproduce within 0.6", and
`attn_only_ft` came in at **exactly −0.60** — at the threshold, not inside it, if
read off the two means. Read correctly, paired, it is −0.60 ± 1.03 and nowhere
near significant. The 0.6 figure is a *floor for claiming an effect*; the
question a reproduction check asks is the different one of whether a difference
is consistent with nondeterminism, and the paired interval answers that directly.
Comparing two means against the floor would have raised a false alarm here, or —
worse, in the other direction — passed a real regression that happened to land at
0.55.

A concrete picture of §3.7e while we are here. Episodes identical to the bit
between the two runs: `full_ft` **14/100**, `attn_only_ft` **7/100**. Against
`frozen_proto`'s **100/100**. So optimiser nondeterminism perturbs roughly nine
episodes in ten individually, and still averages out to inside a point — which is
both why the floor is ~0.6 rather than 0 and why per-episode pairing buys as much
as it does: the perturbation is nearly independent across episodes, so it shrinks
like √n in the mean while the real effect does not.

wd=1e-4 and the rest of the sweep are still running.

### 3.9 Weight decay, part 2: the prediction FAILS

Measured 2026-10-21 08:20 box (wd=1e-2 still running). Paired per-episode
against each baseline's own wd=0 arm, same tuned lr on both sides, t(99).

**The prediction, on record in `run_vit_wd.sh`'s header before any of this ran:**
regularising full fine-tuning helps it; weight decay does not help attention.

| baseline | wd=0 | wd=1e-4 | wd=1e-3 |
|---|---|---|---|
| `full_ft` | 83.56 | −0.31 ± 0.89 | **−1.08 ± 0.89 (settled)** |
| `attn_only_ft` | 85.36 | −0.12 ± 1.01 | +0.61 ± 0.99 |

The second half holds: weight decay does not help attention at any level tested.
**The first half fails in direction.** Weight decay does not help full
fine-tuning — it *hurts* it, by a settled 1.08 points at 1e-3, and the trend
across 0 → 1e-4 → 1e-3 is monotone downward.

**Recorded as FAILED, not void.** There is a principled reason the test may be
invalid, and it is important not to let that reason do the work a new measurement
should do. The reason: weight decay pulls weights toward **zero**, and the
pretrained weights are nowhere near zero. So in a fine-tuning inner loop,
decay-toward-zero does not implement "stay close to the prior" — it actively
destroys the pretrained features that are the entire reason for starting from a
checkpoint. The regulariser that expresses §3.5's intent is L2 toward the
**pretrained weights** (ℓ²-SP), not toward the origin.

That is a real confound and I should have caught it when designing the sweep
rather than when reading it. But it does not rescue the prediction, for the same
reason §3.7d needed its own out-of-sample point: an explanation produced after
seeing a failure is a hypothesis, not a result. **The pre-registered test failed.
The ℓ²-SP version is a different experiment that has not been run.** Only running
it can change this entry.

**Mechanism scoreboard: 2 for 3.**

* §3.6 inner-step test — **confirmed**.
* §3.7d support-size test — **confirmed**, on the gains-over-frozen statistic,
  with one genuinely out-of-sample point (k=20).
* §3.9 regularisation test — **failed**.

A weak post-hoc consistency worth naming and not leaning on: weight decay hurts
`full_ft` (−1.08) more than `attn_only_ft` (+0.61), which is what you would expect
if the damage scales with how many pretrained parameters are being dragged toward
zero. That is consistent with the over-optimisation picture, but it is a story
fitted to the data and carries no weight as evidence.

**What this costs.** §3.5 keeps its two confirmations and loses this one. The
honest summary of the mechanism is now: more inner optimisation helps the small
group and hurts the large one (§3.6), and the large group's deficit shrinks as
the support set grows (§3.7d) — but the one intervention that should have
*repaired* full fine-tuning instead made it worse, and we do not yet know whether
that is because the mechanism is wrong or because the intervention was.

Unchanged: the depth result, which rests on `full_ft` losing at k=5 and not on
any of this.

## 4.2 Two fields named `inner_lr`, and which one is real

Recorded 2026-10-21 04:30 box, after it cost ten minutes and a false alarm.

Reading `manifest.config.inner_lr` from the k=20 `full_ft` measurement gives
**1e-04**, which is *attention's* tuned optimum, not full_ft's 3e-05. For a few
minutes it looked as though the `--tuned` path was measuring every baseline at the
wrong learning rate — i.e. that every headline number in the project was suspect.

**It is not.** `manifest.config` is `vars(args)`, so its `inner_lr` is the CLI
default echoed identically into every file. The rate actually used is the
**top-level** `inner_lr`, written from `runner.inner_lr` after the tuned setting is
applied (`experiments/run_baselines.py:266`), alongside a top-level `tuned` flag.
The k=20 file reads top-level `inner_lr: 3e-05, tuned: true` — correct.

The decisive check, worth keeping because it settles the question in one look:
in `results/vit_2b_eurosat` all eight baselines report
`manifest.config.inner_lr == 1e-04`, while their top-level values range over
1e-06 to 0.03. A single shared value across eight baselines that were
demonstrably tuned to different rates can only be a default.

So the authoritative field is the top-level one, the misleading one is nested
exactly where a reader looks for configuration, and **nothing should ever read a
learning rate out of `manifest.config`.** `tests/test_results_provenance.py` pins
this, and would fail loudly if the two ever silently converged.

Two things that check bought beyond the scare:

* Every one of 172 committed result files records both its effective `inner_lr`
  and its `tuned` flag, so no measurement in the corpus is of unknown provenance.
* **No committed result was measured on the test split.** That discipline now
  rests on a test rather than on anyone remembering it, which is the right place
  for it given it is the one thing in this project that cannot be undone.

Note what this episode was *not*: it was not a wrong number in the document. It
was a wrong field read by me, caught in the same turn, before anything was
written down. The reason it is recorded anyway is that the next reader will look
in the same place for the same reason.

## 5. Standing cautions inherited from the ResNet arm

Three of the first four results in this project were single-draw artefacts, and
`RESULTS_SUBSPACE.md` §7 establishes a ±1.1-point estimator noise floor on
anything derived from an extracted displacement. Before any number here is
written up: check how many draws it rests on, whether its margin clears that
floor, and that the selector opened the parameter set it was supposed to. The last
of those has already failed twice — `bn_only_ft` once ran as full fine-tuning for
a whole sweep, and `last_layer_ft` resolved on ViT to the 1.5k-parameter pre-head
LayerNorm, which would have been a near-frozen baseline reporting a plausible
accuracy. Both are now asserted in `tests/test_selectors_vit.py`.
