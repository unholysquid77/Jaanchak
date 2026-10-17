# What adaptation actually does — probes on ISIC, then EuroSAT

> Read order note: sections 1-4 are the original ISIC probes and are **superseded
> in places**. Section 5 retracts the precomputed-correction magnitude. Section 6
> adds EuroSAT and retracts how sections 1 and 3 were worded: the diagonal basis
> does not win in general, it wins when the shift is large. Section 7 supersedes
> section 5's headline: over three extraction draws the precomputed correction
> recovers 48% of ISIC's headroom once layer4 is excluded, not the 15% section 5
> reports for the all-unit version, and the estimator noise floor is +/-1.1 points
> rather than 0.56.

**Validation split only. No test numbers here.** ~3 GPU-hours, 2026-10-16.
Raw output: `results/subspace/`, `results/unit_benefit/`, `results/isic_groups/`.
Driver: `scripts/run_probe.sh`. ISIC 5-shot, `group_disjoint` by lesion at
0.1/0.4/0.5, split seed 0, episode seed 10,000, ResNet-50, 20 inner steps.

1-shot is excluded throughout and should stay excluded: every ISIC baseline lands in
27.6–30.9 against a 20% chance floor, so it cannot separate methods.

---

## 0. Why these were run

Level 1's gate is null on EuroSAT and worse than null on ISIC, and `bn_only_ft`
adapts 0.22% of the backbone to tie full fine-tuning (49.44 vs 49.55). Read together
that suggests the binding constraint in few-shot adaptation is the **dimensionality**
of the update, not its location — which the gate never addressed, since it modulates
the magnitude of a 23M-parameter update estimated from 25 images.

BatchNorm affine is a per-channel scale and shift: a *diagonal* affine map on each
unit's activations. So `bn_only_ft` is already an activation-subspace method. These
probes test that reading.

## 1. Adaptation is ~90% a per-channel affine, and near rank-1 at the top

`experiments/diagnose_subspace.py`, 200 episodes. Each unit's measured displacement
(adapted vs pretrained activations on the same probe batch — the query set, under
`no_grad`, never seen by the optimiser) is split into the best per-channel affine plus
a residual. Shift and scale together are exactly what BatchNorm can express.

```
                          full_ft (49.61)    bn_only_ft (48.69)
diagonal-expressible           0.903              0.946
non-diagonal residual          0.097              0.054
mean effective rank           12.84              10.20     (of 200 episodes)
```

Per unit, `full_ft`:

```
unit        shift   scale   resid   eff.rank   top1
stem        0.756   0.097   0.150     13.15   0.164
layer1_2    0.916   0.023   0.071     17.03   0.180
layer2_0    0.900   0.035   0.080     20.24   0.159
layer3_5    0.770   0.100   0.172      8.17   0.341
layer4_0    0.637   0.212   0.209      2.65   0.613
layer4_1    0.538   0.294   0.335      2.93   0.582
layer4_2    0.453   0.437   0.296      1.51   0.813
```

Two things to take from this. **Nine-tenths of what fine-tuning does to the
representation is a per-channel affine** — the mechanism behind the `bn_only_ft` tie,
measured rather than asserted. And **the top of the network is near rank-1 across 200
different tasks**: `layer4_2`, the features the ProtoNet head consumes, has effective
rank 1.51 with 81.3% of displacement energy in a single direction.

Effective rank is the participation ratio `(Σs²)²/Σs⁴`, so no threshold is chosen.
`ncomp` in the raw output is `min(episodes, channels)`, so deep units are
episode-limited rather than channel-limited.

Sanity check the measurement passes: `bn_only_ft`'s residual is 0.054 rather than
0.000 even though it *is* diagonal, because the displacement at unit *l* composes
BatchNorm changes from units 1…*l* through nonlinearities. That is the expected
behaviour, and its absence would have meant the decomposition was wrong.

## 2. Nothing for a layer-wise controller to control

`experiments/diagnose_unit_benefit.py`, 30 episodes. Per-unit realised benefit by
direct intervention: `only` adapts a single unit, `loo` adapts all but one. Signals
are read after 5 warmup steps, because at step 0 the two branches are the same network
and drift is identically zero.

```
adapt nothing 38.22      adapt all 50.22

                  vs standalone   vs marginal (loo)   WITHIN-episode
utility             -0.033            +0.146             +0.039
drift               +0.141            -0.009             -0.038
```

**The within-episode column is the one that matters**, because ranking units *inside*
an episode is exactly what a gate does. Both are zero. The pooled ρ ≈ 0.14 with small
p-values is cross-episode structure, not unit ranking — the sort of thing that looks
like a result and is not.

**Drift is confirmed a depth index: ρ = +0.973 ± 0.023** across 30 episodes. So
`−β·D` is a stage-4 detector, i.e. a fixed "freeze the top" depth schedule wearing a
feedback controller's costume — and the baselines say the top is the most productive
thing to adapt. That is a mechanism for drift's negative marginal, not just its size.
Utility's depth ρ is −0.085 ± 0.506: not a depth proxy, simply unpredictive.

**Units are highly redundant.** Standalone benefits sum to **74.5%** while adapting
everything gains **12.0%**. Removing any single unit costs 0.4–2.4%; adapting almost
any single unit alone buys 2–8%. Layer selection cannot matter much when layers are
this interchangeable — which is a direct measurement against ARC's premise that *where*
you adapt dominates *how much*.

## 3. The parameter-type plateau

`experiments/sweep_lr.py` over four new group selectors, 40 configs, each at its own
tuned learning rate on the 3e-6 … 1e-1 grid.

```
                   acc     params      gain   pts per % param      lr
frozen_proto     38.00     0.000%     +0.00          —             —
bn_scale_ft      48.79     0.113%    +10.79        95.5          3e-2
bn_shift_ft      48.32     0.113%    +10.32        91.3          3e-2
bn_only_ft       49.44     0.226%    +11.44        50.6          3e-2
downsample_ft    49.80    11.811%    +11.80         1.0          1e-3
conv_only_ft     49.64    87.996%    +11.64         0.1          1e-4
full_ft          49.55   100.000%    +11.55         0.1          1e-4
stage4_ft        46.52    63.658%     +8.52         0.1          3e-4
```

**Every group lands in 48.3–49.8.** Half of BatchNorm — 26,560 parameters, 0.113% —
captures 10.79 of the 11.55 points full fine-tuning gets, i.e. **93% of the benefit at
about 1/900th of the parameters**.

Call this a **plateau, not a frontier.** The groups are indistinguishable in accuracy
and differ only in cost. `stage4_ft` being the *worst* at 63.7% of parameters makes the
point sharply: parameter count does not predict accuracy either. This corroborates §2's
redundancy result from a completely different direction — almost any sufficiently
expressive slice of the network reaches the same ~49.5.

### A prediction that failed, recorded as such

§1 shows shift carrying ~85% of the diagonal energy, so the prediction was that
`bn_shift_ft` would clearly beat `bn_scale_ft`. Measured: **48.32 vs 48.79, shift loses
by 0.47.** Within noise, but certainly not the predicted clear win.

So **energy share does not translate into accuracy share**: the large-norm component of
the displacement is not the one doing the classification work. This is a genuine caveat
on the subspace argument and it needs explaining before anything is built on a
basis chosen by energy. One candidate: the cosine ProtoNet head normalises embeddings,
so a large common-mode shift at the top is partly divided out — exactly the component
§1 measures as dominant. If that is right, a basis should be selected by *effect on the
head's decision*, not by displacement norm, and §1's spectrum is measuring the wrong
objective. Untested.

## 4. Where this leaves the project

**Established (val, ISIC, 5-shot):**
- 90% of fine-tuning's representational change is a per-channel affine; at the top of
  the network it is near rank-1 across 200 tasks
- neither Level-1 signal predicts, within an episode, which unit is worth adapting
- drift is a depth index (ρ = 0.973), which explains its harm rather than just sizing it
- units are redundant, and every parameter group plateaus at ~49.5
- 0.113% of parameters gets 93% of the available gain

**The synthesis these support:** on a domain-shifted target, "few-shot adaptation" is
mostly a **low-rank, largely task-independent, per-channel domain correction** rather
than task-specific representation learning. That one claim accounts for all three
nulls: the `bn_only_ft` tie, the irrelevance of layer selection, and the uselessness of
a task-conditional controller.

## 5. Tested since, and a correction to §4 (2026-10-16, later)

**The precomputed correction works partially, and the diagonal beats a non-diagonal
basis.** `results/precomputed`, `results/precomputed_v2`, `results/frontier`:

```
variant   trainable   accuracy     gain
identity          0     38.04     +0.01
fitted           17     42.59     +4.55
rank1            17     42.67     +4.63
rank4            68     45.13     +7.09   <- non-diagonal peak
rank16          272     44.79     +6.75
ssf           30336     48.61    +10.57   <- diagonal
```

A non-diagonal activation basis from displacement SVD does not beat per-channel
scale-and-shift, and extra rank does not help — it peaks at rank 4 and declines. That
answers the matched-budget question a literature review names as the field's most
direct missing ablation, and the saturation is what §3's mechanism predicts: the basis
is energy-ordered, `layer4_2` holds 81.7% of its displacement energy in one direction,
and `layer4` is the region measured as useless to harmful. Rank-k spends its first
components on the least useful direction available.

**⚠ §4's headline claim is overstated and must not be repeated as written.** An earlier
run put the precomputed `shift` at 45.21, i.e. "65% of adaptation with zero gradient
steps". That number is a favourable draw. **Extraction is not reproducible**:
`experiments/diagnose_extraction_variance.py` extracts twice in one process — same
seeds, same episodes, same weights — and gets

```
cosine between draws   0.948 (stem) .. 0.999 (layer4_2)
relative difference    0.32  (stem) .. 0.04  (layer4_2)
resulting accuracy     38.23  vs  40.21        spread 1.99 points
```

Both draws sit nearer 38.5 than 45.2, so the correction recovers materially less than
65%, and the true figure needs a mean over extraction seeds with an error bar. The
prime suspect is AMP's `GradScaler` skipping optimiser steps on inf/nan
nondeterministically, which changes how much adaptation each episode actually performed;
`EpisodeRunner` already counts `skipped_steps` for that reason.

Worse, the variance is largest exactly where the signal is. Relative difference falls
monotonically with depth because early-layer displacements are small in absolute norm
(0.11 against 10.3), so noise dominates — but the gain lives in layer2–3, not layer4.
The useful part of the correction is the least stably estimated part.

Everything shift-derived inherits this, rank-k included, since its coefficients
warm-start from the same mean displacement.

### Resolved: AMP is most of it, and the precomputed correction is weak

Rerunning the variance test with AMP off:

```
                      AMP on        AMP off
spread between two    1.99 pts      0.56 pts
identical extractions (38.23/40.21) (40.08/39.52)
```

So `GradScaler` skipping optimiser steps on inf/nan accounts for roughly 3.5x of the
irreproducibility. The residual 0.56 is cuDNN nondeterminism and is tolerable.
Extraction should therefore always run with `--no-amp`, at roughly double the
wall-clock; `scripts/run_eurosat_probe.sh` does.

**And this forces a second, larger retraction.** With extraction stabilised, the
precomputed correction sits at **~39.8** against a frozen backbone's 38.04 and full
fine-tuning's 49.6. That is **1.8 of the 11.5 available points, about 15% — not 65%.**
A precomputed, zero-gradient-step correction is close to useless here. The 45.21 that
produced the 65% figure was an outlier draw, and the two AMP-on draws (38.23, 40.21)
and two AMP-off draws (40.08, 39.52) all agree with the lower value.

This also reverses the "fitting hurts" reading from `results/precomputed_v2`. That
comparison used the inflated 45.21 as its baseline. Against the correct ~39.8, every
fitted variant *helps*: `fitted` 42.59, `rank4` 45.13, `ssf` 48.61. `subspace_frontier`
was internally consistent the whole time (shift 38.55 < fitted 42.59); it was
`precomputed_shift`'s draw that was the outlier.

**What survives unaffected**, because none of it depends on the extracted basis:
~90% of displacement is a per-channel affine; `ssf` (diagonal, fit per episode) beats
every non-diagonal rank at 48.61 against 45.13; energy does not predict accuracy; and
the Level-1 gating null and its diagnosis. The frontier's 3.5-point diagonal advantage
is six times the residual extraction spread, so that conclusion is safe without a
rerun.

**The corrected story:** adaptation's displacement is overwhelmingly diagonal, it
cannot be usefully precomputed, per-episode fitting is necessary, and within
per-episode fitting the diagonal form beats a low-rank non-diagonal basis extracted
from that displacement.

**What these probes do *not* support:** that a cleverer basis beats the diagonal. §3's
plateau says the ~49.5 ceiling is not the diagonal's fault, so "better basis" is
probably not where the win is. The win, if there is one, is in removing test-time
optimisation, not in raising the ceiling.

## 6. The EuroSAT arm: the structure replicates, the *ranking* does not

Everything above rests on ISIC. EuroSAT is the useful contrast, not a second
confirmation: a frozen ResNet-50 already reaches 83.85% there at 5-shot against
full_ft's 86.83%, so the headroom is **2.98 points** versus ISIC's 11.5. If the
findings are about few-shot adaptation they should survive; if they are about
medical imaging, or about large shifts, they should not.

Same protocol, EuroSAT's own 0.2/0.1/0.7 split and full_ft's own tuned
`--inner-lr 1e-5` (deliberately not ISIC's 1e-4), 200 eval episodes, `--no-amp`
throughout. Val split only.

### 6.1 What replicated

**Displacement is still mostly diagonal, and still gets less so with depth.**
Mean diagonal fraction 0.824 (ISIC ~0.90), mean effective rank 8.55. The depth
profile is the same shape: layer1–2 sit at 0.89–0.91 diagonal, layer3 decays
0.885 → 0.766, layer4 bottoms out at 0.68–0.72. The stem is the outlier at
0.643, as on ISIC. So "adaptation is approximately a per-channel affine" is a
property of few-shot adaptation, not of skin lesions.

**The parameter-type plateau replicated, harder.** Each group at its own tuned
learning rate:

| group | % of backbone params | val acc |
|---|---|---|
| `bn_shift_ft` | 0.113% | 86.44 |
| `bn_scale_ft` | 0.113% | 86.72 |
| `downsample_ft` | 11.8% | 87.01 |
| `conv_only_ft` | 88.0% | 87.19 |

A 0.75-point band across a **778×** range of trainable parameters, with CIs of
±0.9. `conv_only_ft` at its own tuned lr even edges past full_ft's 86.83. On
ISIC every group plateaued near 49.5; here near 86.8. Two datasets, two shift
magnitudes, same answer: *which* parameters you adapt is very nearly irrelevant,
and the plateau is the ceiling.

### 6.2 What did not replicate — and this one matters

On ISIC, `ssf` (30,336 params) beat the energy-ordered non-diagonal basis by 3.5
points at every rank — six times the residual extraction spread, so it was safe.
On EuroSAT that ordering **inverts**, and more importantly the whole spread
collapses:

| variant | trainable | val acc | share of the 2.98-pt headroom |
|---|---|---|---|
| `identity` (frozen) | 0 | 83.85 ± 0.91 | 0% |
| `shift` (precomputed, 0 steps) | 0 | 84.95 ± 0.96 | **37%** |
| `fitted` | 17 | 85.76 ± 0.90 | 64% |
| `rank1` | 17 | 86.10 ± 0.91 | 76% |
| `rank2` | 34 | 86.01 ± 0.90 | 72% |
| `rank4` | 68 | 86.15 ± 0.88 | 77% |
| `rank8` | 136 | 86.14 ± 0.88 | 77% |
| `rank16` | 272 | 86.35 ± 0.87 | 84% |
| `ssf` | 30,336 | 85.59 ± 0.87 | 58% |
| `full_ft` | 23.5M | 86.83 | 100% |

Read honestly: `ssf` is 0.76 points *below* `rank16`, which is inside both CIs,
so the claim is **not** "non-diagonal wins on EuroSAT". The claim is that under
a mild shift **nothing wins** — every parameterisation from 17 to 30,336
parameters lands in an 85.6–86.4 band narrower than its own confidence
interval, and `ssf`'s 1784× larger budget buys literally nothing over
`rank1`'s seventeen scalars. On ISIC that same budget bought 3.5 points.

So the ISIC result was never "the diagonal basis is the right basis". It was
"when the shift is large enough that 11.5 points are on the table, you need the
capacity, and the diagonal directions are where that capacity pays". Under a
3-point shift the capacity is dead weight. That is a budget × shift-magnitude
interaction, and it is a better claim than the single-dataset one it replaces —
but it is also a retraction of how §1 and §3 were worded.

There is a mechanism available, not yet a tested one: EuroSAT's displacement is
82.4% diagonal and ISIC's is ~90%, and the basis that wins is the one aligned
with the larger share. One extra dataset would make that a hypothesis with a
prediction rather than a post-hoc pairing; ChestX (a harder shift than ISIC)
should push diagonal share *up* and widen `ssf`'s margin, CropDisease (milder)
should do the opposite. Not run.

### 6.3 One earlier reading that held up

§5 retracted the precomputed correction down to ~15% of ISIC's headroom and I
guessed it would do proportionally *better* where the shift is mild. It does:
**37% of EuroSAT's headroom with zero gradient steps**, against 15% on ISIC.
Same estimator, same code, 2.5× the share. A precomputed per-channel affine
handles mild covariate shift and fails on genuine domain shift — which is a
cleaner statement of what it is than "recovers most of adaptation for free"
ever was.

### 6.4 Status

Raw: `results/subspace_eurosat/`, `results/frontier_eurosat/`,
`results/eurosat_groups/`, log in `logs/eurosat_probe.log`. ~5 GPU-hours.

## 7. Three extraction draws, and layer4 was the problem all along

§5 above retracts a claim built on one favourable extraction draw, then states its
replacement (~39.8, "about 15%") from one draw too. Same mistake, smaller number.
So: three extraction draws, `--extract-seed 0/1/2`, `--seed 0` held fixed so all
three are scored on **identical** evaluation episodes, `--no-amp`. The `identity`
control reproduced frozen_proto to 38.04 with delta 0.00 on every draw, so the
harness is sound and what follows is estimator variance, not a bug.

| unit subset | draw 0 | draw 1 | draw 2 | mean | sd | share of 11.56-pt headroom |
|---|---|---|---|---|---|---|
| `all` | 40.09 | 38.37 | 40.37 | 39.61 | 1.08 | 14% |
| `layer4` only | 39.26 | 39.23 | 39.39 | 39.29 | 0.09 | 11% |
| **`no_layer4`** | **44.77** | **42.47** | **43.36** | **43.53** | 1.16 | **48%** |
| `stem_layer1` | 38.98 | 38.86 | 38.65 | 38.83 | 0.17 | 7% |

### 7.1 The 15% figure survives, and is the wrong number to quote

`all` comes out at 39.61 ± 1.08 — 14%, so §5's "about 15%" was right. But the
spread between draws is **2.00 points**, not the 0.56 that §5 calls the AMP-off
floor and "tolerable". That 0.56 came from two draws; three give 2.00. The floor
measurement was itself underpowered, which is the same failure one level down.
Taken alone, `all` at 39.61 ± 1.08 against frozen's 38.04 ± 0.97 is a gain whose
confidence interval nearly touches zero.

### 7.2 But excluding layer4 nearly triples it

`no_layer4` beats `all` on every draw, and the orderings do not overlap at all:
the *worst* `no_layer4` draw (42.47) is above the *best* `all` draw (40.37). Mean
43.53 against frozen 38.04 is **5.49 points, 48% of the headroom** — not 15%.

Dropping 17 of 23.5M parameters' worth of correction — three residual units —
more than triples what the correction recovers. This is §1's "layer4 is
useless-to-harmful" finding reappearing in a completely different experiment, and
it says the precomputed correction was never as weak as §5 claims; it was being
dragged down by the three units holding 81.7% of the displacement energy. Energy
does not predict accuracy, again, and this time it costs 34 points of headroom
share to ignore.

So the honest statement is: **a precomputed zero-gradient-step correction over
stem–layer3 recovers about half of what full fine-tuning achieves on ISIC, and
including layer4 destroys two thirds of that.** The twice-retracted number was
wrong in the *other* direction the second time.

### 7.3 Where the variance lives, which is the real caution

The per-subset standard deviations are the most useful thing here:

- `layer4` sd **0.09**, `stem_layer1` sd **0.17** — essentially reproducible.
- `all` sd **1.08**, `no_layer4` sd **1.16** — not.

Both unstable subsets contain layer2–3 and both stable ones exclude it. So the
irreproducibility is localised to layer2–3, and layer2–3 is exactly where the gain
is. That is the same conclusion §5 reached from per-unit relative differences,
arrived at independently from accuracy alone. It means every shift-derived result
in this document carries roughly ±1.1 points of estimator noise on the parts that
matter, and the 0.56 "tolerable residual" in §5 should be read as ±1.1 instead.

Anything resting on a margin under ~2.3 points (two sd) between shift-derived
variants needs more draws before it is a finding. The frontier's 3.5-point
diagonal advantage on ISIC clears that. EuroSAT's 0.76-point `ssf`/`rank16` gap in
§6 does not, which is a second reason that section claims no winner there.

Raw: `results/precomputed_seed{0,1,2}/`, log `logs/seed_spread.log`. ~2 GPU-hours.
